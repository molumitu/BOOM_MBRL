import math
from typing import List, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import TensorDataset, DataLoader


def _bpograd(g1, g2, g0_coeff=0.5, bpo_coeff=0.5):
    """
    Optimize gradients based on given tasks and BPO coefficient.

    Args:
        g1 (torch.Tensor): Gradient of the first task.
        g2 (torch.Tensor): Gradient of the second task.
        g0_coeff (float): Coefficient for computing g0 = g0_coeff * g1 + (1-g0_coeff) * g2.
        bpo_coeff (float): Coefficient for BPO.

    Returns:
        torch.Tensor: Optimized gradient.
    """
    gradients = torch.stack((g1, g2), dim=0)  # shape:[num_grads, dim]
    covariance_matrix = gradients.mm(gradients.T)
    device = gradients.device
    g0_coeffs = torch.tensor([g0_coeff, 1-g0_coeff], dtype=torch.float32).reshape(2, 1).to(device)
    g0_square = (g0_coeffs.T.mm(covariance_matrix).mm(g0_coeffs) + 1e-4).squeeze()
    g0_norm = (g0_square).sqrt().squeeze()
    mean_covariance = (g0_coeffs.T.mm(covariance_matrix)).T  # [g0^T*g1 , g0^T*g2]

    weights = torch.zeros(2, 1, dtype=torch.float32).to(device)
    weights.requires_grad_()
    optimizer = torch.optim.Adam([weights], lr=1)
    best_weights = None
    best_objective = np.inf
    
    iter_num = 20
    for iteration in range(iter_num+1):
        optimizer.zero_grad()
        softmax_weights = torch.softmax(weights, 0).to(device)
        objective = (softmax_weights.T.mm(mean_covariance) + 
                     g0_norm * bpo_coeff * (softmax_weights.T.mm(covariance_matrix).mm(softmax_weights) + 1e-4).sqrt())

        if objective.item() < best_objective:
            best_objective = objective.item()
            best_weights = weights.clone()
        
        if iteration < iter_num:
            objective.backward()
            optimizer.step()

    softmax_best_weights = torch.softmax(best_weights, 0).to(device)
    gradient_norm = (softmax_best_weights.T.mm(covariance_matrix).mm(softmax_best_weights) + 1e-4).sqrt().squeeze()

    lambda_param = g0_norm * bpo_coeff / (gradient_norm + 1e-4)
    optimized_gradient = (g0_coeffs + softmax_best_weights * lambda_param).T.mm(gradients).squeeze(0)  # 去除多余维度
    
    weight_1, weight_2 = (g0_coeffs + softmax_best_weights * lambda_param)
    weight_1, weight_2 = float(weight_1.detach()), float(weight_2.detach())
    return optimized_gradient, weight_1, weight_2
    
def _apply_grad_vector_to_params(
    model_params: Iterable[torch.Tensor], grad_vector: torch.Tensor, accumulate: bool = False
):
    """Apply gradient vector to model parameters.

    Args:
        model_params (Iterable[torch.Tensor]): Iterable of model parameter tensors.
        grad_vector (torch.Tensor): A single vector representing the gradients.
        accumulate (bool): Whether to accumulate the gradients or overwrite them.
    """
    # Ensure grad_vector is of type Tensor
    if not isinstance(grad_vector, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got: {type(grad_vector).__name__}")

    # Pointer for slicing the gradient vector for each parameter
    pointer = 0
    for param in model_params:
        num_elements = param.numel()
        # Slice the vector and reshape it to match the parameter's shape
        if accumulate:
            param.grad = (param.grad + grad_vector[pointer:pointer + num_elements].view_as(param).data)
        else:
            param.grad = grad_vector[pointer:pointer + num_elements].view_as(param).data

        # Update the pointer
        pointer += num_elements

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], task1_out_dim: int, task2_out_dim: int):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.shared_backbone = nn.Sequential(*layers)
        
        self.task1_head = nn.Linear(in_dim, task1_out_dim)
        self.task2_head = nn.Linear(in_dim, task2_out_dim)
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shared_features = self.shared_backbone(x)
        task1_pred = self.task1_head(shared_features)
        task2_pred = self.task2_head(shared_features)
        return task1_pred, task2_pred

def custom_loss1(pred1: Tensor, target1: Tensor) -> Tensor:
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(pred1, target1)

def custom_loss2(pred2: Tensor, target2: Tensor) -> Tensor:
    mse_loss = nn.MSELoss()
    return mse_loss(pred2, target2)

def extract_grad_vector(model_params: Iterable[torch.Tensor]) -> Tensor:
    grad_list = []
    for param in model_params:
        if param.grad is None:
            grad_tensor = torch.zeros_like(param, dtype=torch.float32).view(-1)
        else:
            grad_tensor = param.grad.detach().view(-1)
        grad_list.append(grad_tensor)
    return torch.cat(grad_list)

def train_mlp_with_fused_gradient():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 10
    hidden_dims = [64, 32]
    task1_out_dim = 5
    task2_out_dim = 1
    batch_size = 32
    epochs = 100
    lr = 0.001
    g0_coeff = 0.5
    bpo_coeff = 0.5

    model = MLP(input_dim, hidden_dims, task1_out_dim, task2_out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_samples = 1000
    x = torch.randn(num_samples, input_dim, dtype=torch.float32).to(device)
    task1_target = torch.randint(0, task1_out_dim, (num_samples,)).to(device)
    task2_target = torch.randn(num_samples, task2_out_dim, dtype=torch.float32).to(device)

    dataset = TensorDataset(x, task1_target, task2_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_model_params = sum(p.numel() for p in model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_fused_loss = 0.0

        for batch_idx, (batch_x, batch_t1, batch_t2) in enumerate(dataloader):
            task1_pred, task2_pred = model(batch_x)
            loss1 = custom_loss1(task1_pred, batch_t1)
            loss2 = custom_loss2(task2_pred, batch_t2)

            optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            g1 = extract_grad_vector(model.parameters()).to(device)

            optimizer.zero_grad()
            loss2.backward()
            g2 = extract_grad_vector(model.parameters()).to(device) 

            fused_grad, weight1, weight2 = _bpograd(g1, g2, g0_coeff, bpo_coeff)
            fused_grad = fused_grad.to(device)  
            optimizer.zero_grad() 
            try:
                _apply_grad_vector_to_params(model.parameters(), fused_grad) 
            except ValueError as e:
                print(f"fail to fuse - {e}")
                continue
            optimizer.step()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            fused_loss = weight1 * loss1 + weight2 * loss2
            total_fused_loss += fused_loss.item()

        avg_loss1 = total_loss1 / len(dataloader)
        avg_loss2 = total_loss2 / len(dataloader)
        avg_fused_loss = total_fused_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Avg Loss1: {avg_loss1:.4f} | "
              f"Avg Loss2: {avg_loss2:.4f} | "
              f"Avg Fused Loss: {avg_fused_loss:.4f} | "
              f"Loss1 Weight: {weight1:.4f} | Loss2 Weight: {weight2:.4f}")

    return model

if __name__ == "__main__":
    trained_model = train_mlp_with_fused_gradient()