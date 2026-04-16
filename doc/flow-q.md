# Flow Policy + Max Q Actor Training

## 概述

本文档描述了 Flow Policy 的混合训练方法，结合了：
1. **Flow Matching Loss** - 从 MPC 分布学习
2. **Max Q Actor Loss** - 通过 Q 值优化动作

---

## 目标

将 Flow Policy 从纯监督学习（MPC 分布拟合）扩展为监督+强化学习混合训练：

```
total_flow_loss = flow_bc_loss + flow_q_coef * flow_q_loss
```

其中：
- `flow_bc_loss`: Flow matching loss（拟合 MPC 分布）
- `flow_q_loss`: Max Q loss（最大化 Q 值）
- `flow_bc_coef`: 平衡系数（默认 0.1）

---

## 核心修改

### 修改的文件

**[boom/boom_alg.py](boom/boom_alg.py)** - `update_pi()` 函数中的 Flow Policy 更新部分（第 411-459 行）

### 修改内容

```python
if self.cfg.update_flow:
    # =========================
    # 2) Update Flow Policy with Hybrid Loss
    # =========================
    self.flow_optim.zero_grad(set_to_none=True)

    # flatten [H, B, ...] -> [H*B, ...]
    z_flat = zs.reshape(H * B, -1).detach()

    # Reshape MPC samples and weights: [H, B, K, A] -> [H*B, K, A]
    mpc_samples_flat = mpc_action_samples.reshape(H * B, -1, action.shape[-1])
    mpc_weights_flat = mpc_action_weights.reshape(H * B, -1)

    # ========================================
    # Part 1: Flow Matching Loss (监督项)
    # ========================================
    flow_bc_loss, flow_info = self.update_flow(z_flat, mpc_samples_flat, mpc_weights_flat)

    # ========================================
    # Part 2: Max Q Actor Loss (强化学习项)
    # ========================================
    # Freeze critic, only update flow policy
    self.model.track_q_grad(False)

    # Generate actions from flow policy (fully differentiable)
    a_flow = self.model.flow_policy(z_flat)  # [H*B, action_dim]

    # Compute Q-values using shared critic
    flow_q = self.model.Q(z_flat, a_flow, task, return_type="min")  # [H*B, 1]

    # Actor loss: maximize Q-values
    flow_q_loss = -flow_q.mean()

    # Restore critic gradient tracking
    self.model.track_q_grad(True)

    # ========================================
    # Part 3: Combine Losses
    # ========================================
    flow_q_coef = getattr(self.cfg, "flow_q_coef", 1.0)
    total_flow_loss = flow_bc_loss + flow_q_coef * flow_q_loss

    # Backward and update
    total_flow_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        self.model._flow_pi.parameters(),
        self.cfg.grad_clip_norm,
    )
    self.flow_optim.step()
```

---

## 技术细节

### 1. Flow Policy 生成动作（可微分）

```python
# flow_policy 使用 ODE 积分，完全可微
def flow_policy(self, z, n_steps=10):
    x0 = torch.randn(B, action_dim)  # 从标准正态分布采样
    for i in range(n_steps):
        x = integrate(x, t_start, t_end)  # ODE 积分（可微）
    a_flow = tanh(x)
    return a_flow
```

**关键特性：**
- ✅ **完全可微**：ODE 积分使用可微操作
- ✅ **随机采样**：从 x0 ~ N(0, I) 开始
- ✅ **直接复用**：不需要修改 sampling 接口

### 2. 共享 Critic

```python
# 使用现有的 critic，不需要新建
flow_q = self.model.Q(z_flat, a_flow, task, return_type="min")
```

**优势：**
- ✅ **复用现有 Q**：不需要新建 flow 专属 Q
- ✅ **Critic 冻结**：通过 `track_q_grad(False)` 实现
- ✅ **只更新 flow**：只有 `_flow_pi` 参数更新

### 3. 梯度流

```
Loss: total_flow_loss = flow_bc_loss + flow_q_coef * flow_q_loss
                      ↓                ↓
        Flow Matching        Max Q Actor
                      ↓                ↓
           v_θ(x_t, t) ←─────────→ Q(z, a_flow)
                      ↓                ↓
              _flow_pi.parameters()
```

**梯度传播路径：**
```
flow_q_loss → flow_q → a_flow → flow_policy() → _flow_pi
```

### 4. Critic 冻结机制

```python
# 冻结 critic
self.model.track_q_grad(False)
flow_q = self.model.Q(z_flat, a_flow, task, return_type="min")

# 恢复 critic
self.model.track_q_grad(True)
```

**作用：**
- 阻止梯度更新 critic 参数
- 只允许 flow policy 通过 Q 值学习

---

## Loss 组成

### 1. Flow Matching Loss（监督项）

```python
# 从 MPC 分布采样
sample_indices = torch.multinomial(mpc_action_weights, num_samples=1).squeeze(-1)
x1 = mpc_action_samples[batch_indices, sample_indices]

# 从先验分布采样
x0 = torch.randn_like(x1)

# 随机时间
t = torch.rand(N)
xt = (1-t) * x0 + t * x1

# 计算速度场
v_pred = flow_pi(xt, t, z)
v_target = x1 - x0

# Flow matching loss
flow_bc_loss = ||v_pred - v_target||²
```

**作用：**
- 学习从 N(0, I) 到 MPC 分布的变换
- 保留 MPC 的多模态信息
- 提供稳定的训练信号

### 2. Max Q Actor Loss（强化学习项）

```python
# 生成动作
a_flow = flow_policy(z)

# 计算 Q 值
flow_q = Q(z, a_flow)

# 最大化 Q 值
flow_q_loss = -flow_q.mean()
```

**作用：**
- 优化累积回报
- 通过 Q 值指导探索
- 改善长期性能

### 3. 组合 Loss

```python
total_flow_loss = flow_bc_loss + flow_q_coef * flow_q_loss
```

**平衡策略：**
- `flow_q_coef = 0.0`: 纯监督学习（只用 MPC）
- `flow_q_coef = 0.1`: 以 flow matching 为主
- `flow_q_coef = 1.0`: 平衡训练（推荐）
- `flow_q_coef = 10.0`: 以 Q 优化为主

---

## 配置参数

### config.yaml

```yaml
# Flow policy 训练配置
update_flow: true         # 是否更新 flow policy
flow_q_coef: 1.0         # Max Q loss 系数

# MPC 配置
num_elites: 64           # MPC 精英数量（用于生成分布）
num_flow_trajs: 48       # Flow policy 引导轨迹数量
```

---

## 训练流程

### 完整的 Update 循环

```python
def update_pi(self, zs, action, mu, std, mpc_action_samples, mpc_action_weights, task, step):
    H, B, _ = zs.shape

    # =========================
    # 1) Update Gaussian Policy
    # =========================
    self.pi_optim.zero_grad(set_to_none=True)
    self.model.track_q_grad(False)

    # Max Q loss
    _, pis, log_pis, log_std = self.model.pi(zs, task)
    qs = self.model.Q(zs, pis, task, return_type="min")
    q_loss = ((entropy_coef * log_pis - qs).mean() * rho).mean()

    # Min KL loss
    eps = (pis - mu) / std
    forward_kl = gaussian_logprob(eps, std.log())
    fkl_loss = -(forward_kl * rho).mean()

    # Update
    pi_loss = q_loss + fkl_loss
    pi_loss.backward()
    self.pi_optim.step()

    # =========================
    # 2) Update Flow Policy (混合训练)
    # =========================
    if self.cfg.update_flow:
        self.flow_optim.zero_grad(set_to_none=True)

        # Flow Matching Loss
        flow_bc_loss, flow_info = self.update_flow(z_flat, mpc_samples_flat, mpc_weights_flat)

        # Max Q Actor Loss
        self.model.track_q_grad(False)
        a_flow = self.model.flow_policy(z_flat)
        flow_q = self.model.Q(z_flat, a_flow, task, return_type="min")
        flow_q_loss = -flow_q.mean()
        self.model.track_q_grad(True)

        # Combine and update
        total_flow_loss = flow_bc_loss + flow_q_coef * flow_q_loss
        total_flow_loss.backward()
        self.flow_optim.step()

    self.model.track_q_grad(True)
```

---

## 监控指标

### 训练指标

```python
info = {
    # Flow policy 指标
    "flow_bc_loss": float(flow_bc_loss.item()),      # Flow matching loss
    "flow_q_loss": float(flow_q_loss.item()),        # Max Q loss
    "flow_total_loss": float(total_flow_loss.item()), # 总 loss

    # Flow matching 详细指标
    "flow_pred_abs": float(flow_info["flow_pred_abs"].item()),   # 预测速度模长
    "flow_target_abs": float(flow_info["flow_target_abs"].item()), # 目标速度模长
    "flow_xt_abs": float(flow_info["flow_xt_abs"].item()),       # 插值点模长
    "flow_t_mean": float(flow_info["flow_t_mean"].item()),       # 平均时间步
}
```

### 重点关注

1. **flow_q_loss**: 应该逐渐减小（Q 值增大）
2. **flow_bc_loss**: 保持稳定（持续拟合 MPC 分布）
3. **flow_total_loss**: 总体趋势下降
4. **flow_pred_abs vs flow_target_abs**: 应该接近

---

## 预期效果

### 训练初期

- **flow_bc_loss**: 主导，提供稳定信号
- **flow_q_loss**: 较大，Q 值可能为负

### 训练中期

- **flow_bc_loss**: 继续拟合 MPC
- **flow_q_loss**: 逐渐减小，Q 值提升
- **两项 balance**: 达到平衡

### 训练后期

- **flow_bc_loss**: 保持稳定
- **flow_q_loss**: 趋于收敛
- **性能**: Flow policy 超过 MPC

---

## 优势分析

### 1. 混合训练

**纯监督学习（之前）：**
- ❌ 只能模仿 MPC，无法超越
- ❌ 忽略长期回报
- ❌ 受 MPC 质量限制

**混合训练（现在）：**
- ✅ 从 MPC 分布学习（多模态）
- ✅ 通过 Q 值优化（超越 MPC）
- ✅ 结合监督+强化学习

### 2. 梯度信号

**Flow Matching:**
- 提供稳定的监督信号
- 保留分布信息
- 防止崩溃

**Max Q Actor:**
- 优化长期回报
- 引导探索
- 提升性能上限

### 3. 训练稳定性

**Critic 冻结：**
- 防止 critic 被 actor 污染
- 保持 Q 值估计稳定
- 类似于 TD3 的延迟更新

**Loss 权重：**
- `flow_bc_coef` 控制平衡
- 可根据训练阶段调整
- 提供额外的控制维度

---

## 调试指南

### 如果 flow_q_loss 爆炸

**可能原因：**
- Q 值估计不稳定
- 动作超出合理范围
- 学习率过大

**解决方法：**
- 降低 `flow_q_coef`（减弱 Q 优化）
- 检查 Q 函数的更新
- 降低学习率

### 如果 flow_q_loss 不下降

**可能原因：**
- Critic 估计不准确
- 探索不足
- 梯度传播问题

**解决方法：**
- 提高 `flow_q_coef`（加强 Q 优化）
- 增加 exploration noise
- 检查梯度流

### 如果 flow_bc_loss 上升

**可能原因：**
- Q 项主导，忽略 MPC
- Flow policy 忘记 MPC 分布

**解决方法：**
- 提高 `flow_bc_coef`
- 检查 MPC 样本质量
- 降低学习率

---

## 实验建议

### 超参数调优

1. **flow_bc_coef**: 从 [0.01, 0.1, 1.0, 10.0] 中搜索
2. **Learning rate**: flow policy 可能需要不同的 lr
3. **Update frequency**: 考虑延迟更新 flow policy

### 评估指标

1. **Q 值**: flow_q 应该逐渐提高
2. **Return**: 与纯 MPC 比较性能
3. **Distribution**: 检查是否保留多模态

### 消融实验

1. **Only Flow Matching**: `flow_bc_coef = 1, flow_q_loss = 0`
2. **Only Max Q**: `flow_bc_coef = 0`
3. **Hybrid**: 不同 `flow_bc_coef` 值

---

## 总结

### 核心改进

| 方面 | 之前 | 现在 |
|------|------|------|
| **训练目标** | MPC 分布拟合 | MPC + Q 优化 |
| **Loss** | 单项 | 两项组合 |
| **性能上限** | 受限于 MPC | 可能超越 MPC |
| **稳定性** | 依赖 MPC 质量 | 更鲁棒 |

### 关键特性

✅ **可微分采样**: Flow policy 完全可微
✅ **共享 Critic**: 复用现有 Q 函数
✅ **Critic 冻结**: 保持 Q 值稳定
✅ **混合训练**: 监督 + 强化学习
✅ **灵活配置**: 通过 `flow_bc_coef` 控制

### 预期结果

通过混合训练，Flow Policy 应该：
1. 保留 MPC 的多模态特性
2. 通过 Q 优化超越 MPC
3. 具有更好的泛化能力
4. 训练更加稳定

---

## 文件清单

- **[boom/boom_alg.py](boom/boom_alg.py)**: 核心修改（update_pi 函数）
- **[boom/config.yaml](boom/config.yaml)**: 配置参数
- **[flow-debug.md](flow-debug.md)**: MPC 分布监督文档
- **[flow-q.md](flow-q.md)**: 本文档（混合训练）
