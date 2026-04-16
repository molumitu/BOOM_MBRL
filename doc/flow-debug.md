# Flow Matching 监督修改总结

## 修改目标
将"单点动作监督"改成"基于 MPC 样本分布的 flow matching 监督"

---

## 核心变化

### 之前（单点监督）
```python
# 每个状态只用一个 target action (mu) 做 BC
x1 = mu  # [N, action_dim]
```

### 现在（分布监督）
```python
# 从 MPC 样本分布按权重采样
sample_indices = torch.multinomial(mpc_action_weights, num_samples=1).squeeze(-1)
x1 = mpc_action_samples[batch_indices, sample_indices]
```

---

## 详细代码解释

### 关键代码：`torch.multinomial` 详解

```python
sample_indices = torch.multinomial(mpc_action_weights, num_samples=1).squeeze(-1)  # [N]
```

#### 🔍 逐步分解

**1. `torch.multinomial(input, num_samples)`**
- **作用**：根据给定的概率分布进行**有放回采样**
- **参数**：
  - `input`: 概率权重（不需要归一化，但通常和为1）
  - `num_samples`: 每个分布采样的次数
- **返回**：采样得到的索引

**2. 具体示例**

假设我们有一个 batch 的 N=3 个状态，每个状态有 K=4 个 MPC 样本：

```python
# 输入：mpc_action_weights [N, K] = [3, 4]
mpc_action_weights = torch.tensor([
    [0.1, 0.4, 0.3, 0.2],  # 状态0的4个样本权重
    [0.25, 0.25, 0.25, 0.25],  # 状态1的4个样本权重（均匀）
    [0.7, 0.1, 0.1, 0.1],  # 状态2的4个样本权重（集中在第一个）
])

# 执行 multinomial 采样
sample_indices = torch.multinomial(mpc_action_weights, num_samples=1)
# 输出形状：[N, 1] = [3, 1]
# 可能的输出：
# tensor([[1],   # 状态0采样到索引1（权重0.4，概率最高）
#         [2],   # 状态1采样到索引2（均匀分布，随机任一）
#         [0]])  # 状态2采样到索引0（权重0.7，极大概率）

# squeeze(-1) 去掉最后一维
sample_indices = sample_indices.squeeze(-1)
# 输出形状：[N] = [3]
# tensor([1, 2, 0])
```

**3. 为什么这样设计？**

这是实现**从混合分布中采样**的关键：

```python
# MPC 给出的数据
mpc_action_samples: [N, K, action_dim]  # N个状态，每个状态K个候选动作
mpc_action_weights: [N, K]               # 每个候选动作的权重（概率）

# 目标：对每个状态 i，从 {samples[i, 0], ..., samples[i, K-1]} 中
#      按权重 weights[i, :] 采样一个动作作为 x1

# 实现步骤
# 1. 对每个状态，根据权重采样一个索引 [0, K-1]
sample_indices = torch.multinomial(mpc_action_weights, num_samples=1).squeeze(-1)  # [N]

# 2. 用采样的索引提取对应的动作
batch_indices = torch.arange(N, device=device)  # [0, 1, 2, ..., N-1]
x1 = mpc_action_samples[batch_indices, sample_indices]  # [N, action_dim]
```

**4. 直观理解**

可以把 `torch.multinomial` 理解为**转盘赌轮（Roulette Wheel）采样**：

```
状态0的权重分布：[0.1, 0.4, 0.3, 0.2]
转盘：  |---0.1---|--------0.4--------|---0.3---|--0.2--|
索引：     0              1               2        3

采样结果：索引1被选中（占比最大，概率最高）
```

---

## 完整 Flow Matching 流程

```python
def update_flow(self, z, mpc_action_samples, mpc_action_weights):
    """
    使用 MPC 样本分布计算 flow matching loss

    Args:
        z: [N, latent_dim] - 当前状态的 latent 表示
        mpc_action_samples: [N, K, action_dim] - K个MPC候选动作
        mpc_action_weights: [N, K] - 候选动作的权重（概率）
    """
    N, K, action_dim = mpc_action_samples.shape

    # ========================================
    # Step 1: 从 MPC 分布采样 x1
    # ========================================
    sample_indices = torch.multinomial(mpc_action_weights, num_samples=1).squeeze(-1)  # [N]
    batch_indices = torch.arange(N, device=device)  # [0, 1, ..., N-1]
    x1 = mpc_action_samples[batch_indices, sample_indices]  # [N, action_dim]

    # ========================================
    # Step 2: 从先验分布采样 x0
    # ========================================
    x0 = torch.randn_like(x1)  # [N, action_dim]

    # ========================================
    # Step 3: 采样随机时间 t
    # ========================================
    t = torch.rand(N, device=device) * (1.0 - 1e-3)  # [N]
    t_exp = t[:, None]  # [N, 1]

    # ========================================
    # Step 4: 计算插值点 xt
    # ========================================
    xt = (1.0 - t_exp) * x0 + t_exp * x1  # [N, action_dim]

    # ========================================
    # Step 5: 计算目标速度
    # ========================================
    target_vel = x1 - x0  # [N, action_dim]

    # ========================================
    # Step 6: 预测速度
    # ========================================
    flow_input = torch.cat([z, xt, t_exp], dim=-1)  # [N, latent_dim + action_dim + 1]
    pred_vel = self.model._flow_pi(flow_input)  # [N, action_dim]

    # ========================================
    # Step 7: 计算 Flow Matching Loss
    # ========================================
    flow_bc_loss = F.mse_loss(pred_vel, target_vel)

    return flow_bc_loss
```

---

## 修改的文件和函数

### 1. `boom/boom_alg.py`

#### `plan()` 函数
- **修改**：返回 MPC 精英样本和权重
- **新增返回值**：
  - `mpc_samples: [num_elites, action_dim]`
  - `mpc_weights: [num_elites]`

#### `act()` 函数
- **修改**：返回值从 3 个 → 5 个
- **新增返回值**：`mpc_samples`, `mpc_weights`

#### `update_flow()` 函数
- **完全重写**：使用 MPC 分布计算 loss
- **新参数**：
  - `mpc_action_samples: [N, K, action_dim]`
  - `mpc_action_weights: [N, K]`
- **核心逻辑**：按权重从 MPC 分布采样 x1

#### `update_pi()` 函数
- **新增参数**：`mpc_action_samples`, `mpc_action_weights`
- **修改**：调用新的 `update_flow()`

#### `update()` 函数
- **修改**：从 `replay_sample` 解包 MPC 数据

---

### 2. `boom/common/buffer.py`

#### `_prepare_batch()` 函数
- **新增字段**：
  - `mpc_action_samples`
  - `mpc_action_weights`

---

### 3. `boom/trainer/online_trainer.py`

#### `to_td()` 函数
- **新增参数**：`mpc_action_samples`, `mpc_action_weights`
- **默认值**：创建 dummy 数据

#### `train()` 函数
- **修改**：从 `agent.act()` 接收 5 个返回值

#### `eval()` 和 `eval_value()` 函数
- **修改**：解包 5 个返回值

---

## Tensor Shapes 总览

```python
# ============================================
# MPC 规划阶段
# ============================================
mpc_samples:  [num_elites, action_dim]      # MPPI 精英动作
mpc_weights:  [num_elites]                   # 归一化权重

# ============================================
# Buffer 存储
# ============================================
mpc_action_samples: [T, K, action_dim]       # T: 序列长度
mpc_action_weights: [T, K]

# ============================================
# 训练 Batch
# ============================================
mpc_action_samples: [H, B, K, action_dim]    # H: horizon, B: batch_size
mpc_action_weights: [H, B, K]

# ============================================
# Flow Matching 输入
# ============================================
z:                 [H*B, latent_dim]
mpc_samples_flat:  [H*B, K, action_dim]
mpc_weights_flat:  [H*B, K]

# ============================================
# 采样后
# ============================================
x1:                [H*B, action_dim]         # 从 K 个样本中采样 1 个
x0:                [H*B, action_dim]         # 标准正态
xt:                [H*B, action_dim]         # 插值点
target_vel:        [H*B, action_dim]         # x1 - x0
pred_vel:          [H*B, action_dim]         # flow_pi 预测
```

---

## Flow Matching 原理

### 什么是 Flow Matching？

Flow Matching 是一种学习概率分布的方法，通过学习从简单分布（如高斯）到目标分布的**速度场**来生成样本。

### 为什么用 MPC 分布？

**传统 BC（行为克隆）：**
- 问题：只用一个目标动作，丢失了 MPC 的多模态信息
- 示例：MPC 可能发现 3 个不同的好动作，但 BC 只学习均值

**Flow Matching + MPC：**
- 优势：保留 MPC 的完整分布信息
- MPC 的 K 个样本 + 权重 = 目标分布的经验估计
- Flow matching 学习这个分布的生成过程

### 数学原理

目标：学习一个向量场 `v_t(x, t)` 使得：
```
dx/dt = v_t(x, t)
```

从 `x_0 ~ N(0, I)` 到 `x_1 ~ π*(目标分布)` 的 ODE：
```
x_t = (1-t)x_0 + t*x_1
v_t(x_t) = x_1 - x_0
```

Loss：
```
L = E_{t,x0,x1}[||v_θ(x_t, t) - (x_1 - x_0)||²]
```

---

## 实验检查清单

- [x] MPC 规划返回样本和权重
- [x] Buffer 正确存储 MPC 数据
- [x] 训练器正确传递 MPC 数据
- [x] Flow matching 使用 MPC 分布
- [x] 所有函数签名更新
- [x] Tensor shapes 匹配
- [ ] 运行测试验证功能
- [ ] 监控训练指标

---

## 预期行为变化

### 之前
- Flow policy 学习单个目标动作（mu）
- 丢失 MPC 的多模态信息

### 现在
- Flow policy 学习完整的 MPC 分布
- 保留所有高质量动作的信息
- 更好的探索和利用

---

## 调试提示

### 如果 loss 爆炸
检查：
1. `mpc_action_weights` 是否归一化
2. `mpc_action_samples` 的值范围
3. `x1` 和 `x0` 的 scale 是否匹配

### 如果 loss 不下降
检查：
1. MPC 样本质量（是否都是好动作）
2. 权重分布（是否过于集中）
3. 学习率设置

### 监控指标
```python
info = {
    "flow_bc_loss": ...,
    "flow_pred_abs": ...,    # 预测速度的模
    "flow_target_abs": ...,  # 目标速度的模
    "flow_xt_abs": ...,      # 插值点的模
    "flow_t_mean": ...,      # 平均时间步
}
```

---

## 总结

通过这次修改，Flow policy 不再只学习单一目标，而是学习 MPC 的完整分布。这使得模型能够：
1. 保留多模态信息
2. 更好地探索动作空间
3. 提高最终性能

关键是使用 `torch.multinomial` 从权重分布中采样，实现了从**确定性目标**到**分布目标**的转变。
