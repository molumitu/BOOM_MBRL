# Flow Q Coefficient Ablation Study

## 概述

本消融实验对比两种 Flow Policy 训练策略：

1. **Pure Flow Matching** (`flow_q_coef=0`, `extra=ablation_q0`): 纯监督学习，只使用 MPC 分布
2. **Hybrid Training** (`flow_q_coef=1`, `extra=ablation_q1`): 混合训练，结合 Flow Matching + Max Q Actor

## 实验配置

- **环境**: 根据脚本中的 `TASK` 变量（默认 dog-run）
- **训练步数**: 使用 config.yaml 默认值
- **评估频率**: 使用 config.yaml 默认值
- **并行运行**: GPU 0 和 GPU 1
- **区分标识**:
  - GPU 0: `extra="ablation_q0"`
  - GPU 1: `extra="ablation_q1"`

## 使用方法

### 运行消融实验

```bash
cd flow_q_ablation
./run_ablation.sh
```

### 查看实时日志

```bash
# GPU 0: Pure Flow Matching
tail -f logs/pure_flow_gpu0_*.log

# GPU 1: Hybrid Training
tail -f logs/hybrid_gpu1_*.log
```

## 预期结果

### Pure Flow Matching (flow_q_coef=0)
- 只从 MPC 分布学习
- 性能受 MPC 质量
- 训练更稳定

### Hybrid Training (flow_q_coef=1)
- MPC 分布 + Q 优化
- 可能超越 MPC
- 需要稳定

## 对比指标

关注以下指标：

1. **Episode Return**: 最终性能
2. **Flow BC Loss**: MPC 拟合质量
3. **Flow Q Loss**: Q 优化效果
4. **Flow Total Loss**: 总体损失

## 结果分析

### 训练曲线
- Pure Flow Matching: 单调下降
- Hybrid Training: 可能有波动

### 性能对比
- Hybrid 应该 ≥ Pure
- 如果 Hybrid < Pure: 调整 flow_q_coef

## 文件结构

```
flow_q_ablation/
├── README.md              # 本文档
├── run_ablation.sh        # 运行脚本
└── logs/                  # 日志目录
    ├── pure_flow_gpu0_*.log
    └── hybrid_gpu1_*.log
```

## W&B 可视化

在 W&B 项目中查看：
- Project: `flow`
- Group: `flow_q_ablation`
- Runs: `pure_flow_matching`, `hybrid_training`

对比曲线：
- `results/return`: Episode return
- `flow_bc_loss`: Flow matching loss
- `flow_q_loss`: Max Q loss
- `flow_total_loss`: Total loss
