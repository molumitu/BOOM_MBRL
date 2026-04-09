# Flow Q Ablation Results

## 实验配置

| 参数 | Pure Flow Matching | Hybrid Training |
|------|-------------------|-----------------|
| flow_q_coef | 0.0 | 1.0 |
| extra | ablation_q0 | ablation_q1 |
| GPU | 0 | 1 |
| Steps | 默认 | 默认 |
| Eval Freq | 默认 | 默认 |

## 性能对比

| 指标 | Pure Flow (q=0) | Hybrid (q=1) | 改进 |
|------|-----------------|--------------|------|
| Final Return | - | - | - |
| Best Return | - | - | - |
| Eval Success Rate | - | - | - |
| Flow BC Loss | - | - | - |
| Flow Q Loss | - | - | - |
| Total Loss | - | - | - |

## 训练曲线对比

### Episode Return
```
Steps:    0      50k    100k   200k   500k   1M
Pure:
Hybrid:
```

### Flow BC Loss
```
Steps:    0      50k    100k   200k   500k   1M
Pure:
Hybrid:
```

### Flow Q Loss
```
Steps:    0      50k    100k   200k   500k   1M
Pure:     N/A    N/A    N/A    N/A    N/A    N/A
Hybrid:
```

## 结论

### 发现
1.
2.
3.

### 建议
1.
2.
3.

## W&B Links

- Pure Flow Matching: [链接]
- Hybrid Training: [链接]
- Comparison Chart: [链接]

---

## 如何填写结果

### 从日志中提取

```bash
# 查看最终评估结果
grep "eval/episode_reward" logs/pure_flow_gpu0_*.log | tail -1
grep "eval/episode_reward" logs/hybrid_gpu1_*.log | tail -1

# 查看 Flow losses
grep "flow_bc_loss" logs/*_*.log | tail -5
grep "flow_q_loss" logs/*_*.log | tail -5
```

### 从 W&B 下载

1. 进入项目: https://wandb.ai/shallowdream0745-thu/flow
2. 筛选 Group: `flow_q_ablation`
3. 对比两个 run 的指标
4. 导出 CSV 或截图
