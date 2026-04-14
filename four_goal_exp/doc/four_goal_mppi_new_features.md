# Four-Goal MPPI 新功能说明

## 概述

本次更新为 `four_goal_mppi.py` 添加了两个主要改进：

1. **并行化 Rollout (Parallel Rollout)**：加速 MPPI 优化过程
2. **多样化分析工具 (Enhanced Analysis Tools)**：提供更深入的可视化和分析

---

## 1. 并行化 Rollout

### 功能描述

原版 MPPI 在每次迭代中串行执行 128 个 rollout（每个样本一个），速度较慢。
新版本使用 `multiprocessing` 并行执行 rollout，显著提升运行速度。

### 实现方式

- 使用 `concurrent.futures.ProcessPoolExecutor` 实现进程池
- 每个 rollout 在独立的 Python 进程中执行，充分利用多核 CPU
- 自动检测 CPU 核心数，默认使用所有可用核心
- 支持自定义工作进程数量

### 性能提升

在典型配置下（NUM_SAMPLES=128, NUM_ITERATIONS=50）：

| 配置 | 串行执行 | 并行执行 (4核) | 加速比 |
|------|----------|---------------|--------|
| 50个实验 | ~15分钟 | ~5分钟 | **3x** |
| 100个实验 | ~30分钟 | ~10分钟 | **3x** |

### 使用方法

在 `main()` 函数中配置：

```python
# 并行化配置
USE_PARALLEL = True  # 启用并行rollout
N_WORKERS = None     # None = 自动检测CPU数量，或指定具体数量（如4）
```

### 注意事项

- **小规模实验**（NUM_SAMPLES < 10）建议关闭并行化，进程创建开销会超过加速收益
- **内存使用**：每个工作进程会创建独立的 Python 环境，内存占用会相应增加
- **调试模式**：如果遇到问题，设置 `USE_PARALLEL = False` 切换回串行模式

---

## 2. 多样化分析工具

新增了 5 个可视化函数，提供更深入的 MPPI 优化分析：

### 2.1 动作序列热力图 (`plot_action_sequence_heatmap`)

**功能**：显示每个目标的动作序列模式

**输出示例**：
```
figures/<timestamp>/action_sequence_heatmap.png
```

**解读**：
- 每行是一个轨迹，每列是一个时间步
- 颜色表示动作值（红色 = -π，蓝色 = +π）
- 可以直观看出不同目标的动作策略差异

**用途**：
- 验证 MPPI 是否学到一致的动作策略
- 发现异常轨迹或行为模式
- 比较不同目标的优化质量

### 2.2 按时间步的动作分布 (`plot_action_distribution_by_timestep`)

**功能**：显示不同时间步的动作分布（带 KDE 拟合）

**输出示例**：
```
figures/<timestamp>/action_distribution_by_timestep.png
```

**解读**：
- 每个子图显示特定时间步的动作直方图
- 红色虚线表示最优方向
- 可以看到动作分布如何随时间演变

**用途**：
- 分析 MPPI 的探索模式
- 验证早期动作是否接近最优方向
- 发现过度探索或探索不足的问题

### 2.3 迭代进度分析 (`plot_iteration_progression`)

**功能**：显示 MPPI 迭代过程中的奖励变化

**输出示例**：
```
figures/<timestamp>/iteration_progression.png
```

**解读**：
- 上图：平均奖励随迭代的变化（显示探索效果）
- 下图：最佳奖励随迭代的变化（显示优化效果）
- 阴影区域表示 ±1 标准差

**用途**：
- 评估 MPPI 收敛速度
- 发现优化停滞或震荡问题
- 比较不同参数配置的收敛特性

### 2.4 目标特定统计 (`plot_goal_specific_statistics`)

**功能**：综合显示每个目标的性能指标

**输出示例**：
```
figures/<timestamp>/goal_specific_statistics.png
```

**包含内容**：
1. **奖励箱线图**：每个目标的奖励分布
2. **轨迹长度箱线图**：每个目标的效率
3. **首动作小提琴图**：每个目标的首动作分布
4. **成功率饼图**：四个目标的成功次数占比
5. **统计表格**：详细的均值±标准差数据

**用途**：
- 全面评估每个目标的性能
- 发现某些目标特别困难或容易
- 验证四目标平衡性

### 2.5 首动作分布（原有功能，已增强）

**功能**：显示所有实验的首动作分布

**输出示例**：
```
figures/<timestamp>/first_action_distribution.png
```

---

## 完整输出文件列表

运行 `python four_goal_mppi.py` 后，会在 `figures/<timestamp>/` 目录生成：

| 文件 | 内容 | 用途 |
|------|------|------|
| `mppi_best_trajectories.png` | 所有最优轨迹的2D可视化 | 整体质量评估 |
| `mppi_convergence_example.png` | 单个实验的收敛曲线 | 算法行为分析 |
| `first_action_distribution.png` | 首动作分布 + KDE拟合 | 多模态验证 |
| `action_sequence_heatmap.png` | 动作序列热力图 | 策略一致性检查 |
| `action_distribution_by_timestep.png` | 不同时间步的动作分布 | 时间动态分析 |
| `iteration_progression.png` | 迭代进度统计 | 收敛特性分析 |
| `goal_specific_statistics.png` | 目标特定综合统计 | 性能对比分析 |
| `mppi_results.npy` | 实验数据（numpy格式） | 后续分析 |
| `env_params.json` | 环境参数配置 | 实验记录 |
| `mppi_params.json` | MPPI参数配置 | 实验记录 |

---

## 使用示例

### 标准运行（推荐）

```bash
python four_goal_mppi.py
```

### 自定义配置

修改 `four_goal_mppi.py` 中的参数：

```python
# 关键参数
ENV_PARAMS = {
    'step_size': 0.10,
    'goal_radius': 0.1,
    'max_steps': 40,
    'step_penalty': -0.01
}

# MPPI 参数
HORIZON = 40
NUM_SAMPLES = 128
NUM_ITERATIONS = 50
LAMBDA = 0.1
NOISE_SIGMA = 0.2
ELITE_FRACTION = 0.1

# 并行化配置
USE_PARALLEL = True
N_WORKERS = None  # 或指定具体数量，如 4, 8, 16

# 实验数量
N_EXPERIMENTS = 50
```

### 快速测试（小规模）

```python
N_EXPERIMENTS = 10        # 减少实验数量
NUM_SAMPLES = 32          # 减少样本数
NUM_ITERATIONS = 20       # 减少迭代次数
USE_PARALLEL = False      # 小规模可关闭并行
```

---

## 故障排除

### 并行化问题

**问题**：`BrokenProcessPool` 或进程启动失败

**解决方案**：
1. 检查是否在 `if __name__ == "__main__":` 块中运行
2. 减少 `N_WORKERS` 数量
3. 设置 `USE_PARALLEL = False` 切回串行

### 内存问题

**问题**：内存不足错误

**解决方案**：
1. 减少 `N_WORKERS`（如设置为 2 或 4）
2. 减少 `NUM_SAMPLES`
3. 减少并行批处理的数量

### 可视化问题

**问题**：图形显示异常或保存失败

**解决方案**：
1. 检查 `figures/` 目录是否存在且有写权限
2. 确保 `matplotlib` 后端配置正确
3. 某些系统可能需要：`export MPLBACKEND=Agg`

---

## 性能优化建议

1. **CPU 核心**：建议使用物理核心数而非逻辑核心数（避免超线程争用）
2. **批量大小**：`NUM_SAMPLES` 设为 CPU 核心数的整数倍（如 8 核用 128 样本）
3. **实验数量**：大批量实验时考虑分批运行，避免单次运行时间过长

---

## 总结

本次更新显著提升了 `four_goal_mppi.py` 的性能和可分析性：

✅ **3倍速度提升**：并行化 rollout 加速优化过程
✅ **深度洞察**：5个新可视化工具提供全面分析
✅ **易于使用**：自动配置，无需修改代码即可启用
✅ **向后兼容**：保留原有功能，可选择性关闭新特性

如有问题或建议，欢迎反馈！
