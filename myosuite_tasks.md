# MyoSuite 任务详情

本文档总结了 MyoSuite 环境中各个任务的状态空间和动作空间维数。

## 手部任务 (Hand Tasks)

### 1. myo-reach - 手部到达任务
- **环境ID**: `myoHandReachFixed-v0`
- **状态维数**: 115
- **动作维数**: 39
- **说明**: 固定目标到达任务

### 2. myo-reach-hard - 手部到达任务（困难）
- **环境ID**: `myoHandReachRandom-v0`
- **状态维数**: 115
- **动作维数**: 39
- **说明**: 随机目标到达任务

### 3. myo-pose - 手部姿态任务
- **环境ID**: `myoHandPoseFixed-v0`
- **状态维数**: 108
- **动作维数**: 39
- **说明**: 固定姿态任务

### 4. myo-pose-hard - 手部姿态任务（困难）
- **环境ID**: `myoHandPoseRandom-v0`
- **状态维数**: 108
- **动作维数**: 39
- **说明**: 随机姿态任务

### 5. myo-obj-hold - 物体抓持任务
- **环境ID**: `myoHandObjHoldFixed-v0`
- **状态维数**: 91
- **动作维数**: 39
- **说明**: 固定物体抓持任务

### 6. myo-obj-hold-hard - 物体抓持任务（困难）
- **环境ID**: `myoHandObjHoldRandom-v0`
- **状态维数**: 91
- **动作维数**: 39
- **说明**: 随机物体抓持任务

### 7. myo-key-turn - 钥匙转动任务
- **环境ID**: `myoHandKeyTurnFixed-v0`
- **状态维数**: 93
- **动作维数**: 39
- **说明**: 固定钥匙转动任务

### 8. myo-key-turn-hard - 钥匙转动任务（困难）
- **环境ID**: `myoHandKeyTurnRandom-v0`
- **状态维数**: 93
- **动作维数**: 39
- **说明**: 随机钥匙转动任务

### 9. myo-pen-twirl - 笔旋转任务
- **环境ID**: `myoHandPenTwirlFixed-v0`
- **状态维数**: 83
- **动作维数**: 39
- **说明**: 固定笔旋转任务

### 10. myo-pen-twirl-hard - 笔旋转任务（困难）
- **环境ID**: `myoHandPenTwirlRandom-v0`
- **状态维数**: 83
- **动作维数**: 39
- **说明**: 随机笔旋转任务

## 手臂任务 (Arm Tasks)

### 11. myo-arm-reach - 手臂到达任务
- **环境ID**: `myoArmReachFixed-v0`
- **状态维数**: 78
- **动作维数**: 32
- **说明**: 固定目标手臂到达任务

### 12. myo-arm-reach-hard - 手臂到达任务（困难）
- **环境ID**: `myoArmReachRandom-v0`
- **状态维数**: 78
- **动作维数**: 32
- **说明**: 随机目标手臂到达任务

## 总结

### 动作空间
- **手部任务**: 所有手部任务使用相同的动作空间维数 **39**
- **手臂任务**: 手臂任务使用动作空间维数 **32**

### 状态空间
- **最小状态维数**: 78 (myo-arm-reach)
- **最大状态维数**: 115 (myo-reach)
- **手部任务状态维数范围**: 83-115
- **手臂任务状态维数**: 78

### 任务类型说明
- **Fixed**: 目标/物体位置固定
- **Random**: 目标/物体位置随机化，增加了任务难度

## 使用方法

在 [boom/config.yaml](boom/config.yaml) 中设置：
```yaml
task: myo-reach  # 选择上述任一任务
env_type: myosuite
```

## 简单任务推荐 (入门测试用)

对于初次使用 MyoSuite 或需要快速验证算法正确性的场景，推荐以下两个简单任务：

### 1. myo-reach (手部到达任务 - 固定目标) ⭐ 最推荐新手
```yaml
task: myo-reach
env_type: myosuite
```
- **环境ID**: `myoHandReachFixed-v0`
- **状态维数**: 115
- **动作维数**: 39
- **难度等级**: ★☆☆☆☆ (最简单)
- **特点**:
  - 最基础的手部到达任务，目标位置固定
  - 任务直接：控制手部到达预定目标位置
  - Reward 相对密集，学习收敛快
  - 适合快速验证代码和算法框架
- **训练建议**: 
  - 预期训练时间：500K-1M iterations 即可收敛
  - 适合作为第一个 MyoSuite 任务测试
  - 可以先训练这个版本，成功后再尝试 myo-reach-hard

### 2. myo-arm-reach (手臂到达任务 - 固定目标) ⭐ 维数最少
```yaml
task: myo-arm-reach
env_type: myosuite
```
- **环境ID**: `myoArmReachFixed-v0`
- **状态维数**: 78 (所有任务中最少)
- **动作维数**: 32 (所有任务中最少)
- **难度等级**: ★★☆☆☆ (简单)
- **特点**:
  - 状态和动作空间维数都是最小的，计算效率高
  - 手臂运动比手部精细操作简单
  - 适合调试和快速实验
  - 可以用更少的计算资源进行训练
- **训练建议**:
  - 预期训练时间：500K-1M iterations
  - 适合资源有限时的快速验证
  - 可以作为测试新算法特性的基准任务

### 简单任务对比

| 任务 | 状态维数 | 动作维数 | 计算需求 | 学习难度 | 推荐用途 |
|------|----------|----------|----------|----------|----------|
| myo-reach | 115 | 39 | 中等 | ★☆☆☆☆ | 新手入门、快速验证 |
| myo-arm-reach | 78 | 32 | 低 | ★☆☆☆☆ | 资源有限、快速实验 |

### 使用流程建议

**初学者推荐路径**：
1. 先训练 `myo-reach` 验证框架
2. 再训练 `myo-arm-reach` 测试不同任务
3. 然后尝试困难版本（myo-reach-hard 或 myo-arm-reach-hard）
4. 最后挑战复杂任务（如下面的 myo-pen-twirl-hard）

**配置示例**：
```yaml
# 简单任务快速测试
task: myo-reach
env_type: myosuite
steps: 500_000  # 简单任务可以少训练一些
batch_size: 256
# 其他参数保持默认
```

---

## 任务推荐 (与 DMC Locomotion 任务难度相当)

基于对 DMC locomotion 任务（如 dog-run、humanoid-run）的复杂度分析，推荐以下难度相当的 MyoSuite 任务：

### DMC 任务参考
- **dog-run**: 状态维数 ~220（多模态），动作维数 38，全身协调控制
- **humanoid-run**: 类似复杂度的全身运动控制任务

### 推荐任务列表

#### 1. myo-pen-twirl-hard (笔旋转任务 - 困难版) ⭐ 强烈推荐
```yaml
task: myo-pen-twirl-hard
env_type: myosuite
```
- **环境ID**: `myoHandPenTwirlRandom-v0`
- **状态维数**: 83
- **动作维数**: 39（与 dog-run 的 38 维接近）
- **难度匹配**:
  - 需要精细的手指协调和序列控制
  - 动态平衡和精确控制，类似 locomotion 的连续协调要求
  - 涉及复杂的物体操纵，需要多阶段策略学习
- **特点**: 随机笔位置，需要泛化能力，最具挑战性的精细操作任务之一

#### 2. myo-key-turn-hard (钥匙转动任务 - 困难版) ⭐ 强烈推荐
```yaml
task: myo-key-turn-hard
env_type: myosuite
```
- **环境ID**: `myoHandKeyTurnRandom-v0`
- **状态维数**: 93
- **动作维数**: 39
- **难度匹配**:
  - 涉及复杂的力控制和手指协调
  - 需要精确的时序控制，类似完成一个完整运动周期
  - 多阶段任务：接近→抓握→转动→释放
- **特点**: 随机钥匙位置，需要精确的力和位置控制

#### 3. myo-obj-hold-hard (物体抓持任务 - 困难版)
```yaml
task: myo-obj-hold-hard
env_type: myosuite
```
- **环境ID**: `myoHandObjHoldRandom-v0`
- **状态维数**: 91
- **动作维数**: 39
- **难度匹配**:
  - 需要稳定的力和位置控制
  - 类似运动中的姿态维持和平衡控制
  - 持续的控制调整和反馈学习
- **特点**: 随机物体位置，需要稳定的抓持策略

### 难度对比表

| 任务 | 状态维数 | 动作维数 | 控制复杂度 | 学习难度 |
|------|----------|----------|------------|----------|
| dog-run | ~220 (多模态) | 38 | 全身协调 + 平衡 | ★★★★☆ |
| myo-pen-twirl-hard | 83 | 39 | 精细操作 + 动态控制 | ★★★★☆ |
| myo-key-turn-hard | 93 | 39 | 序列控制 + 力控 | ★★★★☆ |
| myo-obj-hold-hard | 91 | 39 | 稳定性控制 | ★★★☆☆ |

### 使用建议

**从最推荐的任务开始**：
1. **myo-pen-twirl-hard**: 最具挑战性，适合测试算法的复杂策略学习能力
2. **myo-key-turn-hard**: 复杂序列控制，适合评估时序建模能力

**配置示例**（在 [boom/config.yaml](boom/config.yaml)）：
```yaml
task: myo-pen-twirl-hard
env_type: myosuite
obs: state
# 其他训练参数保持不变...
```

**训练建议**：
- 这些任务需要较长的训练时间（建议 2M+ iterations）
- Reward 稀疏，可能需要调整 reward_coef 或探索策略
- Random 版本需要更好的泛化能力，是评估算法鲁棒性的好选择

## 参考资料
- MyoSuite 官方文档: https://myosuite.readthedocs.io/
- MyoSuite GitHub: https://github.com/MyoHub/myosuite
