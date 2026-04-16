# DMControl Suite 任务总结

DMControl (DeepMind Control) Suite 是一个基于物理的控制任务套件，提供了 51 个连续控制任务，涵盖了从简单到复杂的各种控制场景。

## 任务概览

- **总任务数**: 51 个
- **域数量**: 25 个
- **观察类型**: state (状态), rgb (图像)
- **环境类型**: dm_control

## 任务分类

### 🏃 运动类任务 (Locomotion)

#### 四足动物
- **Cheetah (猎豹)**
  - `cheetah-run` - 猎豹向前奔跑

- **Dog (狗)**
  - `dog-fetch` - 狗取物
  - `dog-run` - 狗奔跑
  - `dog-stand` - 狗站立
  - `dog-trot` - 狗小跑
  - `dog-walk` - 狗行走

- **Quadruped (四足机器人)**
  - `quadruped-escape` - 四足机器人逃脱
  - `quadruped-fetch` - 四足机器人取物
  - `quadruped-run` - 四足机器人奔跑
  - `quadruped-walk` - 四足机器人行走

#### 人形机器人
- **Humanoid (人形)**
  - `humanoid-run` - 人形机器人奔跑
  - `humanoid-run_pure_state` - 人形机器人奔跑（纯状态）
  - `humanoid-stand` - 人形机器人站立
  - `humanoid-walk` - 人形机器人行走

- **Humanoid CMU (CMU 人形动作捕获)**
  - `humanoid_CMU-run` - CMU 人形奔跑
  - `humanoid_CMU-stand` - CMU 人形站立
  - `humanoid_CMU-walk` - CMU 人形行走

#### 双足/单足机器人
- **Walker (双足机器人)**
  - `walker-run` - 双足机器人奔跑
  - `walker-stand` - 双足机器人站立
  - `walker-walk` - 双足机器人行走

- **Hopper (单足跳跃机器人)**
  - `hopper-hop` - 单足机器人跳跃
  - `hopper-stand` - 单足机器人站立

#### 水中生物
- **Fish (鱼)**
  - `fish-swim` - 鱼游泳
  - `fish-upright` - 鱼保持直立

- **Swimmer (游泳者)**
  - `swimmer-swimmer15` - 15关节游泳者
  - `swimmer-swimmer6` - 6关节游泳者

### 🎯 操纵类任务 (Manipulation)

#### 机械臂
- **Reacher (机械臂)**
  - `reacher-easy` - 简单机械臂到达任务
  - `reacher-hard` - 困难机械臂到达任务

- **Manipulator (机械手)**
  - `manipulator-bring_ball` - 机械手带球
  - `manipulator-bring_peg` - 机械手带销
  - `manipulator-insert_ball` - 机械手插球
  - `manipulator-insert_peg` - 机械手插销

#### 手指
- **Finger (手指)**
  - `finger-spin` - 手指旋转
  - `finger-turn_easy` - 手指转动（简单）
  - `finger-turn_hard` - 手指转动（困难）

#### 堆叠
- **Stacker (堆叠)**
  - `stacker-stack_2` - 堆叠2个方块
  - `stacker-stack_4` - 堆叠4个方块

### 🎪 平衡与摆动类任务 (Balance & Swing)

#### 摆系统
- **Acrobot (双摆)**
  - `acrobot-swingup` - 双摆摆起
  - `acrobot-swingup_sparse` - 双摆摆起（稀疏奖励）

- **Cartpole (倒立摆)**
  - `cartpole-balance` - 倒立摆平衡
  - `cartpole-balance_sparse` - 倒立摆平衡（稀疏奖励）
  - `cartpole-swingup` - 倒立摆摆起
  - `cartpole-swingup_sparse` - 倒立摆摆起（稀疏奖励）
  - `cartpole-three_poles` - 三级倒立摆
  - `cartpole-two_poles` - 二级倒立摆

- **Pendulum (单摆)**
  - `pendulum-swingup` - 单摆摆起

### 🎱 其他任务 (Others)

#### 目标任务
- **Ball_in_cup (球在杯中)**
  - `ball_in_cup-catch` - 将球接人杯中

#### 导航任务
- **Point_mass (质点)**
  - `point_mass-easy` - 质点导航（简单）
  - `point_mass-hard` - 质点导航（困难）

#### 控制理论
- **Lqr (线性二次调节器)**
  - `lqr-lqr_2_1` - LQR 2D 1D控制
  - `lqr-lqr_6_2` - LQR 6D 2D控制

## 在配置文件中使用

### 基本配置

在 `boom/config.yaml` 中设置任务：

```yaml
# 环境类型
env_type: dm_control

# 任务选择（格式：domain-task）
task: cheetah-run      # 猎豹奔跑
task: walker-walk      # 双足行走
task: humanoid-run     # 人形奔跑
task: cartpole-balance # 倒立摆平衡
task: reacher-easy     # 简单机械臂
```

### 观察类型

```yaml
obs: state   # 状态观察（默认）
obs: rgb     # 图像观察
```

### 任务命名规范

任务的命名格式为：`domain-task_variant`

- `domain`: 环境域（如 cheetah, walker, humanoid）
- `task`: 具体任务（如 run, walk, stand）
- `variant`: 任务变体（可选，如 sparse, easy, hard）

例如：
- `cheetah-run` - 猎豹奔跑
- `cartpole-balance_sparse` - 倒立摆平衡（稀疏奖励）
- `reacher-easy` - 简单机械臂

## 任务难度分级

### 🟢 入门级任务
适合初学者和快速测试：
- `cartpole-balance` - 倒立摆平衡
- `pendulum-swingup` - 单摆摆起
- `reacher-easy` - 简单机械臂
- `point_mass-easy` - 简单质点导航

### 🟡 中级任务
需要一定的控制策略：
- `cartpole-swingup` - 倒立摆摆起
- `walker-walk` - 双足行走
- `hopper-hop` - 单足跳跃
- `cheetah-run` - 猎豹奔跑

### 🔴 高级任务
复杂的运动控制和操纵：
- `humanoid-run` - 人形机器人奔跑
- `dog-run` - 狗奔跑
- `manipulator-insert_peg` - 机械手插销
- `stacker-stack_4` - 堆叠4个方块

## 观察空间与动作空间

### 观察空间
- **state**: 关节位置、速度、外部物体位置等
- **rgb**: 64×64 或 84×84 RGB 图像

### 动作空间
- 所有任务都是连续动作空间
- 动作范围通过 wrapper 归一化到 [-1, 1]
- 不同任务的动作维度不同（通常为 1-6 维）

## 奖励类型

### Dense Reward (密集奖励)
- 每个时间步都有连续的奖励值
- 大部分任务使用密集奖励
- 有利于学习效率

### Sparse Reward (稀疏奖励)
- 只在关键事件发生时给予奖励
- 标记为 `_sparse` 的任务
- 更接近真实场景但学习难度更高

## 相关文件

- **环境实现**: [boom/envs/dmcontrol.py](boom/envs/dmcontrol.py)
- **任务定义**: [boom/envs/tasks/](boom/envs/tasks/)
- **评估脚本**: [eval_dmc.py](eval_dmc.py)

## 参考资料

- [DMControl Suite 官方文档](https://github.com/deepmind/dm_control)
- [论文: DeepMind Control Suite](https://arxiv.org/abs/1801.00690)
- 任务设计基于物理引擎 MuJoCo

## 更新日志

- 2024: 添加 custom 任务支持
- 持续更新：添加新的任务和域
