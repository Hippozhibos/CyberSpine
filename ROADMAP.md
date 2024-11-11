# 强化学习项目计划

## 目标
1. 改用 Actor-Critic 方法。
2. 增加观察变量：
   - RGB 图像信息
   - 平衡信息（陀螺仪数据）
3. 调整任务和奖励函数设置：
   - 先让 agent reach 近处的目标，随后逐渐学习向较远距离移动。

## 实现步骤
### 1. 增加观察变量
- **RGB 图像信息**
  - 从 MuJoCo 中提取 RGB 图像。
  - 设计 CNN 特征提取器。
  - 将图像特征与其他观察变量结合。
- **平衡信息**
  - 配置 MuJoCo 的陀螺仪，获取平衡数据。
  - 将平衡数据整合进观察空间。

### 2. 改用 Actor-Critic 方法
- 设计并实现 actor 和 critic 网络。
- 调试网络，确保稳定学习。

### 3. 调整任务和奖励函数设置
- 设计奖励机制，针对近距离目标设定奖励。
- 动态调整任务难度，逐步扩展目标距离。

## 每日进度记录

### 2024-11-01
- 在example.ipynb中编辑，增加observation的信息。
- 可以load并plot图像（rgb & gray）。
- 已配置相应传感器，可获得gyro, accelerometer, velocimeter数据。

### 2024-11-04
- 设置了CNN基类
- 在update_policy中加入了CNN的参数
- 尝试在main函数中加入CNN
- 尝试使用torchvision将图像转为灰度图，报错，考虑放弃，尝试原有转换方法，不要在这里过多纠结。

### 2024-11-05
- 在example.ipynb中添加图像信息成功
- 接下来尝试将图像信息处理代码合并进在PG_demo.py中

### 2024-11-06
- 在PG_demo.py中添加图像信息成功
- 接下来尝试加入vestibule信息

### 2024-11-07
- 找到前庭信息的构成，gyro, accelerometer, velocimeter, z-axis,可作为一个10维向量输给PG。
- 可以开始考虑重新设计奖励函数，也即重新设计task,先去掉foraging target. 直接将target与头部的距离设为loss,同时加上姿态惩罚。暂时先不用CNN和视觉信息。

### 2024-11-08
- 将前庭信息（12维向量）添加给了PG。关节角/角速度+CNN+前庭 -> PG
- 训1000个episode看看效果
- 准备修改task, env, reward，试试看



>>>
## 步骤调整：
## 1. 再次尝试使用Benchmark任务 + RLlib的方案
- **修改world model算法**
  - 可行再添加肌肉维度

## 2. 重点是专注于强化学习算法设计，而不是为设计算法的人提供服务。
<<<

