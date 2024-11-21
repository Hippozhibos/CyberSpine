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


>>> 2024-11-11
## 步骤调整：
## 1. 再次尝试使用Benchmark任务 + RLlib的方案
- **修改world model算法**
  - 可行再添加肌肉维度

## 2. 重点是专注于强化学习算法设计，而不是为设计算法的人提供服务。

## 3. 重新回到dreamerv3的代码，在其基础上，尝试调整网络架构。Env先使用dm_control的任务。
<<<

### 2024-11-12
- 修改mice_env与熟悉dreamerv3并行；
- 在mice_env中添加了go_to_target, 下一步修改task；
- 熟悉dreamerv3，重点熟悉其转换DMC env的代码；

### 2024-11-13
- go_to_target.py 暂不需要较大修改，其定义的observable里没有egocentric camera信息，但是有target位置，因此可以模拟非视觉引导的环境；
- 接下来重点是在example.ipynb以及PG_demo.py中修改policy网络的代码，使agent能够按照target信息做任务。
- 考虑是否要将walker的位置判定标准，从root改为head

### 2024-11-15
- 已将PG_demo.py中的policy网络改为只接受qpos, qvel, vestibule,target_pos,测试了5个episode，效果不错，reward均为0.
- 已将walker的位置改为head, 为了配合计算reward;
- 修改go_to_target.py中的reward,加入了模拟前庭信息所需的传感器，前庭信息的计算方式有待修改。
- 准备阅读dreamerv3的代码，重点是dmc env wrapper的部分，理想情况是能把现有的mice go to task接到dreamerv3框架下。


### 2024-11-16
- 在dreamerv3的example.py下运行locom_rodent_maze_forage task, world model网络构建成功，但mujoco部分有报错：
  - UnboundLocalError: cannot access local variable 'env' where it is not associated with a value
  - dreamerv3的encoder decoder是将walker的所有observation空间都作为了高维向量了，没有进行任何区分选择。

### 2024-11-18
- 11月16日遇到的报错，在可以使用GPU资源后消失，已使用example.py在locom_rodent_maze_forage task任务里训练dreamerv3,从11月17日中午到11月18日晚，目前单卡已运行超过24h，6559152 steps；
- 细读dreamerv3/agent_v1.py, 尝试在dreamerv3代码基础上修改agent的构成框架。

### 2024-11-19
- 11月19日上午，dreamerv3训练进行到898万steps，仍在继续。
- 11月19日晚， 手动停止dreamerv3训练。训练进行到1e7以上，但默认steps数量是1e10，不停止的话，单张GPU要跑很久。
- Checkpoint 保存在/home/zhangzhibo/logdir/20241117T135935-example，以及复制在/home/zhangzhibo/CheckpointBackup。但想load checkpoint直接定义agent完成任务，还不知道怎么做。

### 2024-11-20
- 在CyberSpine/example.ipynb中，可以load dreamerv3及其子文件夹，但还没能成功load checkpoint。已询问原作者issue, https://github.com/danijar/dreamerv3/issues/2 , 等待回复。

### 2024-11-21
- 在CyberSpine/example.ipynb中, 用自己的代码呈现训练的checkpoint失败，这里的问题在于dreamerv3包装后的环境和dm_control的环境并不完全一样，我不能在dm_contrl的代码中直接render dreamerv3的环境。
- 目前还没找到dreamerv3的render代码，可能要再问一个issue。但估计理想情况还是基于dreamerv3的代码做可视化。
- dreamerv3训练默认的1e10是step number, 而不是episode number, 每个episode仍然是1500steps, 所以文章中的1e7的量级应该对应的是1e7个episode.
- 目前训练的结果是episode的reward可以提高到200左右，但波动仍较强。
- 应该有什么加速训练的设置，不然目前这个训练速度，1e7个episode要以年为单位才能跑完。