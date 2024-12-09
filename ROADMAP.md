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
- 也许应该先eval, 让actor跑一遍，把数据生成出来，再render.尝试run eval_1.py.

### 2024-11-25
- eval_1.py 频繁报错，很奇怪。
- dreamerv3训练过程中生成的.npz文件中，可能存在可用于render的rgb数据（取决于生成npz文件时有没有保留rgb信息），但training留下的npz文件都是egocentric camera的，且是用于replay的，视频较小，且不清晰。不确定是不是在eval步骤之后，就可以生成完整的npz文件。
- eval.py 也会频繁报错，dreamerv3的代码实在是远未工程化，很难用。再想想，应该还有别的办法。

### 2024-11-26
- 几个可能的render路径：
    - dmc.py中提到了env.render;
    - Danijar 提到的 mujoco render的设置：
        if 'MUJOCO_GL' not in os.environ:
          os.environ['MUJOCO_GL'] = 'egl'
          os.environ['MUJOCO_GL'] = 'MUJOCO_PY'
      这个感觉不太对，首先必须是在非headless的情况下才能用后者；其次，这个代码要再train或者eval一次才行，但我现在这两者都没法实现。
    - Danijar 提到了两Tensorboard,赶紧看看怎么回事。
- eval.py 跑起来了！！！！！！！！！！！操他妈的操他妈的操他妈的！！！！！啊啊啊啊啊啊啊啊！
- tensoboard 使用方法：tensorboard --logdir='~/logdir/20241117T135935-example'
  - 可以产生可视化图表，但还不知道怎么产生render的视频。
  - 用法还有待进一步摸索

### 2024-11-27
- tensorboard中可以可视化replay中生成的.npz文件，也即只有在train时保留的图像信息，才有可能被tensorboard呈现出来。
  - 试试，在obs中增加一个相机位置？
  - In https://github.com/danijar/dreamerv3/issues/2 , search:
    “If you env returns an image key as part of the observation dictionary, it will already get rendered and can be viewed in TensorBoard. Does that work for your use case?”
- 另外，试试修改定义环境的代码，也许是dmc.py，把环境render的结果保留下来。
- 注意，由于在train时，设置：env = dmc.DMC('locom_rodent_maze_forage', image=False, camera=5)
    - 导致logdir中的obs没有image，而只有log_image;
    - 导致在eval时，也必须设置image=False.
    - tensorboard中展示的画面，正是训练时产生的replay画面

### 2024-11-28
- 重新开始一轮的train, image设置为true, tensorboard中自动生成了顶部相机的视图，训练时间和上次没有区别。
    - 有没有可能我在这里中止train,修改camera_id再从当前checkpoint继续训练呢？
- dreamerv3 默认使用单卡训练，但存在使用单机多卡训练的可能，需要对JAX的设置做一些修改；
    - 如果使用RLlib,也许可以更简单的改为单机多卡设置，但需要重新摸索在RLlib中启动locom_rodent_maze_forage环境下训练的方法；

### 2024-11-29
- image = True，图像信息会进入观察空间，并且也是encoder的输入。为结合真实情况，image = True, Camera = 5 比较合理。但这样的话，估计render出来不会好看
- 新开image = True, camera = 5的训练
    - 不清楚dmc.py中所对应的camera是在哪里定义的，具体位置是什么。id为5的camera也不是egocentric_camera. 
    - 也许得尝试其他的camera_id.
        - 尝试医学院集群

### 2024-12-02
- 已在医学院集群上跑了camera= 0 2 4 训练设置；如果不设置camera id,默认是0，不是4.
  - 所有的mujoco环境都在序号为0的GPU上进行，网络训练则可以放在其他显卡上进行，不知道为什么；

- 后续步骤：
  - 1. 用rat model在 go_to_target 任务上跑
  - 2. 改rat_model的观察空间
  - 3. 改dreamerv3中agent的结构
  - 4. 改用mice model

  - 先找dreamerv3框架下，在哪里能定义task

### 2024-12-03
- !!!!!ssh集群之后，在tmux窗口下运行强化学习程序，不然ssh连接断开，训练也会终止！！！！
- 不知道为什么，打开集群上的zhangzhibo文件夹，总是会默认打开ostrichrl环境，并且好像会影响dreamerv3程序的运行

### 2024-12-04
- 

### 2024-12-09
- mice model in go-to-target(mice) task: /logdir/20241208T163146-example
- rat model in go-to-target(rat) task: /logdir/20241206T095419-example