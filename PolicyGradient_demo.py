# Prepare env
import os
import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
  with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")

# print('Installing dm_control...')
# !pip install -q dm_control>=1.0.18

# Configure dm_control to use the EGL rendering backend (requires GPU)
# %env MUJOCO_GL=egl
os.environ['MUJOCO_GL']='egl'

print('Checking that the dm_control installation succeeded...')
try:
  from dm_control import suite
  env = suite.load('cartpole', 'swingup')
  pixels = env.physics.render()
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".')
else:
  del pixels, suite



# Code RL below 
import numpy as np
from env import mice_env
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Policy Gradient (PG)
# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_mean = nn.Linear(1024, action_size)  # 输出动作的均值
        self.fc_log_std = nn.Linear(1024, action_size)  # 输出动作的对数标准差

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)  # 转化为标准差
        return mean, std
    
# 定义图像特征提取网络
class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.flatten = nn.Flatten()

        # 计算最终的特征图大小以更新全连接层参数
        # mujoco render的输出为64x64
        # 64x64 -> conv1 (30x30) -> conv2 (13x13) -> conv3 (5x5)
        self.fc = nn.Linear(64 * 5 * 5, feature_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x  # 返回低维特征向量
    
def scene_transfer(scene, conv_network, device):
    gray_scene = np.dot(scene[..., :3], [0.2989, 0.5870, 0.1140])
    scene_tensor = torch.tensor(gray_scene, dtype=torch.float32, device=device)
    scene_tensor = scene_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
    features = torch.squeeze(conv_network(scene_tensor))

    return features

# 策略更新
def update_policy(trajectory, policy_network, optimizer, checkpoint_dir, best_loss, episode, device, gamma=0.99):
    returns = []
    R = 0
    for r in trajectory['rewards'][::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, device=device)
    obs = torch.stack(trajectory['observations']).to(device)
    actions = torch.stack(trajectory['actions']).to(device)
    
    means, stds = policy_network(obs)
    if torch.isnan(means).any() or torch.isnan(stds).any():
        print("NaN detected in policy network outputs!")
        print("Means:", means)
        print("Stds:", stds)
        return best_loss  # 直接返回，避免 NaN 值进一步传播

    stds = torch.clamp(stds, min=1e-6)  # 避免除以零
    action_distribution = torch.distributions.Normal(means, stds)
    log_probs = action_distribution.log_prob(actions).sum(axis=-1)  # 对每个自由度计算对数概率并相加
    
    # returns = torch.tensor(returns).to(log_probs.device)
    loss = -(log_probs * returns).mean()  # 损失是负的期望回报
    
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)  # 梯度裁剪

    optimizer.step()

    # 权重裁剪应用在更新后，确保权重不会超出范围
    with torch.no_grad():
        for param in policy_network.parameters():
            param.clamp_(-10, 10)
    
     # Check for new best loss and save checkpoint if found
    if loss.item() < best_loss:
        best_loss = loss.item()
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode{episode+1}_loss{best_loss:.4f}.pth')
        torch.save({
            'model_state_dict': policy_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'episode': episode+1
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path} with loss: {best_loss:.4f}")

    # Print the loss after each update
    print(f"Policy Update Loss: {loss.item()}")

    return best_loss



def main():
    # Define env
    env = mice_env.rodent_maze_forage()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 手动设置vestibule的维度
    # vestibule = [self.gyro, self.accelerometer, self.velocimeter, self.world_zaxis]
    vestibule_dim=12 

    # 初始化CNN
    feature_dim=256
    conv_network = CNNFeatureExtractor(feature_dim=feature_dim).to(device)

    # Access the current positions and velocities
    qpos = env.physics.data.qpos[7:]
    qvel = env.physics.data.qvel[6:]    
    obs_size = len(qpos)+len(qvel)+feature_dim+vestibule_dim
    action_size = len(qpos)

    # 初始化策略网络
    policy_network = PolicyNetwork(obs_size, action_size).to(device)

    # # Example of loading the saved model
    # policy_network = PolicyNetwork(obs_size, action_size)
    # policy_network.load_state_dict(torch.load('policy_network.pth'))
    # policy_network.eval()  # Set to evaluation mode for testing
    
    
    # 定义联合模型参数优化
    params = list(policy_network.parameters()) + list(conv_network.parameters())    
    optimizer = optim.Adam(params, lr=1e-4)

    # 定义 checkpoint 保存路径和初始化最优 loss
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    # 定义批次大小和经验缓冲区
    batch_size = 1  # 每 n 个 episode 批量更新
    # Initialize experience buffer
    buffer = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}
    # Track rewards
    episode_rewards = []

    max_episodes = 5
    # Training loop for multiple episodes
    for episode in range(max_episodes):
        time_step = env.reset()
        qpos = time_step.observation['walker/joints_pos']
        qvel = time_step.observation['walker/joints_vel']
        scene = time_step.observation['walker/egocentric_camera']
        gyro = time_step.observation['walker/sensors_gyro']
        accel = time_step.observation['walker/sensors_accelerometer']
        veloc = time_step.observation['walker/sensors_velocimeter']
        world_z = time_step.observation['walker/world_zaxis']
        obs = torch.cat([torch.tensor(qpos, dtype=torch.float32, device=device),
                         torch.tensor(qvel, dtype=torch.float32, device=device),
                         scene_transfer(scene, conv_network, device),
                         torch.tensor(gyro, dtype=torch.float32, device=device),
                         torch.tensor(accel, dtype=torch.float32, device=device),
                         torch.tensor(veloc, dtype=torch.float32, device=device),
                         torch.tensor(world_z, dtype=torch.float32, device=device)], dim=0)

        episode_reward = 0  # Reset episode reward
        with tqdm(total = 1500, desc=f"Episode {episode+1}", leave=False) as pbar:
            while not time_step.last():
                # Sample action from policy
                mean, std = policy_network(obs)
                std = torch.clamp(std, min=1e-6, max=1.0)

                if torch.isnan(mean).any() or torch.isnan(std).any():
                    # print("Detected NaN in mean or std values.")
                    # print("Mean:", mean)
                    # print("Std:", std)
                    # 跳过当前 episode 的训练
                    print(f"Skipping episode {episode+1} due to NaN values.")
                    break
                
                action_distribution = torch.distributions.Normal(mean, std)
                action = action_distribution.sample()
                action = torch.clamp(action, min=0.01, max=0.99)
                action = action.to(device)

                # Step the environment
                next_time_step = env.step(action.cpu().numpy())

                # Store in buffer
                buffer['observations'].append(obs)
                buffer['actions'].append(action)
                buffer['rewards'].append(next_time_step.reward)

                # Update the cumulative reward for the episode
                episode_reward += next_time_step.reward

                # Prepare next observation
                qpos_next = next_time_step.observation['walker/joints_pos']
                qvel_next = next_time_step.observation['walker/joints_vel']
                scene_next = next_time_step.observation['walker/egocentric_camera']
                gyro_next = next_time_step.observation['walker/sensors_gyro']
                accel_next = next_time_step.observation['walker/sensors_accelerometer']
                veloc_next = next_time_step.observation['walker/sensors_velocimeter']
                obs = torch.cat([torch.tensor(qpos_next, dtype=torch.float32, device=device),
                                torch.tensor(qvel_next, dtype=torch.float32, device=device),
                                scene_transfer(scene_next, conv_network, device),
                                torch.tensor(gyro_next, dtype=torch.float32, device=device),
                                torch.tensor(accel_next, dtype=torch.float32, device=device),
                                torch.tensor(veloc_next, dtype=torch.float32, device=device),
                                torch.tensor(world_z, dtype=torch.float32, device=device)], dim=0)
                
                buffer['next_observations'].append(obs)
                time_step = next_time_step
                pbar.update(1)

        # Store episode reward
        episode_rewards.append(episode_reward)

        # 批处理更新策略
        if (episode + 1) % batch_size == 0:
            best_loss = update_policy(buffer, policy_network, optimizer, checkpoint_dir, best_loss, episode, device)
            buffer = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}  # 重置缓冲区
        
        # Print episode reward
        print(f"Episode {episode+1}/{max_episodes}, Reward: {episode_reward}")
    
    # Save the trained policy network
    torch.save(policy_network.state_dict(), 'policy_network.pth')

    # After all episodes, print the reward list
    print("All Episode Rewards:", episode_rewards)

if __name__ == "__main__":
    main()