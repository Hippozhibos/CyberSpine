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

# 策略更新
def update_policy(trajectory, policy_network, optimizer, gamma=0.99):
    returns = []
    R = 0
    for r in trajectory['rewards'][::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    obs = torch.stack(trajectory['observations'])
    
    # print("actions size:", trajectory['actions'])
    actions = torch.stack(trajectory['actions'])
    
    means, stds = policy_network(obs)
    action_distribution = torch.distributions.Normal(means, stds)
    log_probs = action_distribution.log_prob(actions).sum(axis=-1)  # 对每个自由度计算对数概率并相加

    loss = -(log_probs * returns).mean()  # 损失是负的期望回报
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss after each update
    print(f"Policy Update Loss: {loss.item()}")



def main():
    # Define env
    env = mice_env.rodent_maze_forage()
    action_spec = env.action_spec()

    # Access the current positions and velocities
    qpos = env.physics.data.qpos[7:]
    qvel = env.physics.data.qvel[6:]    
    obs_size = len(qpos)+len(qvel)
    action_size = len(qpos)

    # 初始化策略网络
    policy_network = PolicyNetwork(obs_size, action_size)
    optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)


    # 采样数据并更新
    trajectory = {'observations': [],'discount':[], 'actions': [], 'rewards': []}
    time_step = env.reset()
    qpos = time_step.observation['walker/joints_pos']
    qvel = time_step.observation['walker/joints_vel']
    obs = torch.cat([torch.tensor(qpos, dtype=torch.float32), torch.tensor(qvel, dtype=torch.float32)], dim=0)

    max_episodes = 5

    # Initialize experience buffer
    buffer = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}

    # Track rewards
    episode_rewards = []

    # Training loop for multiple episodes
    for episode in range(max_episodes):
        time_step = env.reset()
        qpos = time_step.observation['walker/joints_pos']
        qvel = time_step.observation['walker/joints_vel']
        obs = torch.cat([torch.tensor(qpos, dtype=torch.float32), torch.tensor(qvel, dtype=torch.float32)], dim=0)

        episode_reward = 0  # Reset episode reward
        step = 0
        with tqdm(total = 1500, desc=f"Episode {episode+1}", leave=False) as pbar:
            while not time_step.last():
                # Sample action from policy
                mean, std = policy_network(obs)
                std = torch.clamp(std, min=1e-3)
                action_distribution = torch.distributions.Normal(mean, std)
                action = action_distribution.sample()
                action = torch.clamp(action, min=0.01, max=0.99)

                # Step the environment
                next_time_step = env.step(action)

                # Store in buffer
                buffer['observations'].append(obs)
                buffer['actions'].append(action)
                buffer['rewards'].append(next_time_step.reward)

                # Update the cumulative reward for the episode
                episode_reward += next_time_step.reward

                # Prepare next observation
                qpos_next = next_time_step.observation['walker/joints_pos']
                qvel_next = next_time_step.observation['walker/joints_vel']
                next_obs = torch.cat([torch.tensor(qpos_next, dtype=torch.float32), torch.tensor(qvel_next, dtype=torch.float32)], dim=0)
                buffer['next_observations'].append(next_obs)

                obs = next_obs
                time_step = next_time_step
                step += 1
                pbar.update(1)

        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # After episode ends, update policy using the buffer
        update_policy(buffer, policy_network, optimizer)

        # Reset buffer for next episode
        buffer = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}


        # Print episode reward
        print(f"Episode {episode + 1}/{max_episodes}, Reward: {episode_reward}")
        print(f"step_num: {step}")
    
    # Save the trained policy network
    torch.save(policy_network.state_dict(), 'policy_network.pth')

    # After all episodes, print the reward list
    print("All Episode Rewards:", episode_rewards)

if __name__ == "__main__":
    main()

    # # Example of loading the saved model
    # policy_network = PolicyNetwork(obs_size, action_size)
    # policy_network.load_state_dict(torch.load('policy_network.pth'))
    # policy_network.eval()  # Set to evaluation mode for testing

# ADD code to SAVE THE NN WEIGHTS!!!!!!!!!!!!