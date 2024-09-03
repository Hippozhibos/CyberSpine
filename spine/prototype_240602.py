import numpy as np

import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from dm_control import suite
from dm_control import viewer
from CyberMice.assets.CyberMice import Mice

# Load the environment
env = suite.load(domain_name="cartpole", task_name="balance")

# Define a control policy
def control_policy(time_step):
    position = time_step.observation['position']
    velocity = time_step.observation['velocity']
    action = -1.0 * position - 0.1 * velocity
    return np.clip(action, -1.0, 1.0)  # Ensure action is within valid range

# Function to run the environment
def run_environment(env, policy, num_steps=1000):
    time_step = env.reset()
    for _ in range(num_steps):
        action = policy(time_step)
        time_step = env.step(action)
        env.physics.render()

# Run the viewer with the policy
viewer.launch(env, policy=control_policy)
