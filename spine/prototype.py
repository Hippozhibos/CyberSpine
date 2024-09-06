import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
# print(torch.cuda.is_available())  # This should return True if GPU is enabled.
# print(torch.cuda.get_device_name(0))  # Should print the name of your GPU.
import torch.nn as nn
import torch.optim as optim

model = mj.MjModel.from_xml_path(xml_path) # MuJoCo model
data = mj.MjData(model) # Mujoco data
cam = mj.MjvCamera() # Abstruct camera
opt = mj.MjvOption() # visualization options
num_actuators = model.nu

# Define Muscle Decoder
class MuscleDecoder(nn.Module):
    def __init__(self):
        super(MuscleDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(int(num_actuators/4), int(num_actuators/2)),
            nn.ReLU(),
            nn.Linear(int(num_actuators/2), int(num_actuators))
        )

    def forward(self, x):
        return self.decoder(x)

# Define Sensor Encoder
class SensorEncoder(nn.Module):
    def __init__(self):
        super(SensorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(num_actuators), int(num_actuators/2)),
            nn.ReLU(),
            nn.Linear(int(num_actuators/2), int(num_actuators/4))
        )

    def forward(self, x):
        return self.encoder(x)

# Define KalmanEstimator
class KalmanEstimator(nn.Module):
    def __init__(self):
        super(KalmanEstimator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_actuators, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(num_actuators*2, num_actuators)
        )

    def forward(self, x):
        return self.decoder(x)
    
class FullNetwork(nn.Module):
    def __init__(self):
        super(FullNetwork, self).__init__()
        self.part1 = MuscleDecoder()
        self.part2 = KalmanEstimator()
        self.part3 = SensorEncoder()

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        return x


# Test the autoencoder networks
muscle_decoder = MuscleDecoder()
sensor_encoder = SensorEncoder()
kalman_estimator = KalmanEstimator()
full_network = FullNetwork()

# Print the architectures
print("Muscle Decoder:")
print(muscle_decoder)
print("\nSensor Encoder:")
print(sensor_encoder)
print("\nKalman Estimator:")
print(kalman_estimator)
print("\nFullNetwork:")
print(full_network)
