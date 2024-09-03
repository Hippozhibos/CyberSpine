import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

xml_path = r"CyberMice/assets/CyberMiceJointActuated_2.xml"
simend = 5 #simulation time 
print_camera_config = 0 # set to 1 to print camera config
                        # this is useful for initializing view of the model

# for callback function
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0 

model = mj.MjModel.from_xml_path(xml_path) # MuJoCo model
data = mj.MjData(model) # Mujoco data
cam = mj.MjvCamera() # Abstruct camera
opt = mj.MjvOption() # visualization options

# Define Muscle Encoder
class MuscleEncoder(nn.Module):
    def __init__(self):
        super(MuscleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.encoder(x)

# Define Muscle Decoder
class MuscleDecoder(nn.Module):
    def __init__(self):
        super(MuscleDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 120)
        )

    def forward(self, x):
        return self.decoder(x)

# Define Sensor Encoder
class SensorEncoder(nn.Module):
    def __init__(self):
        super(SensorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.encoder(x)

# Define Sensor Decoder
class SensorDecoder(nn.Module):
    def __init__(self):
        super(SensorDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 120)
        )

    def forward(self, x):
        return self.decoder(x)

# Define Muscle Autoencoder
class MuscleAutoencoder(nn.Module):
    def __init__(self):
        super(MuscleAutoencoder, self).__init__()
        self.encoder = MuscleEncoder()
        self.decoder = MuscleDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define Sensor Autoencoder
class SensorAutoencoder(nn.Module):
    def __init__(self):
        super(SensorAutoencoder, self).__init__()
        self.encoder = SensorEncoder()
        self.decoder = SensorDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Test the autoencoder networks
muscle_autoencoder = MuscleAutoencoder()
sensor_autoencoder = SensorAutoencoder()

# Print the architectures
print("Muscle Autoencoder:")
print(muscle_autoencoder)
print("\nSensor Autoencoder:")
print(sensor_autoencoder)

# Define loss function and optimizer
criterion = nn.MSELoss()
muscle_optimizer = optim.Adam(muscle_autoencoder.parameters(), lr=0.001)
sensor_optimizer = optim.Adam(sensor_autoencoder.parameters(), lr=0.001)

def controller (model,data):
    pass
    # num_actuators = model.na
    # muscle_activation = np.zeros([num_actuators,1])

    # muscle_length = data.actuator_length

def set_torque_servo(actuator_no, flag):
    if(flag==0):
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1

def set_position_servo(actuator_no, kp):
    model.actuator_gainprm[actuator_no,0] = kp
    model.actuator_biasprm[actuator_no,1] = -kp

def set_velocity_servo(actuator_no, kv):
    model.actuator_gainprm[actuator_no,0] = kv
    model.actuator_biasprm[actuator_no,2] = -kv   

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model,data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no button down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return
    
    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

# Init GLFW, creat window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# initialize the controller here. This function is called once, in the beginning
cam.azimuth = 89.83044433593757 ; cam.elevation = -20 ; cam.distance = 15.04038754800176
cam.lookat = np.array([0.0, 0.0, 0.0])

# intialize the controller
# init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

N = 100
mj.mj_forward(model,data)

# print(position_Q)

i = 0
time = 0
dt =0.0025

max_loss = 0.01
muscle_losses = []  # List to store muscle autoencoder losses
sensor_losses = []  # List to store sensor autoencoder losses


while not glfw.window_should_close(window):
    time_prev = data.time
    # time_prev = time

    # Training loop
    muscle_running_loss = 0.0
    sensor_running_loss = 0.0
    epoch = 0

    while (data.time - time_prev < 1.0/60.0):
    # while (time - time_prev < 1.0/60.0):

        num_actuators = model.na
        muscle_activation = np.zeros([num_actuators])
        muscle_activation[int(i%num_actuators)] = 0.9
        data.ctrl[:] = muscle_activation
        muscle_activation = torch.tensor(muscle_activation, dtype=torch.float32)
        
        
        mj.mj_step(model, data)

        muscle_length = data.actuator_length
        muscle_length = torch.tensor(muscle_length, dtype=torch.float32)

        
        muscle_optimizer.zero_grad()
        muscle_activation_est = muscle_autoencoder(muscle_activation)
        muscle_loss = criterion(muscle_activation_est, muscle_activation)
        muscle_loss.backward()
        muscle_optimizer.step()
        muscle_running_loss += muscle_loss.item()

        sensor_optimizer.zero_grad()
        muscle_length_est = sensor_autoencoder(muscle_length)
        sensor_loss = criterion(muscle_length_est, muscle_length)
        sensor_loss.backward()
        sensor_optimizer.step()
        muscle_running_loss += muscle_loss.item()

    i += 1

    # Calculate average losses
    avg_muscle_loss = muscle_running_loss / i
    avg_sensor_loss = sensor_running_loss / i

    muscle_losses.append(avg_muscle_loss)  # Append muscle loss to the list
    sensor_losses.append(avg_sensor_loss)  # Append sensor loss to the list


    # print(f"Epoch {epoch + 1}, Muscle Loss: {avg_muscle_loss}, Sensor Loss: {avg_sensor_loss}")

    # Check if the average loss is below the desired threshold
    if avg_muscle_loss < max_loss and avg_sensor_loss < max_loss:
        print("Training stopped: Loss reached the desired threshold.")
        break


    # if (data.time>=simend):
    #     plt.figure(1)
    #     plt.subplot(2,1,1)
    #     # plt.plot(t,qact0,'r-')
    #     # plt.plot(t,qref0,'k');
    #     plt.plot(t,np.subtract(qref0,qact0),'k')
    #     plt.ylabel("error position joint 0")
    #     plt.subplot(2,1,2)
    #     # plt.plot(t,qact1,'r-')
    #     # plt.plot(t,qref1,'k');
    #     plt.plot(t,np.subtract(qref1,qact1),'k')
    #     plt.ylabel("error position joint 0")
    #     plt.show(block=False)
    #     plt.pause(10)
    #     plt.close()
    #     break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0,0,viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =', cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance =',cam.distance)
        print('cam.lookat = np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

# Plot the losses after training
print("muscle_losses", muscle_losses)
print("\nsensor_losses", sensor_losses)
plt.plot(muscle_losses, label='Muscle Autoencoder Loss')
plt.plot(sensor_losses, label='Sensor Autoencoder Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()

glfw.terminate()

