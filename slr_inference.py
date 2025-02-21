#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, SportModeState, LowCmd, IMUState, MotorState
#from math import tanh
import time
import numpy as np
import math
import sys
# sys.path.append("/home/koyo/unitree_sdk2_python")
# from unitree_sdk2py.utils.crc import CRC

# import onnx
# import onnxruntime
import os
import torch

import matplotlib.pyplot as plt

from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

from scipy.spatial.transform import Rotation as R

GRAVITY_VEC_W = np.array([0, 0, -1])  # Gravity vector in world frame

KEY_BOARD = 1



class SLRInference(Node):
    def __init__(self):
        super().__init__("rl_main_node") #node name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.load_model()

        self.init_params()

        # For lowcmd
        # self.crc = CRC()

        # Initialize ros2 subs and pubs
        self.ros2_set()

        # Initialize lowcmd
        self.init_cmd()

        # Prompt the user to enter velocity commands
        #input_str = input("Enter velocity commands separated by spaces (e.g., '0.9 -0.9 0.4'): ")
        # Convert the input string to a numpy array of floats
        #self.velocity_commands = np.array([float(x) for x in input_str.split()])

        # Start
        print("Press enter to start")
        input()
        self.standup()
        # breakpoint()

    def init_params(self):
        # Initialize velocity attributes
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.lin_vel_z = 0.0
        self.ang_vel_x = 0.0
        self.ang_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.joint_pos_ros2 = np.zeros(12)
        self.joint_pos_array = np.zeros(12)
        self.joint_vel_array = np.zeros(12)
        self.last_action = np.zeros(12)
        self.projected_gravity = np.zeros(3)
        self.height_scan = np.zeros(187)
        if KEY_BOARD:
            self.lin_x = 0.0
            self.lin_y = 0.0
            self.ang_z = 0.0
            self.velocity_commands = np.array([self.lin_x, self.lin_y, self.ang_z])
        else:
            self.velocity_commands = np.array([0.3, 0.0, 0.0])

        self.prev_time = None
        self.lowpass_ax = 0
        self.lowpass_ay = 0
        self.lowpass_az = 0
        self.ax = 0
        self.ay = 0
        self.az = 0
        self.highpass_ax = 0
        self.highpass_ay = 0
        self.highpass_az = 0
        self.old_ax = 0
        self.old_ay = 0
        self.old_az = 0
        self.lin_vel_x = 0
        self.lin_vel_y = 0
        self.lin_vel_z = 0
    
        #self.dt = 2.0
        self.dt = 0.002
        self.running_time = 0.0
        self.rate_count = 0
        self.posCmd = np.zeros(12)
        self.P_STIFF = 10.0   #80.0
        self.posStiffCmd = np.full(12, self.P_STIFF)
        self.init_pos = np.zeros(12)
            
        
        self.num_envs = 1
        self.num_obs = 528
        # Adjust the number of properties to exclude linear velocity features
        # num_props = 48 - 3
        num_props = 48
        # Store configuration parameters as class variables
        self.num_props = num_props
        self.num_hist = 10
        self.num_latents = 20
        self.num_dofs = 12

        self.obs = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_hist, self.num_props, device=self.device, dtype=torch.float)
        self.actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_history_buf = torch.zeros(self.num_envs, self.num_hist, self.num_dofs, device=self.device, dtype=torch.float)

        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

        self.clip_actions = 100
        self.clip_obs = 100

        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.action_scale = 0.25
        self.hip_scale_reduction = 0.4

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

                                          #hip thigh calf
        self.default_dof_pos = np.array([-0.1, 0.8, -1.5,  #FR
                                          0.1, 0.8, -1.5,  #FL
                                         -0.1, 1.0, -1.5,  #RR
                                          0.1, 1.0, -1.5]) #RL
        self.default_dof_pos_gym = self.ros2_to_gym(self.default_dof_pos)

        default_dof_pos_tensor = torch.from_numpy(self.default_dof_pos)
        self.default_dof_pos_tensor = default_dof_pos_tensor.reshape(self.num_envs, self.num_dofs)
        self.default_dof_pos_tensor = self.default_dof_pos_tensor.to(self.device)

        self.start_dof_pos = np.array([0.0, 0.9, -1.8, 
                                       0.0, 0.9, -1.8, 
                                       0.0, 0.9, -1.8, 
                                       0.0, 0.9, -1.8])
        
        # self.gym_default_dof_pos = self.ros2_to_gym(self.default_dof_pos)

        self.cmd_lin_x = 0.0
        self.cmd_lin_y = 0.0
        self.cmd_ang_z = 0.0
        self.max_lin_x = 0.5
        self.min_lin_x = -0.5
        self.max_lin_y = 0.5
        self.min_lin_y = -0.5
        self.max_ang_z = 1.0
        self.min_ang_z = -1.0

        self.motor_strength = np.array([0.9495, 0.9978, 1.0156, 1.0877, 1.0432, 1.0514, 0.9638, 0.9352, 1.0294, 1.0940, 1.0774, 0.9963])
        
        self.vis_count = 0

        self.kp = 35.0
        self.kd = 1.0


    def load_model(self):
        # Specify the paths
        # kp = 35.0, kp = 1.0
        policy_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/best_sdk/policy_sdk_40000.jit" 
        encoder_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/best_sdk/encoder_sdk_40000.jit"
        # policy_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/friction_mass_DRrange/policy_friction_mass_DRrange_50000.jit"
        # encoder_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/friction_mass_DRrange/encoder_friction_mass_DRrange_50000.jit"
        # policy_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/friction_mass_DRrange/policy_friction_mass_DRrange_40000.jit"
        # encoder_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/friction_mass_DRrange/encoder_friction_mass_DRrange_40000.jit"
        # policy_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/triplet_1e3/policy_triplet_1e3_40000.jit"
        # encoder_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/triplet_1e3/encoder_triplet_1e3_40000.jit"
        # policy_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/best/policy_best_40000.jit"
        # encoder_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/best/encoder_best_40000.jit"


        # Load the models
        self.policy = torch.jit.load(policy_path)
        self.encoder = torch.jit.load(encoder_path)
        self.polic = self.policy.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # Set both models to evaluation mode (for inference)
        self.policy.eval()
        self.encoder.eval()

        # model_dict = torch.load(os.path.join('/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/unitree_rl/slr_models/model_40000.pt'), map_location='cuda:0')
        # policy.load_state_dict(model_dict['model_state_dict'])
        # policy.eval()
        # self.policy = policy.to(self.device)

    def ros2_set(self):
        #ROS2 settings
        self.lowstate_topic = "/lowstate"
        self.sportstate_topic = "/sportmodestate"
        # self.lowcmd_topic = "/lowcmd"
        self.lowcmd_topic = "/lowcmd_raw"
        if KEY_BOARD:
            self.cmd_topic = "/cmd_vel"

        # Subscribers
        self.lowstate_suber = self.create_subscription(LowState, self.lowstate_topic, self.lowstate_callback, 10)
        self.sportstate_suber = self.create_subscription(SportModeState, self.sportstate_topic, self.sportstate_callback, 10)
        if KEY_BOARD:
            self.keyboard_suber = self.create_subscription(Twist, self.cmd_topic, self.keyboard_callback, 10)

        # Publisher
        self.lowcmd_puber = self.create_publisher(LowCmd, self.lowcmd_topic, 10)

        timer_period = 0.005
        # timer_period = 1.0

        self.timer_ = self.create_timer(timer_period, self.lowcmd_publish) #callback to execute every 0.02 seconds (50 Hz)
        

    def keyboard_callback(self, data):  
        self.cmd_lin_x = data.linear.x
        self.cmd_lin_y = data.linear.y
        self.cmd_ang_z = data.angular.z
        # self.cmd_lin_x = np.clip(cmd_lin_x, self.min_lin_x, self.max_lin_x)
        # self.cmd_lin_y = np.clip(cmd_lin_y, self.min_lin_y, self.max_lin_y)
        # self.cmd_ang_z = np.clip(cmd_ang_z, self.min_ang_z, self.max_ang_z)
        

    def calculate_projected_gravity(self, imu_orientation):
        """
        Calculate the projected gravity vector in the robot's frame using the inverse quaternion rotation.
        
        Args:
            imu_orientation: The orientation quaternion in (w, x, y, z). Shape is (1, 4).
        
        Returns:
            The projected gravity vector in (x, y, z). Shape is (1, 3).
        """
        projected_gravity = self.quat_rotate_inverse(imu_orientation, GRAVITY_VEC_W.reshape(1, 3))
        return projected_gravity
    
    def quat_rotate_inverse(self, q, v):
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q: The quaternion in (w, x, y, z). Shape is (1, 4).
            v: The vector in (x, y, z). Shape is (1, 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (1, 3).
        """
        q_w = q[:, 0].reshape(-1, 1)  # Shape (1, 1)
        q_vec = q[:, 1:]  # Shape (1, 3)
        a = v * (2.0 * q_w**2 - 1.0)  # Shape (1, 3)
        b = np.cross(q_vec, v) * q_w * 2.0  # Shape (1, 3)
        c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0  # Shape (1, 3)
        return a - b + c

    def process_imu_data(self, filter_coefficient=0.7):
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0, 0.0, 0.0  # Return zero velocities for the first call
        # Calculate time span since last measurement
        time_span = current_time - self.prev_time
        self.prev_time = current_time

        # Low-pass filter
        self.lowpass_ax = self.lowpass_ax * filter_coefficient + self.ax * (1 - filter_coefficient)
        self.lowpass_ay = self.lowpass_ay * filter_coefficient + self.ay * (1 - filter_coefficient)
        self.lowpass_az = self.lowpass_az * filter_coefficient + self.az * (1 - filter_coefficient)

        # High-pass filter
        self.highpass_ax = self.ax - self.lowpass_ax
        self.highpass_ay = self.ay - self.lowpass_ay
        self.highpass_az = self.az - self.lowpass_az
        # self.highpass_ax = self.ax + self.lowpass_ax
        # self.highpass_ay = self.ay + self.lowpass_ay
        # self.highpass_az = self.az + self.lowpass_az

        # Calculate speed (trapezoidal integration of acceleration)
        self.lin_vel_x = ((self.highpass_ax + self.old_ax) * time_span) / 2
        self.old_ax = self.highpass_ax
        self.lin_vel_y = ((self.highpass_ay + self.old_ay) * time_span) / 2
        self.old_ay = self.highpass_ay
        self.lin_vel_z = ((self.highpass_az + self.old_az) * time_span) / 2
        self.old_az = self.highpass_az

        base_lin_vel = np.array([self.lin_vel_x, self.lin_vel_y, self.lin_vel_z])
        base_ang_vel = np.array([self.ang_vel_x, self.ang_vel_y, self.ang_vel_z])
        self.base_lin_vel_history = np.concatenate((self.base_lin_vel_history[:, 3:], base_lin_vel.reshape((-1, 3))), axis=-1)
        self.base_ang_vel_history = np.concatenate((self.base_ang_vel_history[:, 3:], base_ang_vel.reshape((-1, 3))), axis=-1)
        # print("self.base_lin_vel_history ", self.base_lin_vel_history)
        # print("self.base_ang_vel_history ", self.base_ang_vel_history)
        #self.baselin_count += 1
        #print("self.baselin_count ", self.baselin_count)

    def ros2_to_gym(self, ros2):
        gym = [ros2[3], ros2[4], ros2[5],  
               ros2[0], ros2[1], ros2[2], 
               ros2[9], ros2[10], ros2[11], 
               ros2[6], ros2[7], ros2[8]]
        return gym

        # <input: ROS2>
        # FR_hip          0
        # FR_thig         1
        # FR_calf         2

        # FL_hip          3
        # FL_thig         4
        # FL_calf         5

        # RR_hip          6
        # RR_thig         7
        # RR_calf         8

        # RL_hip          9
        # RL_thig         10
        # RL_calf         11

        # <output: IsaacGym>
        # FL_hip_joint      3
        # FL_thigh_joint    4
        # FL_calf_joint     5

        # FR_hip_joint      0
        # FR_thigh_joint    1
        # FR_calf_joint     2

        # RL_hip_joint      9
        # RL_thigh_joint    10
        # RL_calf_joint     11

        # RR_hip_joint      6
        # RR_thigh_joint    7
        # RR_calf_joint     8

    def ros2_to_gym_tensor(self, ros2):
        gym = [ros2[3], ros2[4], ros2[5],  
               ros2[0], ros2[1], ros2[2], 
               ros2[9], ros2[10], ros2[11], 
               ros2[6], ros2[7], ros2[8]]
        return torch.tensor(gym).reshape(self.num_envs, self.num_dofs).to(device=self.device)

    def gym_to_ros2(self, gym):
        ros2 = [gym[3], gym[4], gym[5], 
                gym[0], gym[1], gym[2], 
                gym[9], gym[10], gym[11], 
                gym[6], gym[7], gym[8]]
        return ros2
    
        # <input: IsaacGym>
        # FL_hip_joint      0
        # FL_thigh_joint    1
        # FL_calf_joint     2

        # FR_hip_joint      3
        # FR_thigh_joint    4
        # FR_calf_joint     5

        # RL_hip_joint      6
        # RL_thigh_joint    7
        # RL_calf_joint     8

        # RR_hip_joint      9
        # RR_thigh_joint    10
        # RR_calf_joint     11

        # <output: ROS2>
        # FR_hip          3
        # FR_thig         4
        # FR_calf         5

        # FL_hip          0
        # FL_thig         1
        # FL_calf         2

        # RR_hip          9
        # RR_thig         10
        # RR_calf         11

        # RL_hip          6
        # RL_thig         7
        # RL_calf         8

    def gym_to_ros2_tensor(self, gym):
        ros2 = [gym[3], gym[4], gym[5], 
                gym[0], gym[1], gym[2], 
                gym[9], gym[10], gym[11], 
                gym[6], gym[7], gym[8]]
        return torch.tensor(ros2).to(device=self.device)

    def lowstate_callback(self, data):
        # Info IMU states
        imu = data.imu_state
        ang_vel_x = imu.gyroscope[0]
        ang_vel_y = imu.gyroscope[1]
        ang_vel_z = imu.gyroscope[2]
        # self.get_logger().info(f"ang_vel_x: {ang_vel_x}; ang_vel_y: {ang_vel_y}; ang_vel_z: {ang_vel_z}")
        self.ang_vel_received = True
        base_ang_vel = torch.tensor([ang_vel_x, ang_vel_y, ang_vel_z]).to(self.device)
        self.base_ang_vel = base_ang_vel.reshape(self.num_envs, 3)

        #can not get these
        # self.roll = self.imu.rpy[0]
        # self.pitch = self.imu.rpy[1]
        # self.yaw = self.imu.rpy[2]
        # self.get_logger().info(f"roll: {self.roll}; pitch: {self.pitch}; yaw: {self.yaw}")

        q0 = imu.quaternion[0]
        q1 = imu.quaternion[1]
        q2 = imu.quaternion[2]
        q3 = imu.quaternion[3]
        imu_orientation = np.array([q0, q1, q2, q3])  # Example quaternion (identity quaternion)
        imu_orientation = imu_orientation.reshape(1, 4)
        r = R.from_quat(imu_orientation)
        euler = r.as_euler('xyz', degrees=True)  # x-axis is vertical axis
        # print("Euler orientation", euler)

        projected_gravity = self.calculate_projected_gravity(imu_orientation)
        self.projected_gravity = torch.from_numpy(projected_gravity).to(self.device)

        lin_ax = imu.accelerometer[0]
        lin_ay = imu.accelerometer[1]
        lin_az = imu.accelerometer[2]
        # print("lin_ax: ", lin_ax)
        print("lin_ay: ", lin_ay) # This is forward
        print("lin_az: ", lin_az)

        motor = [None] * 12  # Initialize a list with 12 elements
        # Initialize empty lists to collect joint positions and velocities
        joint_pos_list = []
        joint_vel_list = []
        for i in range(12):
            motor[i] = data.motor_state[i]
            joint_pos = motor[i].q
            joint_vel = motor[i].dq
            self.joint_received = True
            # Append each joint position and velocity to the lists
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            #self.get_logger().info(f"num: {i}")
            #self.get_logger().info(f"joint_pos : {joint_pos}")
            #self.get_logger().info(f"joint_vel : {joint_vel}")
        dof_pos_np = np.array(joint_pos_list)
        dof_vel = np.array(joint_vel_list)
        dof_pos = torch.from_numpy(dof_pos_np).to(self.device)
        dof_vel = torch.from_numpy(dof_vel).to(self.device)
        self.dof_pos = dof_pos.reshape(self.num_envs, self.num_dofs)
        self.dof_vel = dof_vel.reshape(self.num_envs, self.num_dofs)
        # #ros2 (joint_pos_list, joint_vel_list) -> gym (joint_pos_list_gym, joint_vel_list_gym)      
        # joint_pos_list_gym = self.ros2_to_gym(joint_pos_list)
        # joint_vel_list_gym = self.ros2_to_gym(joint_vel_list)
        # #Convert the lists into numpy arrays
        # self.joint_pos_array = np.array(joint_pos_list_gym)
        # self.joint_vel_array = np.array(joint_vel_list_gym)

    def sportstate_callback(self, data):
        # Info motion states
        lin_vel_x = data.velocity[0]
        lin_vel_y = data.velocity[1]
        lin_vel_z = data.velocity[2]
        yaw = data.yaw_speed
        self.lin_vel_received = True
        base_lin_vel = torch.tensor([lin_vel_x, lin_vel_y, lin_vel_z]).to(self.device)
        self.base_lin_vel = base_lin_vel.reshape(self.num_envs, 3)
        #self.get_logger().info(f"lin_vel_x: {self.lin_vel_x}; lin_vel_y: {self.lin_vel_y}; lin_vel_z: {self.lin_vel_z}; yaw: {self.yaw}")


        # Info foot states (foot position and velocity in body frame)
        self.foot_pos = data.foot_position_body
        self.foot_vel = data.foot_speed_body
        # self.get_logger().info(f"Foot position and velcity relative to body -- num: 0; x: {self.foot_pos[0]}; y: {self.foot_pos[1]}; z: {self.foot_pos[2]}, vx: {self.foot_vel[0]}; vy: {self.foot_vel[1]}; vz: {self.foot_vel[2]}")
        # self.get_logger().info(f"Foot position and velcity relative to body -- num: 1; x: {self.foot_pos[3]}; y: {self.foot_pos[4]}; z: {self.foot_pos[5]}, vx: {self.foot_vel[3]}; vy: {self.foot_vel[4]}; vz: {self.foot_vel[5]}")
        # self.get_logger().info(f"Foot position and velcity relative to body -- num: 2; x: {self.foot_pos[6]}; y: {self.foot_pos[7]}; z: {self.foot_pos[8]}, vx: {self.foot_vel[6]}; vy: {self.foot_vel[7]}; vz: {self.foot_vel[8]}")
        # self.get_logger().info(f"Foot position and velcity relative to body -- num: 3; x: {self.foot_pos[9]}; y: {self.foot_pos[10]}; z: {self.foot_pos[11]}, vx: {self.foot_vel[9]}; vy: {self.foot_vel[10]}; vz: {self.foot_vel[11]}")


    def compute_obs(self):
        self.episode_length_buf += 1
        # print("base lin vel: ", self.base_lin_vel.device)
        # print("base ang vel: ", self.base_ang_vel.device)
        # print("projected gravity: ", self.projected_gravity.device)
        # print("commands: ", self.commands.device)
        # print("dof pos: ", self.dof_pos.device)
        # print("dof_vel: ", self.dof_vel.device)
        # print(self.base_lin_vel.shape)
        # print(self.base_ang_vel.shape)  # Check the shape of this tensor
        # print(self.projected_gravity.shape)  # Check the shape of this tensor
        # print(self.commands[:, :3].shape)  # Check the shape of this tensor
        # print((self.dof_pos - self.default_dof_pos_tensor).shape)  # Check the shape of this tensor
        # print(self.dof_vel.shape)  # Check the shape of this tensor
        # print(self.action_history_buf[:,-1].shape)  # Check the shape of this tensor
        # print("action t-1: ", self.action_history_buf[:,-1])
        # print("shape of self.dof_pos: ", self.dof_pos.shape)
        # print("shape of self.default_dof_pos_tensor: ", self.default_dof_pos_tensor.shape)
        # print("subtracted result: ", (self.dof_pos - self.default_dof_pos_tensor)[0])
        # print("scale: ", self.dof_pos_scale)
        # print("result: ", self.ros2_to_gym_tensor((self.dof_pos - self.default_dof_pos_tensor)[0] * self.dof_pos_scale))
        # breakpoint()
        
        obs_buf =torch.cat((
                self.base_lin_vel * self.lin_vel_scale,
                self.base_ang_vel  * self.ang_vel_scale,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos_tensor) * self.dof_pos_scale,
                self.dof_vel * self.dof_vel_scale,
                # self.ros2_to_gym_tensor((self.dof_pos - self.default_dof_pos_tensor)[0] * self.dof_pos_scale),
                # self.ros2_to_gym_tensor(self.dof_vel[0] * self.dof_vel_scale),
                self.action_history_buf[:,-1]),dim=-1)
        # print("obs_buf shape:", obs_buf.shape)
        # print("self.obs_history_buf shape:", self.obs_history_buf.shape)
        # input("here")
        self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.num_hist, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        # print("self.obs_history_buff: ", self.obs_history_buf)

        # self.vis_count += 1
        if self.vis_count == 1000: 
            # Visualize the key components
            self.visualize_observations(
                self.base_lin_vel.cpu().numpy(),
                self.base_ang_vel.cpu().numpy(),
                self.projected_gravity.cpu().numpy(),
                self.commands[:, :3].cpu().numpy(),
                (self.dof_pos - self.default_dof_pos_tensor).cpu().numpy(),
                self.dof_vel.cpu().numpy(),
                self.action_history_buf[:, -1].cpu().detach().numpy()
            )
    
    def visualize_observations(self, base_lin_vel, base_ang_vel, projected_gravity, commands, dof_pos_diff, dof_vel, action_history):
        fig, axs = plt.subplots(7, 1, figsize=(10, 14))
        fig.suptitle("Visualization of Observations")

        # Base Linear Velocity
        axs[0].plot(base_lin_vel[0], marker='o')
        axs[0].set_title("Base Linear Velocity (x, y, z)")
        axs[0].legend(['x', 'y', 'z'])

        # Base Angular Velocity
        axs[1].plot(base_ang_vel[0], marker='o')
        axs[1].set_title("Base Angular Velocity (x, y, z)")
        axs[1].legend(['x', 'y', 'z'])

        # Projected Gravity
        axs[2].plot(projected_gravity[0], marker='o')
        axs[2].set_title("Projected Gravity (x, y, z)")
        axs[2].legend(['x', 'y', 'z'])

        # Commands.cpu().numpy()
        axs[3].plot(commands[0], marker='o')
        axs[3].set_title("Velocity Commands (lin_x, lin_y, ang_z)")
        axs[3].legend(['lin_x', 'lin_y', 'ang_z'])

        # Difference from Default DOF Position
        axs[4].plot(dof_pos_diff[0], marker='o')
        axs[4].set_title("DOF Position Difference")
        axs[4].legend([f"Joint {i}" for i in range(dof_pos_diff.shape[1])])

        # DOF Velocity
        axs[5].plot(dof_vel[0], marker='o')
        axs[5].set_title("DOF Velocity")
        axs[5].legend([f"Joint {i}" for i in range(dof_vel.shape[1])])

        # Action History (Last Step)
        axs[6].plot(action_history[0], marker='o')
        axs[6].set_title("Action History (Last Step)")
        axs[6].legend([f"Action {i}" for i in range(action_history.shape[1])])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
        plt.show()
    
    def process_action(self, gym_action):
        move = gym_action * self.action_scale
        # for i in range(4):
        #     move[i] *= self.hip_scale_reduction
        move[0] *= self.hip_scale_reduction
        move[3] *= self.hip_scale_reduction
        move[6] *= self.hip_scale_reduction
        move[9] *= self.hip_scale_reduction
        target_pos_gym = self.default_dof_pos + move
        # target_pos_gym = self.default_dof_pos_gym + move
        # target_pos = target_pos.tolist()
        return target_pos_gym

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        #self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        #self.cfg.control.action_scale
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        # actions = self.reindex(actions)
        # actions = actions.to(self.device)
        # actions = self.gym_to_ros2(actions)
        # self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions).cpu()
        actions_np = np.array(self.actions.detach())
        actions_np = actions_np[0]
        # actions_list = actions_np.tolist()
        target_pos_gym = self.process_action(actions_np)
        # target_pos_gym *= self.motor_strength
        # target_pos = self.gym_to_ros2(target_pos_gym)

        cmd_msg = LowCmd()
        cmd_msg.head[1] = 0xEF
        cmd_msg.level_flag = 0xFF
        cmd_msg.gpio = 0

        for i in range(12):
            cmd_msg.motor_cmd[i].mode = 0x01
            # cmd_msg.motor_cmd[i].q = float(actions_list[i])  # Access the first element
            cmd_msg.motor_cmd[i].q = float(target_pos_gym[i])
            cmd_msg.motor_cmd[i].kp = self.kp  #25.0   #100.0   #25.0     # Position(rad) control kp gain
            cmd_msg.motor_cmd[i].dq = 0.0           # Target angular velocity(rad/s)
            cmd_msg.motor_cmd[i].kd = self.kd   #2.0   #0.5     # Position(rad) control kd gain
            cmd_msg.motor_cmd[i].tau = 0.0          # target torque (N.m)
        self.lowcmd_puber.publish(cmd_msg)

        self.compute_obs()

        obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
 
        return obs_buf
    

    # # Extract relevant data from observations
    # def extract(self, observations):
    #     # Extract proprioceptive data and observation history
    #     prop = observations[:, 3:self.num_props + 3]
    #     hist = observations[:, -self.num_hist * (self.num_props + 3):].view(-1, self.num_hist, self.num_props + 3)[:, :, 3:]
    #     return hist, prop

    # Extract relevant data from observations
    def extract(self, observations):
        # Extract proprioceptive data and observation history
        prop = observations[:, 3:self.num_props]
        hist = observations[:, -self.num_hist * (self.num_props):].view(-1, self.num_hist, self.num_props)[:, :, 3:]
        return hist, prop

    # Encode observation history and current properties into latent representations
    def encode(self, obs_hist, prop):
        obs = prop
        obs_hist_full = torch.cat([obs_hist[:, 1:, :], obs.unsqueeze(1)], dim=1)  # Combine history and current observations
        b, _, _ = obs_hist_full.size()

        # Ensure both tensors are of the same dtype (torch.float32 in this case)
        obs_hist_full = obs_hist_full.to(torch.float32)  # Convert to float32 (if necessary)

        self.z = self.encoder(obs_hist_full.reshape(b, -1))  # Pass through the encoder
        return obs, self.z

    # Generate deterministic actions for inference
    def act_inference(self, observations):
        obs_hist, prop = self.extract(observations)  # Extract observations
        obs, self.z = self.encode(obs_hist, prop)  # Encode observations
        actor_obs = torch.cat([self.z.detach(), obs], dim=-1)  # Combine latent and current observations
        actor_obs = actor_obs.to(torch.float32)
        actions_mean = self.policy(actor_obs)  # Get mean actions from the actor
        return actions_mean

    def lowcmd_publish(self):
        if not self.standup_completed:
            # Standup not completed, do not proceed
            return

        self.commands[:, 0] = self.cmd_lin_x
        self.commands[:, 1] = self.cmd_lin_y
        self.commands[:, 2] = self.cmd_ang_z
        self.commands[:, 3] = 0

        actions = self.act_inference(self.obs)
        # print("action: ", actions)
        
        self.obs = self.step(actions)
        # print("obs shape: ", self.obs.shape)



     
    def standup(self):
        cmd_msg = LowCmd()

        # dt = 0.0005
        # runing_time = 0.0
        dt = 0.002
        runing_time = 0.0

        while True:
            runing_time += dt
            # if (runing_time < 12.0):
            if (runing_time < 100.0):
                # Stand up in first 3 second
                #print("stand up")
                # Total time for standing up or standing down is about 1.2s
                # phase = np.tanh(runing_time / 4.8)
                phase = np.tanh(runing_time / 80.0)

                cmd_msg.head[0] = 0xFE
                cmd_msg.head[1] = 0xEF
                cmd_msg.level_flag = 0xFF
                cmd_msg.gpio = 0
                for i in range(12):
                    cmd_msg.motor_cmd[i].mode = 0x01 # 0x0A
                    cmd_msg.motor_cmd[i].q = phase * self.default_dof_pos[i] + (
                        1 - phase) * self.start_dof_pos[i]
                    cmd_msg.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                    cmd_msg.motor_cmd[i].dq = 0.0
                    cmd_msg.motor_cmd[i].kd = 3.5
                    cmd_msg.motor_cmd[i].tau = 0.0

                    # print("desired position when standing up ", cmd_msg.motor_cmd[i].q)
                    # cmd_msg.crc = Crc(cmd_msg)

                # get_crc(cmd_msg)
                self.lowcmd_puber.publish(cmd_msg)  # Publish lowcmd message
                

            else:
                break

        self.standup_completed = True
        self.get_logger().info("StandUp completed, transitioning to gait control")
 

    def init_cmd(self):
        cmd_msg = LowCmd()
        # Initialize the motor_cmd list with 20 MotorCmd objects
        for i in range(20):
            cmd_msg.motor_cmd[i].mode = 0x01 #0x0A  # Set torque mode, 0x00 is passive mode
            cmd_msg.motor_cmd[i].q = 0.0
            cmd_msg.motor_cmd[i].kp = 0.0
            cmd_msg.motor_cmd[i].dq = 0.0
            cmd_msg.motor_cmd[i].kd = 0.0
            cmd_msg.motor_cmd[i].tau = 0.0
        



   
def main(args=None):
    rclpy.init(args=args)  # Initialize rclpy
    rl_main_node = SLRInference()  # Create an instance of the SLRInference class
    rclpy.spin(rl_main_node)                               # Keeps the node running, processing incoming messages
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# <input: IsaacGym>
# FL_hip_joint      0
# FL_thigh_joint    1
# FL_calf_joint     2

# FR_hip_joint      3
# FR_thigh_joint    4
# FR_calf_joint     5

# RL_hip_joint      6
# RL_thigh_joint    7
# RL_calf_joint     8

# RR_hip_joint      9
# RR_thigh_joint    10
# RR_calf_joint     11

# <output: ROS2>
# FR_hip          3
# FR_thig         4
# FR_calf         5

# FL_hip          0
# FL_thig         1
# FL_calf         2

# RR_hip          9
# RR_thig         10
# RR_calf         11

# RL_hip          6
# RL_thig         7
# RL_calf         8




# <input: ROS2>
# FR_hip          0
# FR_thig         1
# FR_calf         2

# FL_hip          3
# FL_thig         4
# FL_calf         5

# RR_hip          6
# RR_thig         7
# RR_calf         8

# RL_hip          9
# RL_thig         10
# RL_calf         11

# <output: IsaacGym>
# FL_hip_joint      3
# FL_thigh_joint    4
# FL_calf_joint     5

# FR_hip_joint      0
# FR_thigh_joint    1
# FR_calf_joint     2

# RL_hip_joint      9
# RL_thigh_joint    10
# RL_calf_joint     11

# RR_hip_joint      6
# RR_thigh_joint    7
# RR_calf_joint     8





# <input: IsaacGym>
# FL_hip_joint      0
# FL_thigh_joint    1
# FL_calf_joint     2

# FR_hip_joint      3
# FR_thigh_joint    4
# FR_calf_joint     5

# RL_hip_joint      6
# RL_thigh_joint    7
# RL_calf_joint     8

# RR_hip_joint      9
# RR_thigh_joint    10
# RR_calf_joint     11

# <output: unitree go2 sdk>
# FR_hip_joint        3    0
# FR_thigh_joint      4
# FR_calf_joint       5 

# FL_hip_joint        0    3
# FL_thigh_joint      1
# FL_calf_joint       2

# RR_hip_joint        9    6
# RR_thigh_joint      10
# RR_calf_joint       11

# RL_hip_joint        6    9
# RL_thigh_joint      7
# RL_calf_joint       8
