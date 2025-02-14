#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, SportModeState, LowCmd, IMUState, MotorState
from math import tanh
import time
import numpy as np
# from unitree_sdk2py.utils.crc import CRC

import onnx
import onnxruntime

import matplotlib.pyplot as plt

#from sklearn.preprocessing import MinMaxScaler, StandardScaler

INFO_IMU = 1        # Set 1 to info IMU states
INFO_MOTOR = 1      # Set 1 to info motor states
INFO_VEL = 1        # set 1 to info vel states
INFO_FOOT_STATE = 0 # Set 1 to info foot states (foot position and velocity in body frame)

MEAN_STD_CHECK = 0
Z_NORMALIZATION = 0

INPUTDISPLAY = 0
OUTPUTDISPLAY = 0

SMOOTH = 0
SENSOR = 0
OUTPUT = 0
MEAN_VISUALIZATION = 0

STANDUP = 1 # Set 1 if you use publish lowcmd for standup

HIGH_FREQ = 0 # Set 1 to subscribe to low states with high frequencies (500Hz)

class RlSubPub(Node):
    def __init__(self):
        super().__init__("rl_suber_puber")

        #load onnx model
        self.load_model()

        # Initialize velocity attributes
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.lin_vel_z = 0.0
        self.ang_vel_x = 0.0
        self.ang_vel_y = 0.0
        self.ang_vel_z = 0.0

        # Flags to check if data is received
        self.lin_vel_received = False
        self.ang_vel_received = False
        self.joint_received = False

        #Check Sensors
        self.list_lin_vel_x = []
        self.list_lin_vel_y = []
        self.list_lin_vel_z = []
        self.list_ang_vel_x = []
        self.list_ang_vel_y = []
        self.list_ang_vel_z = []
        self.list_joint_pos_0 = []
        self.list_joint_pos_1 = []
        self.list_joint_pos_2 = []
        self.list_joint_pos_3 = []
        self.list_joint_pos_4 = []
        self.list_joint_pos_5 = []
        self.list_joint_pos_6 = []
        self.list_joint_pos_7 = []
        self.list_joint_pos_8 = []
        self.list_joint_pos_9 = []
        self.list_joint_pos_10 = []
        self.list_joint_pos_11 = []
        self.list_joint_vel_0 = []
        self.list_joint_vel_1 = []
        self.list_joint_vel_2 = []
        self.list_joint_vel_3 = []
        self.list_joint_vel_4 = []
        self.list_joint_vel_5 = []
        self.list_joint_vel_6 = []
        self.list_joint_vel_7 = []
        self.list_joint_vel_8 = []
        self.list_joint_vel_9 = []
        self.list_joint_vel_10 = []
        self.list_joint_vel_11 = []

        #Check Output
        self.list_raw_action_0 = []
        self.list_raw_action_1 = []
        self.list_raw_action_2 = []
        self.list_raw_action_3 = []
        self.list_raw_action_4 = []
        self.list_raw_action_5 = []
        self.list_raw_action_6 = []
        self.list_raw_action_7 = []
        self.list_raw_action_8 = []
        self.list_raw_action_9 = []
        self.list_raw_action_10 = []
        self.list_raw_action_11 = []
        self.list_orbit_target_joint_0 = []
        self.list_orbit_target_joint_1 = []
        self.list_orbit_target_joint_2 = []
        self.list_orbit_target_joint_3 = []
        self.list_orbit_target_joint_4 = []
        self.list_orbit_target_joint_5 = []
        self.list_orbit_target_joint_6 = []
        self.list_orbit_target_joint_7 = []
        self.list_orbit_target_joint_8 = []
        self.list_orbit_target_joint_9 = []
        self.list_orbit_target_joint_10 = []
        self.list_orbit_target_joint_11 = []
        self.list_ros2_target_joint_0 = []
        self.list_ros2_target_joint_1 = []
        self.list_ros2_target_joint_2 = []
        self.list_ros2_target_joint_3 = []
        self.list_ros2_target_joint_4 = []
        self.list_ros2_target_joint_5 = []
        self.list_ros2_target_joint_6 = []
        self.list_ros2_target_joint_7 = []
        self.list_ros2_target_joint_8 = []
        self.list_ros2_target_joint_9 = []
        self.list_ros2_target_joint_10 = []
        self.list_ros2_target_joint_11 = []



        #Visualization
        # Initialize an empty list to hold the arrays
        self.obs_accumulated_arrays = [] #list type
        self.lin_vel_accumulated_arrays = []
        self.ang_vel_accumulated_arrays = []
        self.projected_gravity_accumulated_arrays = []
        self.velocity_commands_accumulated_arrays = []
        self.joint_pos_accumulated_arrays = []
        self.joint_vel_accumulated_arrays = []
        self.height_scan_accumulated_arrays = []
        self.action_accumulated_arrays = []

        # Initialize last action
        #self.last_action = np.random.uniform(-0.5, 0.5, 12)
        self.last_action = np.zeros(12)
        #self.projected_gravity = np.array([-0.1, 0.0, -1.0])
        self.projected_gravity = np.array([0.0, 0.0, -1.0])
        #self.velocity_commands = np.array([0.8, -0.8, 0.4])
        self.velocity_commands = np.array([0.5, 0.0, 0.0])
        #self.height_scan = np.zeros(187)

        #ros2        
        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
            1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ],
                                        dtype=float)
        
        # orbit_standup = self.ros2_to_orbit(stand_up_joint_pos)
        # print("orbit_standup ", orbit_standup)
        # breakpoint()


        # For lowcmd
        # self.crc = CRC()

        # Initialize ros2 subs and pubs
        self.ros2_set()

        # Initialize the normalizer
        self.z_normalizer = ZNormalizer()

        # Set adjustable parameters
        self.set_params()

        # Initialize lowcmd
        self.init_cmd()

        # Prompt the user to enter velocity commands
        #input_str = input("Enter velocity commands separated by spaces (e.g., '0.9 -0.9 0.4'): ")
        # Convert the input string to a numpy array of floats
        #self.velocity_commands = np.array([float(x) for x in input_str.split()])

        # stand up
        if STANDUP:
            # Start
            print("Press enter to start")
            input()
            self.standup()

        if INPUTDISPLAY:
            self.input_display()
        
        if OUTPUTDISPLAY:
            self.output_display()


    def load_model(self):
        # load onnx model and start session
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_12000.onnx" #default settings
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_6500_specify_joint_order.onnx" #specify joint order afterwards
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_model_height_noise_0.5.onnx" #get back to default height noise afterwards
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_model_specify_initial_pos_vel.onnx" #specify initial pos and vel afterwards
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_model_non_projected_gravity.onnx" #get back to defalt projected gravity setting
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_model_action_scale_0.2.onnx" #get back to defau action scale
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_action_scale_0.2.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_normalized.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_start.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_reverse_hip.onnx" #non height and reserve hip afterwards
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_reverse_hip_even.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_reverse_hip_even_bh10_05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_reverse_hip_even_bh10_03_flat.onnx"
        ##model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_flat.onnx" #worked best so far
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_04_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_04_flat_DR1.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_04_jpl3_06_flat_DR2.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_flat_DR3.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_uc1_calf_flat_DR2.onnx"
        ##model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_rear_thigh_bh10_03_uc1_calf_flat.onnx" #worked best so far
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_jd01_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_rear_thigh_bh10_03_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_034_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_hip_reduction05_flat.onnx" #worked best so far without self._scale
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_034_hip_reduction05_flat.onnx"
        #---model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_reduction05_flat.onnx" #worked best so far without self._scale
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_hip_thigh_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_thigh_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_increase2_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_04_hip_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_reduction08_flat.onnx"
        ##model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_reduction02_flat.onnx" #this is good
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_rear_thigh_bh10_034_thigh_reduction025_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_034_thigh_reduction02_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_action_scale0.1_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_reduction04_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_reduction04_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_034_thigh_reduction04_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_thigh_reduction06_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_jd005_thigh_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_even_bh10_03_jpl3_0.9_ucl_calf_thigh_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_normalised_non_height_orbit_pos_bh10_03_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_normalised_non_height_mujoco_pos_bh10_03_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_normalised_non_height_mujoco_pos_bh10_03_thigh_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_normalised_non_height_mujoco_pos_bh10_03_hip_reduction05_flat.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_mujoco_pos_bh10_03_thigh_reduction05_flat_DR2.onnx"
        #model_path = "/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/files/policy_non_height_non_projectedgravity_mujoco_pos_bh10_03_thigh_reduction05_flat_DR2.onnx"

        #model_path = "/home/koyo/go1_ws/src/unitree_rl/src/unitree_rl/files/policy_critic_mass_low_pos_fat107_mass_heading_false_z03_hip_reduction05_flat.onnx"
        #model_path = "/home/koyo/go1_ws/src/unitree_rl/src/unitree_rl/files/policy_critic_mass_low_pos_Ffat107_mass_heading_false_z03_hip_reduction05_flat.onnx"
        #model_path = "/home/koyo/go1_ws/src/unitree_rl/src/unitree_rl/files/policy_critic_mass_low_pos2_fat107_mass_heading_false_z03_hip_reduction05_flat.onnx"
        #model_path = "/home/koyo/go1_ws/src/unitree_rl/src/unitree_rl/files/policy_critic_mass_low_pos2_fat_ol105_mass_heading_false_z03_lin_range06_hip_reduction05_flat.onnx"
        model_path = "/home/koyo/go1_ws/src/unitree_rl/src/unitree_rl/files/policy_go1_actuator_rapid_pos_go1reward_mass_non_reduction_flat.onnx"

        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))
        self.sess = onnxruntime.InferenceSession(model.SerializeToString())

        # input data
        self.input_name = self.sess.get_inputs()[0].name
        print("Input name  :", self.input_name)
        input_shape = self.sess.get_inputs()[0].shape
        print("Input shape :", input_shape)
        input_type = self.sess.get_inputs()[0].type
        print("Input type  :", input_type)
        # Input name  : obs
        # Input shape : [1, 235] -> [1, 48]
        # Input type  : tensor(float)

        # output data
        self.output_name = self.sess.get_outputs()[0].name
        print("Output name  :", self.output_name)  
        output_shape = self.sess.get_outputs()[0].shape
        print("Output shape :", output_shape)
        output_type = self.sess.get_outputs()[0].type
        print("Output type  :", output_type)
        # Output name  : actions
        # Output shape : [1, 12]
        # Output type  : tensor(float)
        
    def ros2_set(self):
        #ROS2 settings
        self.lowstate_topic = "/lowstate"
        self.sportstate_topic = "/sportmodestate"
        self.lowcmd_topic = "/lowcmd"

        # Subscribers
        self.lowstate_suber = self.create_subscription(LowState, self.lowstate_topic, self.lowstate_callback, 10)
        self.sportstate_suber = self.create_subscription(SportModeState, self.sportstate_topic, self.sportstate_callback, 10)

        # Publisher
        self.lowcmd_puber = self.create_publisher(LowCmd, self.lowcmd_topic, 10)

        #timer_period = 0.001 #callback to execute every 0.001 seconds (1000 Hz)
        #works with policy_non_height_even_bh10_03_flat.onnx
        timer_period = 0.005 #callback to execute every 0.005 seconds (200 Hz) 
        #timer_period = 0.01 #callback to execute every 0.01 seconds (100 Hz)
        #timer_period = 0.02 #callback to execute every 0.02 seconds (50 Hz)
        #timer_period = 0.05 #callback to execute every 0.05 seconds (20 Hz)
        #timer_period = 0.1 #callback to execute every 0.1 seconds (10 Hz)
        #timer_period = 0.25 #callback to execute every 0.25 seconds (4 Hz)
        #timer_period = 0.5
        #timer_period = 1.0
        self.timer_ = self.create_timer(timer_period, self.lowcmd_callback) #callback to execute every 0.02 seconds (50 Hz)

    def set_params(self):
        self.mean_std_interval = 10000
        self.visualization_interval = 1000
        self.sensor_interval = 1000
        self.output_interval = 1000
        self.lowcmd_count = 0
        #rear_thigh = 0.708813 #woked with policy_non_height_rear_thigh_bh10_03_uc1_calf_flat.onnx
        rear_thigh = 0.608813

        # #ros2                                  hip        thigh       calf
        # self.stand_up_joint_pos = np.array([0.00571868, 0.608813,  -1.21763,                #FR
        #                                     -0.00571868, 0.608813,  -1.21763,               #FL
        #                                      0.00571868, rear_thigh, -1.21763,  #0.608813     #RR
        #                                     -0.00571868, rear_thigh,  -1.21763],dtype=float)  #RL
        
        #                                         #hip thigh calf
        # self.stand_up_joint_pos = np.array([-0.1, 0.6, -1.5,  #FR
        #                                         0.1, 0.6, -1.5,  #FL
        #                                         -0.1, 1.2, -1.5,  #RR
        #                                         0.1, 1.2, -1.5]) #RL
        
                                                #hip thigh calf
        self.stand_up_joint_pos = np.array([-0.1, 0.4, -1.4,  #FR
                                            0.1, 0.4, -1.4,  #FL
                                            -0.1, 1.0, -1.4,  #RR
                                            0.1, 1.0, -1.4]) #RL
        
        #0.708813 woked with policy_non_height_rear_thigh_bh10_03_uc1_calf_flat.onnx
        
        # self.stand_up_joint_pos_hip_reverse = np.array([-0.00571868, 0.608813,  -1.21763,                #FR
        #                                                 0.00571868, 0.608813,  -1.21763,                 #FL
        #                                                -0.00571868, 0.608813, -1.21763,    #-1.11763      #RR
        #                                                 0.00571868, 0.608813,  -1.21763],dtype=float)    #RL
        #orbit
        self.orbit_stand_up_joint_pos = self.ros2_to_orbit(self.stand_up_joint_pos)
        #print("self.orbit_stand_up_joint_pos ", self.orbit_stand_up_joint_pos)
        # self.up_offset = np.array([-0.00571868, 0.00571868, -0.00571868, 0.00571868,          
        #                            0.608813,   0.608813,    0.608813,   0.608813,                
        #                            -1.21763,   -1.21763,    -1.01763,   -1.01763], dtype=float)


        #                                               FL           FR           RL          RR
        # self.orbit_stand_up_joint_pos = np.array([-0.00571868, 0.00571868, -0.00571868, 0.00571868,               hip
        #                                             0.608813,   0.608813,    0.608813,   0.608813,                thigh
        #                                             -1.21763,   -1.21763,    -1.01763,   -1.01763], dtype=float)  calf
        # # #                                             FL           FR           RL          RR
        # self.orbit_stand_up_joint_pos = np.array([-0.00571868, 0.00571868, -0.00571868, 0.00571868, 
        #                                             0.608813,   0.608813,    0.608813,   0.608813, 
        #                                             -1.21763,   -1.21763,    -1.21763,   -1.21763], dtype=float)

        #self._action_scale = 0.2 #worked with policy_non_height_even_bh10_03_flat.onnx 
        # and policy_non_height_rear_thigh_bh10_03_uc1_calf_flat.onnx 
        # and policy_non_height_even_bh10_03_thigh_reduction05_flat.onnx

        self._action_scale = 0.25
        self._hip_scale_reduction = 0.5
        self._thigh_scale_reduction = 0.5
        self._hip_thigh_scale_reduction = 0.5
        #self._scale = 0.8 #worked with policy_non_height_even_bh10_03_flat.onnx
        #self._scale = 0.6 #worked with policy_non_height_rear_thigh_bh10_03_uc1_calf_flat.onnx
        #self._scale = 0.8

    def lowstate_callback(self, data):
        if INFO_IMU:
            # Info IMU states
            self.imu = data.imu_state
            self.ang_vel_x = self.imu.gyroscope[0]
            self.ang_vel_y = self.imu.gyroscope[1]
            self.ang_vel_z = self.imu.gyroscope[2]
            #self.get_logger().info(f"ang_vel_x: {self.imu.gyroscope[0]}; ang_vel_y: {self.imu.gyroscope[1]}; ang_vel_z: {self.imu.gyroscope[2]}")
            self.ang_vel_received = True

            #can not get these
            # self.roll = self.imu.rpy[0]
            # self.pitch = self.imu.rpy[1]
            # self.yaw = self.imu.rpy[2]
            # self.get_logger().info(f"roll: {self.roll}; pitch: {self.pitch}; yaw: {self.yaw}")
           
        if INFO_MOTOR:
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

            #ros2 (joint_pos_list, joint_vel_list) -> orbit (joint_pos_list_orbit, joint_vel_list_orbit)      
            joint_pos_list_orbit = self.ros2_to_orbit(joint_pos_list)
            joint_vel_list_orbit = self.ros2_to_orbit(joint_vel_list)
            #Convert the lists into numpy arrays
            self.joint_pos_array = np.array(joint_pos_list_orbit)
            self.joint_vel_array = np.array(joint_vel_list_orbit)

    def sportstate_callback(self, data):
        if INFO_VEL:
            # Info motion states
            self.lin_vel_x = data.velocity[0]
            self.lin_vel_y = data.velocity[1]
            self.lin_vel_z = data.velocity[2]
            self.yaw = data.yaw_speed
            self.lin_vel_received = True
            #self.get_logger().info(f"lin_vel_x: {self.lin_vel_x}; lin_vel_y: {self.lin_vel_y}; lin_vel_z: {self.lin_vel_z}; yaw: {self.yaw}")

        if INFO_FOOT_STATE:
            # Info foot states (foot position and velocity in body frame)
            self.foot_pos = data.foot_position_body
            self.foot_vel = data.foot_speed_body
            self.get_logger().info(f"Foot position and velcity relative to body -- num: 0; x: {self.foot_pos[0]}; y: {self.foot_pos[1]}; z: {self.foot_pos[2]}, vx: {self.foot_vel[0]}; vy: {self.foot_vel[1]}; vz: {self.foot_vel[2]}")
            self.get_logger().info(f"Foot position and velcity relative to body -- num: 1; x: {self.foot_pos[3]}; y: {self.foot_pos[4]}; z: {self.foot_pos[5]}, vx: {self.foot_vel[3]}; vy: {self.foot_vel[4]}; vz: {self.foot_vel[5]}")
            self.get_logger().info(f"Foot position and velcity relative to body -- num: 2; x: {self.foot_pos[6]}; y: {self.foot_pos[7]}; z: {self.foot_pos[8]}, vx: {self.foot_vel[6]}; vy: {self.foot_vel[7]}; vz: {self.foot_vel[8]}")
            self.get_logger().info(f"Foot position and velcity relative to body -- num: 3; x: {self.foot_pos[9]}; y: {self.foot_pos[10]}; z: {self.foot_pos[11]}, vx: {self.foot_vel[9]}; vy: {self.foot_vel[10]}; vz: {self.foot_vel[11]}")


    def check_sensors(self):
        lin_vel_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_sensor_normalised/lin_vel_{self.lowcmd_count}.png"
        ang_vel_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_sensor_normalised/ang_vel_{self.lowcmd_count}.png"
        joint_pos_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_sensor_normalised/joint_pos_{self.lowcmd_count}.png"
        joint_vel_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_sensor_normalised/joint_vel_{self.lowcmd_count}.png"

        # Create a figure with two subplots (1 row, 3 columns)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        # Plot the graph
        ax1.plot(self.list_lin_vel_x)  # Provide custom x-values
        ax1.set_title('lin_vel_x')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('lin_vel_x')

        # Plot the graph
        ax2.plot(self.list_lin_vel_y)
        ax2.set_title('lin_vel_y')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('lin_vel_y')

        # Plot the graph
        ax3.plot(self.list_lin_vel_z)
        ax3.set_title('lin_vel_z')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('lin_vel_z')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Construct the file path using an f-string with the variable
        #lin_vel_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/policy_model_specify_initial_pos_vel/lin_vel_{self.lowcmd_count}.png'
        # Save the combined figure to a file
        plt.savefig(lin_vel_file_path)
        # Clear the figure to free up memory
        plt.clf()

        # Create a figure with two subplots (1 row, 3 columns)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        # Plot the graph
        ax1.plot(self.list_ang_vel_x)  # Provide custom x-values
        ax1.set_title('ang_vel_x')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('ang_vel_x')

        # Plot the graph
        ax2.plot(self.list_ang_vel_y)
        ax2.set_title('ang_vel_y')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('ang_vel_y')

        # Plot the graph
        ax3.plot(self.list_ang_vel_z)
        ax3.set_title('ang_vel_z')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('ang_vel_z')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Construct the file path using an f-string with the variable
        #ang_vel_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/policy_model_specify_initial_pos_vel/ang_vel_{self.lowcmd_count}.png'
        # Save the combined figure to a file
        plt.savefig(ang_vel_file_path)
        # Clear the figure to free up memory
        plt.clf()


        # Create a figure with 4 rows and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        # Flatten the 2D array of axes for easy iteration
        axs = axs.flatten()

        # Plot the graph
        axs[0].plot(self.list_joint_pos_0)  # Provide custom x-values
        axs[0].set_title('Joint Position 0')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Position')
        # Plot the graph
        axs[1].plot(self.list_joint_pos_1)  # Provide custom x-values
        axs[1].set_title('Joint Position 1')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Position')
        # Plot the graph
        axs[2].plot(self.list_joint_pos_2)  # Provide custom x-values
        axs[2].set_title('Joint Position 2')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Position')
        # Plot the graph
        axs[3].plot(self.list_joint_pos_3)  # Provide custom x-values
        axs[3].set_title('Joint Position 3')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Position')
        # Plot the graph
        axs[4].plot(self.list_joint_pos_4)  # Provide custom x-values
        axs[4].set_title('Joint Position 4')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Position')
        # Plot the graph
        axs[5].plot(self.list_joint_pos_5)  # Provide custom x-values
        axs[5].set_title('Joint Position 5')
        axs[5].set_xlabel('Time Step')
        axs[5].set_ylabel('Position')
        # Plot the graph
        axs[6].plot(self.list_joint_pos_6)  # Provide custom x-values
        axs[6].set_title('Joint Position 6')
        axs[6].set_xlabel('Time Step')
        axs[6].set_ylabel('Position')
        # Plot the graph
        axs[7].plot(self.list_joint_pos_7)  # Provide custom x-values
        axs[7].set_title('Joint Position 7')
        axs[7].set_xlabel('Time Step')
        axs[7].set_ylabel('Position')
        # Plot the graph
        axs[8].plot(self.list_joint_pos_8)  # Provide custom x-values
        axs[8].set_title('Joint Position 8')
        axs[8].set_xlabel('Time Step')
        axs[8].set_ylabel('Position')
        # Plot the graph
        axs[9].plot(self.list_joint_pos_9)  # Provide custom x-values
        axs[9].set_title('Joint Position 9')
        axs[9].set_xlabel('Time Step')
        axs[9].set_ylabel('Position')
        # Plot the graph
        axs[10].plot(self.list_joint_pos_10)  # Provide custom x-values
        axs[10].set_title('Joint Position 10')
        axs[10].set_xlabel('Time Step')
        axs[10].set_ylabel('Position')
        # Plot the graph
        axs[11].plot(self.list_joint_pos_11)  # Provide custom x-values
        axs[11].set_title('Joint Position 11')
        axs[11].set_xlabel('Time Step')
        axs[11].set_ylabel('Position')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the combined figure to a file
        plt.savefig(joint_pos_file_path)
        # Clear the figure to free up memory
        plt.clf()


        # Create a figure with 4 rows and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        # Flatten the 2D array of axes for easy iteration
        axs = axs.flatten()

        # Plot the graph
        axs[0].plot(self.list_joint_vel_0)  # Provide custom x-values
        axs[0].set_title('Joint Velocity 0')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Velocity')
        # Plot the graph
        axs[1].plot(self.list_joint_vel_1)  # Provide custom x-values
        axs[1].set_title('Joint Velocity 1')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Velocity')
        # Plot the graph
        axs[2].plot(self.list_joint_vel_2)  # Provide custom x-values
        axs[2].set_title('Joint Velocity 2')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Velocity')
        # Plot the graph
        axs[3].plot(self.list_joint_vel_3)  # Provide custom x-values
        axs[3].set_title('Joint Velocity 3')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Velocity')
        # Plot the graph
        axs[4].plot(self.list_joint_vel_4)  # Provide custom x-values
        axs[4].set_title('Joint Velocity 4')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Velocity')
        # Plot the graph
        axs[5].plot(self.list_joint_vel_5)  # Provide custom x-values
        axs[5].set_title('Joint Velocity 5')
        axs[5].set_xlabel('Time Step')
        axs[5].set_ylabel('Velocity')
        # Plot the graph
        axs[6].plot(self.list_joint_vel_6)  # Provide custom x-values
        axs[6].set_title('Joint Velocity 6')
        axs[6].set_xlabel('Time Step')
        axs[6].set_ylabel('Velocity')
        # Plot the graph
        axs[7].plot(self.list_joint_vel_7)  # Provide custom x-values
        axs[7].set_title('Joint Velocity 7')
        axs[7].set_xlabel('Time Step')
        axs[7].set_ylabel('Velocity')
        # Plot the graph
        axs[8].plot(self.list_joint_vel_8)  # Provide custom x-values
        axs[8].set_title('Joint Velocity 8')
        axs[8].set_xlabel('Time Step')
        axs[8].set_ylabel('Velocity')
        # Plot the graph
        axs[9].plot(self.list_joint_vel_9)  # Provide custom x-values
        axs[9].set_title('Joint Velocity 9')
        axs[9].set_xlabel('Time Step')
        axs[9].set_ylabel('Velocity')
        # Plot the graph
        axs[10].plot(self.list_joint_vel_10)  # Provide custom x-values
        axs[10].set_title('Joint Velocity 10')
        axs[10].set_xlabel('Time Step')
        axs[10].set_ylabel('Velocity')
        # Plot the graph
        axs[11].plot(self.list_joint_vel_11)  # Provide custom x-values
        axs[11].set_title('Joint Velocity 11')
        axs[11].set_xlabel('Time Step')
        axs[11].set_ylabel('Velocity')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the combined figure to a file
        plt.savefig(joint_vel_file_path)
        # Clear the figure to free up memory
        plt.clf()

        breakpoint()
    
    def check_output(self):
        raw_action_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_output/raw_action_{self.lowcmd_count}.png"
        orbit_target_joint_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_output/orbit_target_joint_{self.lowcmd_count}.png"
        ros2_target_joint_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_output/ros2_target_joint_{self.lowcmd_count}.png"

        # Create a figure with 4 rows and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        # Flatten the 2D array of axes for easy iteration
        axs = axs.flatten()
        # Plot the graph
        axs[0].plot(self.list_raw_action_0)  # Provide custom x-values
        axs[0].set_title('Raw Action 0')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Position')
        # Plot the graph
        axs[1].plot(self.list_raw_action_1)  # Provide custom x-values
        axs[1].set_title('Raw Action 1')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Position')
        # Plot the graph
        axs[2].plot(self.list_raw_action_2)  # Provide custom x-values
        axs[2].set_title('Raw Action 2')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Position')
        # Plot the graph
        axs[3].plot(self.list_raw_action_3)  # Provide custom x-values
        axs[3].set_title('Raw Action 3')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Position')
        # Plot the graph
        axs[4].plot(self.list_raw_action_4)  # Provide custom x-values
        axs[4].set_title('Raw Action 4')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Position')
        # Plot the graph
        axs[5].plot(self.list_raw_action_5)  # Provide custom x-values
        axs[5].set_title('Raw Action 5')
        axs[5].set_xlabel('Time Step')
        axs[5].set_ylabel('Position')
        # Plot the graph
        axs[6].plot(self.list_raw_action_6)  # Provide custom x-values
        axs[6].set_title('Raw Action 6')
        axs[6].set_xlabel('Time Step')
        axs[6].set_ylabel('Position')
        # Plot the graph
        axs[7].plot(self.list_raw_action_7)  # Provide custom x-values
        axs[7].set_title('Raw Action 7')
        axs[7].set_xlabel('Time Step')
        axs[7].set_ylabel('Position')
        # Plot the graph
        axs[8].plot(self.list_raw_action_8)  # Provide custom x-values
        axs[8].set_title('Raw Action 8')
        axs[8].set_xlabel('Time Step')
        axs[8].set_ylabel('Position')
        # Plot the graph
        axs[9].plot(self.list_raw_action_9)  # Provide custom x-values
        axs[9].set_title('Raw Action 9')
        axs[9].set_xlabel('Time Step')
        axs[9].set_ylabel('Position')
        # Plot the graph
        axs[10].plot(self.list_raw_action_10)  # Provide custom x-values
        axs[10].set_title('Raw Action 10')
        axs[10].set_xlabel('Time Step')
        axs[10].set_ylabel('Position')
        # Plot the graph
        axs[11].plot(self.list_raw_action_11)  # Provide custom x-values
        axs[11].set_title('Raw Action 11')
        axs[11].set_xlabel('Time Step')
        axs[11].set_ylabel('Position')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the combined figure to a file
        plt.savefig(raw_action_file_path)
        # Clear the figure to free up memory
        plt.clf()


        # Create a figure with 4 rows and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        # Flatten the 2D array of axes for easy iteration
        axs = axs.flatten()
        # Plot the graph
        axs[0].plot(self.list_orbit_target_joint_0)  # Provide custom x-values
        axs[0].set_title('Joint Position 0')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Position')
        # Plot the graph
        axs[1].plot(self.list_orbit_target_joint_1)  # Provide custom x-values
        axs[1].set_title('Joint Position 1')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Position')
        # Plot the graph
        axs[2].plot(self.list_orbit_target_joint_2)  # Provide custom x-values
        axs[2].set_title('Joint Position 2')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Position')
        # Plot the graph
        axs[3].plot(self.list_orbit_target_joint_3)  # Provide custom x-values
        axs[3].set_title('Joint Position 3')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Position')
        # Plot the graph
        axs[4].plot(self.list_orbit_target_joint_4)  # Provide custom x-values
        axs[4].set_title('Joint Position 4')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Position')
        # Plot the graph
        axs[5].plot(self.list_orbit_target_joint_5)  # Provide custom x-values
        axs[5].set_title('Joint Position 5')
        axs[5].set_xlabel('Time Step')
        axs[5].set_ylabel('Position')
        # Plot the graph
        axs[6].plot(self.list_orbit_target_joint_6)  # Provide custom x-values
        axs[6].set_title('Joint Position 6')
        axs[6].set_xlabel('Time Step')
        axs[6].set_ylabel('Position')
        # Plot the graph
        axs[7].plot(self.list_orbit_target_joint_7)  # Provide custom x-values
        axs[7].set_title('Joint Position 7')
        axs[7].set_xlabel('Time Step')
        axs[7].set_ylabel('Position')
        # Plot the graph
        axs[8].plot(self.list_orbit_target_joint_8)  # Provide custom x-values
        axs[8].set_title('Joint Position 8')
        axs[8].set_xlabel('Time Step')
        axs[8].set_ylabel('Position')
        # Plot the graph
        axs[9].plot(self.list_orbit_target_joint_9)  # Provide custom x-values
        axs[9].set_title('Joint Position 9')
        axs[9].set_xlabel('Time Step')
        axs[9].set_ylabel('Position')
        # Plot the graph
        axs[10].plot(self.list_orbit_target_joint_10)  # Provide custom x-values
        axs[10].set_title('Joint Position 10')
        axs[10].set_xlabel('Time Step')
        axs[10].set_ylabel('Position')
        # Plot the graph
        axs[11].plot(self.list_orbit_target_joint_11)  # Provide custom x-values
        axs[11].set_title('Joint Position 11')
        axs[11].set_xlabel('Time Step')
        axs[11].set_ylabel('Position')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the combined figure to a file
        plt.savefig(orbit_target_joint_file_path)
        # Clear the figure to free up memory
        plt.clf()

        # Create a figure with 4 rows and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        # Flatten the 2D array of axes for easy iteration
        axs = axs.flatten()
        # Plot the graph
        axs[0].plot(self.list_ros2_target_joint_0)  # Provide custom x-values
        axs[0].set_title('Joint Position 0')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Position')
        # Plot the graph
        axs[1].plot(self.list_ros2_target_joint_1)  # Provide custom x-values
        axs[1].set_title('Joint Position 1')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Position')
        # Plot the graph
        axs[2].plot(self.list_ros2_target_joint_2)  # Provide custom x-values
        axs[2].set_title('Joint Position 2')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Position')
        # Plot the graph
        axs[3].plot(self.list_ros2_target_joint_3)  # Provide custom x-values
        axs[3].set_title('Joint Position 3')
        axs[3].set_xlabel('Time Step')
        axs[3].set_ylabel('Position')
        # Plot the graph
        axs[4].plot(self.list_ros2_target_joint_4)  # Provide custom x-values
        axs[4].set_title('Joint Position 4')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Position')
        # Plot the graph
        axs[5].plot(self.list_ros2_target_joint_5)  # Provide custom x-values
        axs[5].set_title('Joint Position 5')
        axs[5].set_xlabel('Time Step')
        axs[5].set_ylabel('Position')
        # Plot the graph
        axs[6].plot(self.list_ros2_target_joint_6)  # Provide custom x-values
        axs[6].set_title('Joint Position 6')
        axs[6].set_xlabel('Time Step')
        axs[6].set_ylabel('Position')
        # Plot the graph
        axs[7].plot(self.list_ros2_target_joint_7)  # Provide custom x-values
        axs[7].set_title('Joint Position 7')
        axs[7].set_xlabel('Time Step')
        axs[7].set_ylabel('Position')
        # Plot the graph
        axs[8].plot(self.list_ros2_target_joint_8)  # Provide custom x-values
        axs[8].set_title('Joint Position 8')
        axs[8].set_xlabel('Time Step')
        axs[8].set_ylabel('Position')
        # Plot the graph
        axs[9].plot(self.list_ros2_target_joint_9)  # Provide custom x-values
        axs[9].set_title('Joint Position 9')
        axs[9].set_xlabel('Time Step')
        axs[9].set_ylabel('Position')
        # Plot the graph
        axs[10].plot(self.list_ros2_target_joint_10)  # Provide custom x-values
        axs[10].set_title('Joint Position 10')
        axs[10].set_xlabel('Time Step')
        axs[10].set_ylabel('Position')
        # Plot the graph
        axs[11].plot(self.list_ros2_target_joint_11)  # Provide custom x-values
        axs[11].set_title('Joint Position 11')
        axs[11].set_xlabel('Time Step')
        axs[11].set_ylabel('Position')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the combined figure to a file
        plt.savefig(ros2_target_joint_file_path)
        # Clear the figure to free up memory
        plt.clf()


    def process_orbit_action(self, orbit_action):
        # # hip scale reduction
        # move = orbit_action * self._action_scale
        # for i in range(4):
        #     move[i] *= self._hip_scale_reduction
        # target_pos = self.orbit_stand_up_joint_pos + move

        # # thigh scale reduction
        # move = orbit_action * self._action_scale
        # for i in range(4):
        #     move[i+4] *= self._thigh_scale_reduction
        # target_pos = self.orbit_stand_up_joint_pos + move

        # # hip and thigh scale reduction
        # move = orbit_action * self._action_scale
        # for i in range(8):
        #     move[i] *= self._hip_thigh_scale_reduction
        # target_pos = self.orbit_stand_up_joint_pos + move

        move = orbit_action * self._action_scale
        target_pos = self.orbit_stand_up_joint_pos + move
        return target_pos
        #return self.orbit_stand_up_joint_pos

    def input_display(self):
        # Initialize the plots
        plt.ion()  # Turn on interactive mode

        # First figure with four subplots
        fig1, axs1 = plt.subplots(2, 1)
        fig1.suptitle('lin_vel, ang_vel')
        self.lines1 = []

        x_lin_ang = np.array([0, 1, 2])
        lin_vel = np.zeros(3)
        ang_vel = np.zeros(3)

        # Create lines for first figure
        self.lines1.append(axs1[0].plot(x_lin_ang, lin_vel)[0])
        axs1[0].set_title('lin_vel')
        axs1[0].set_ylim(-1.5, 1.5)

        self.lines1.append(axs1[1].plot(x_lin_ang, ang_vel)[0])
        axs1[1].set_title('ang_vel')
        axs1[1].set_ylim(-3.0, 3.0)

        # Adjust layout
        fig1.tight_layout()
        #fig1.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase space between plot


        # Second figure with two subplots
        fig2, axs2 = plt.subplots(2, 1)
        fig2.suptitle('joint_pos, joint_vel')
        self.lines2 = []

        x_joint = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        joint_pos = np.zeros(12)
        joint_vel = np.zeros(12)

        self.lines2.append(axs2[0].plot(x_joint, joint_pos)[0])
        axs2[0].set_title('joint_pos')
        axs2[0].set_ylim(-4.0, 4.0)

        self.lines2.append(axs2[1].plot(x_joint, joint_vel)[0])
        axs2[1].set_title('joint_vel')
        axs2[1].set_ylim(-7.0, 7.0)

        # Adjust layout
        fig2.tight_layout()
        #fig1.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase space between plot


        # Third figure with two subplots
        fig3, axs3 = plt.subplots(2, 1)
        fig3.suptitle('projected_gravity and velocity_commands')
        self.lines3 = []

        # Create lines for second figure
        self.lines3.append(axs3[0].plot(x_lin_ang, self.projected_gravity)[0])
        axs3[0].set_title('projected_gravity')

        self.lines3.append(axs3[1].plot(x_lin_ang, self.velocity_commands)[0])
        axs3[1].set_title('velocity_commands')

        # Set plot limits
        for ax in axs3:
            ax.set_ylim(-1.5, 1.5)

        # Adjust layout
        fig3.tight_layout()
        #fig1.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase space between plot

    def output_display(self):
        # Initialize the plots
        plt.ion()  # Turn on interactive mode

        # Fourth figure with two subplots
        fig4, axs4 = plt.subplots(1, 1)
        fig4.suptitle('action')
        self.lines4 = []

        # Initialization
        x_joint = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        action = np.zeros(12)

        self.lines4.append(axs4.plot(x_joint, action)[0])
        axs4.set_title('action')
        axs4.set_ylim(-7.0, 7.0)

        # Adjust layout
        fig4.tight_layout()

        # Fifth figure with two subplots
        fig5, axs5 = plt.subplots(1, 1)
        fig5.suptitle('ros2_actions')
        self.lines5 = []

        # Initialization
        ros2_action = np.zeros(12)

        self.lines5.append(axs5.plot(x_joint, ros2_action)[0])
        axs5.set_title('ros2_action')
        axs5.set_ylim(-7.0, 7.0)

        # Adjust layout
        fig5.tight_layout()

    def update_subplot(self, line, new_data):
        line.set_ydata(new_data)
        # line.axes.relim()  # Recompute the data limits
        # line.axes.autoscale_view()  # Rescale the view
        plt.draw()
        plt.pause(0.1)  # Adjust pause time to control the update frequency

    def ros2_to_orbit(self, ros2):
        orbit = [ros2[3], ros2[0], ros2[9], ros2[6],
                 ros2[4], ros2[1], ros2[10], ros2[7],
                 ros2[5], ros2[2], ros2[11], ros2[8]]
        return orbit
    
    def orbit_to_ros2(self, obit):
        ros2 = [obit[1], obit[5], obit[9], 
                obit[0], obit[4], obit[8], 
                obit[3], obit[7], obit[11], 
                obit[2], obit[6], obit[10]]
        return ros2

    def lowcmd_callback(self):
        if not (self.lin_vel_received and self.ang_vel_received and self.joint_received):
            # If data has not been received from both topics, do nothing
            return

        self.lowcmd_count += 1
        cmd_msg = LowCmd()

        if SMOOTH:
            self.list_lin_vel_x.append(self.lin_vel_x)
            self.list_lin_vel_y.append(self.lin_vel_y)
            self.list_lin_vel_z.append(self.lin_vel_z)
            self.list_ang_vel_x.append(self.ang_vel_x)
            self.list_ang_vel_y.append(self.ang_vel_y)
            self.list_ang_vel_z.append(self.ang_vel_z)
            
            self.joint_pos_accumulated_arrays.append(self.joint_pos_array)
            self.joint_vel_accumulated_arrays.append(self.joint_vel_array)

            self.smoothing()
            #breakpoint()
        
        if Z_NORMALIZATION:
            #integrate the followings as each of them is obtained in different callbacks
            lin_vel = np.array([self.lin_vel_x, self.lin_vel_y, self.lin_vel_z])
            ang_vel = np.array([self.ang_vel_x, self.ang_vel_y, self.ang_vel_z])
            #print("lin_vel before normalization ", lin_vel)
            # print("ang_vel before normalization ", ang_vel)
            #print("joint_pos before normalization ", self.joint_pos_array)
            # print("joint_vel before normalization ", self.joint_vel_array)

            #append values at the time and
            self.lin_vel_accumulated_arrays.append(lin_vel)
            self.ang_vel_accumulated_arrays.append(ang_vel)
            self.joint_pos_accumulated_arrays.append(self.joint_pos_array)
            self.joint_vel_accumulated_arrays.append(self.joint_vel_array)

            #compute mean and std dynamically based on the above four values
            #update self._mean and self._std
            self.compute_mean_std()

            #get normalized values
            self.normalized_lin_vel = self.z_normalizer.normalize(lin_vel, self.lin_vel_mean_array, self.lin_vel_std_array)
            self.normalized_ang_vel = self.z_normalizer.normalize(ang_vel, self.ang_vel_mean_array, self.ang_vel_std_array)
            self.normalized_joint_pos = self.z_normalizer.normalize(self.joint_pos_array, self.joint_pos_mean_array, self.joint_pos_std_array)
            self.normalized_joint_vel = self.z_normalizer.normalize(self.joint_vel_array, self.joint_vel_mean_array, self.joint_vel_std_array)
        
            #print("lin_vel after normalization ", self.normalized_lin_vel)
            # print("ang_vel after normalization ", self.normalized_ang_vel)
            #print("joint_pos after normalization ", self.normalized_joint_pos)
            # print("joint_vel after normalization ", self.normalized_joint_vel)

            #For varification
            #denormalized_lin_vel = self.normalizer.denormalize(self.normalized_lin_vel, self.lin_vel_mean_array, self.lin_vel_std_array)
            #print("denormalized_lin_vel ", denormalized_lin_vel)

            #breakpoint()

        if SENSOR:
            if Z_NORMALIZATION:
                self.list_lin_vel_x.append(self.normalized_lin_vel[0])
                self.list_lin_vel_y.append(self.normalized_lin_vel[1])
                self.list_lin_vel_z.append(self.normalized_lin_vel[2])
                self.list_ang_vel_x.append(self.normalized_ang_vel[0])
                self.list_ang_vel_y.append(self.normalized_ang_vel[1])
                self.list_ang_vel_z.append(self.normalized_ang_vel[2])
                self.list_joint_pos_0.append(self.normalized_joint_pos[0])
                self.list_joint_pos_1.append(self.normalized_joint_pos[1])
                self.list_joint_pos_2.append(self.normalized_joint_pos[2])
                self.list_joint_pos_3.append(self.normalized_joint_pos[3])
                self.list_joint_pos_4.append(self.normalized_joint_pos[4])
                self.list_joint_pos_5.append(self.normalized_joint_pos[5])
                self.list_joint_pos_6.append(self.normalized_joint_pos[6])
                self.list_joint_pos_7.append(self.normalized_joint_pos[7])
                self.list_joint_pos_8.append(self.normalized_joint_pos[8])
                self.list_joint_pos_9.append(self.normalized_joint_pos[9])
                self.list_joint_pos_10.append(self.normalized_joint_pos[10])
                self.list_joint_pos_11.append(self.normalized_joint_pos[11])
                self.list_joint_vel_0.append(self.normalized_joint_vel[0])
                self.list_joint_vel_1.append(self.normalized_joint_vel[1])
                self.list_joint_vel_2.append(self.normalized_joint_vel[2])
                self.list_joint_vel_3.append(self.normalized_joint_vel[3])
                self.list_joint_vel_4.append(self.normalized_joint_vel[4])
                self.list_joint_vel_5.append(self.normalized_joint_vel[5])
                self.list_joint_vel_6.append(self.normalized_joint_vel[6])
                self.list_joint_vel_7.append(self.normalized_joint_vel[7])
                self.list_joint_vel_8.append(self.normalized_joint_vel[8])
                self.list_joint_vel_9.append(self.normalized_joint_vel[9])
                self.list_joint_vel_10.append(self.normalized_joint_vel[10])
                self.list_joint_vel_11.append(self.normalized_joint_vel[11])
            else:
                self.list_lin_vel_x.append(self.lin_vel_x)
                self.list_lin_vel_y.append(self.lin_vel_y)
                self.list_lin_vel_z.append(self.lin_vel_z)
                self.list_ang_vel_x.append(self.ang_vel_x)
                self.list_ang_vel_y.append(self.ang_vel_y)
                self.list_ang_vel_z.append(self.ang_vel_z)
                self.list_joint_pos_0.append(self.joint_pos_array[0])
                self.list_joint_pos_1.append(self.joint_pos_array[1])
                self.list_joint_pos_2.append(self.joint_pos_array[2])
                self.list_joint_pos_3.append(self.joint_pos_array[3])
                self.list_joint_pos_4.append(self.joint_pos_array[4])
                self.list_joint_pos_5.append(self.joint_pos_array[5])
                self.list_joint_pos_6.append(self.joint_pos_array[6])
                self.list_joint_pos_7.append(self.joint_pos_array[7])
                self.list_joint_pos_8.append(self.joint_pos_array[8])
                self.list_joint_pos_9.append(self.joint_pos_array[9])
                self.list_joint_pos_10.append(self.joint_pos_array[10])
                self.list_joint_pos_11.append(self.joint_pos_array[11])
                self.list_joint_vel_0.append(self.joint_vel_array[0])
                self.list_joint_vel_1.append(self.joint_vel_array[1])
                self.list_joint_vel_2.append(self.joint_vel_array[2])
                self.list_joint_vel_3.append(self.joint_vel_array[3])
                self.list_joint_vel_4.append(self.joint_vel_array[4])
                self.list_joint_vel_5.append(self.joint_vel_array[5])
                self.list_joint_vel_6.append(self.joint_vel_array[6])
                self.list_joint_vel_7.append(self.joint_vel_array[7])
                self.list_joint_vel_8.append(self.joint_vel_array[8])
                self.list_joint_vel_9.append(self.joint_vel_array[9])
                self.list_joint_vel_10.append(self.joint_vel_array[10])
                self.list_joint_vel_11.append(self.joint_vel_array[11])

            if self.lowcmd_count == self.sensor_interval:
                self.check_sensors()
                #breakpoint()

        if INPUTDISPLAY:
            #self.standup()
            #integrate the followings as each of them is obtained in different callbacks
            lin_vel = np.array([self.lin_vel_x, self.lin_vel_y, self.lin_vel_z])
            ang_vel = np.array([self.ang_vel_x, self.ang_vel_y, self.ang_vel_z])

            self.update_subplot(self.lines1[0], lin_vel)
            self.update_subplot(self.lines1[1], ang_vel)

            self.update_subplot(self.lines2[0], self.joint_pos_array)
            self.update_subplot(self.lines2[1], self.joint_vel_array)

            self.update_subplot(self.lines3[0], self.projected_gravity)
            self.update_subplot(self.lines3[1], self.velocity_commands)
            
            time.sleep(0.1)  # Simulate time delay for real-time update

        obs = self.integrate_subs()
        obs = obs.astype(np.float32)
        #print("obs: ", obs) #input should be numpy array with float 32 values
        #print("obs type: ", type(obs))
        #print("obs shape: ", obs.shape) #(1, 235)
        action = self.sess.run([self.output_name], {self.input_name: obs}) #orbit
        # print("action: ", action) 
        # print("action type", type(action)) #<class 'list'>
        # action = action[0]
        # print("action: ", action)
        # print("action type", type(action)) #<class 'numpy.ndarray'>
        action = action[0].flatten() #<class 'numpy.ndarray'> #orbit
        # print("action: ", action) 
        # print("action type", type(action)) #<class 'numpy.ndarray'>
        #print("action[0]", action[0])
        #print("action[0] type", type(action[0])) #<class 'numpy.ndarray'>
        #print("action[0] dtype", action[0].dtype) #float32
        #breakpoint()

        if MEAN_VISUALIZATION:
            lin_vel = obs[0, 0:3]
            ang_vel = obs[0, 3:6]
            projected_gravity = obs[0, 6:9]
            velocity_commands = obs[0, 9:12]
            joint_pos = obs[0, 12:24]
            joint_vel = obs[0, 24:36]
            #height_scan = obs[0, 48:]

            #visualization, data analysis
            self.obs_accumulated_arrays.append(obs)
            self.action_accumulated_arrays.append(action)
            self.lin_vel_accumulated_arrays.append(lin_vel)
            self.ang_vel_accumulated_arrays.append(ang_vel)
            self.projected_gravity_accumulated_arrays.append(projected_gravity)
            self.velocity_commands_accumulated_arrays.append(velocity_commands)
            self.joint_pos_accumulated_arrays.append(joint_pos)
            self.joint_vel_accumulated_arrays.append(joint_vel)
            #self.height_scan_accumulated_arrays.append(height_scan)

            if self.lowcmd_count == self.visualization_interval:
                self.visualize_mean()
                #breakpoint()

        if MEAN_STD_CHECK:
            lin_vel = obs[0, 0:3]
            ang_vel = obs[0, 3:6]
            joint_pos = obs[0, 12:24]
            joint_vel = obs[0, 24:36]
            #height_scan = obs[0, 48:]

            #visualization, data analysis
            self.lin_vel_accumulated_arrays.append(lin_vel)
            self.ang_vel_accumulated_arrays.append(ang_vel)
            self.joint_pos_accumulated_arrays.append(joint_pos)
            self.joint_vel_accumulated_arrays.append(joint_vel)

            if self.lowcmd_count == self.mean_std_interval:
                self.compute_mean_std()
                breakpoint()

        #orbit 
        self.last_action = np.array(action) #record lonast action as a numpy array 
        self.last_action = self.last_action.reshape(12) #change the shape into (12,)

        #orbit
        target_joints_orbit = self.process_orbit_action(action) #<class 'numpy.ndarray'>
        #orbit -> ros2
        self.target_joints_ros2 = self.orbit_to_ros2(target_joints_orbit)
        print("self.target_joints_ros2: ", self.target_joints_ros2)
        print("type: ", type(self.target_joints_ros2))
        breakpoint()
        
        for i in range(12):
            cmd_msg.motor_cmd[i].q =float(self.target_joints_ros2[i]) #float(self.stand_up_joint_pos[i])  # Target angular(rad)
            cmd_msg.motor_cmd[i].kp = 20.0  #25.0   #100.0   #25.0     # Position(rad) control kp gain
            cmd_msg.motor_cmd[i].dq = 0.0           # Target angular velocity(rad/s)
            cmd_msg.motor_cmd[i].kd = 0.5   #2.0   #0.5     # Position(rad) control kd gain
            cmd_msg.motor_cmd[i].tau = 0.0          # target torque (N.m)
            #cmd_msg.crc = self.crc.Crc(cmd_msg)
        # cmd_msg.motor_cmd[0].tau = -0.65
        # cmd_msg.motor_cmd[3].tau = 0.65
        # cmd_msg.motor_cmd[6].tau = -0.65
        # cmd_msg.motor_cmd[9].tau = 0.65
        #breakpoint()    

        self.lowcmd_puber.publish(cmd_msg)
        #self.get_logger().info("Publishing the low level command")

        #orbit -> ros2 
        self.ros2_raw_action = self.orbit_to_ros2(action)
        self.check_hip()
        self.check_thigh()
        self.check_calf()
        #print("self.lin_vel_x ", self.lin_vel_x )

        if OUTPUT:
            self.list_raw_action_0.append(action[0])
            self.list_raw_action_1.append(action[1])
            self.list_raw_action_2.append(action[2])
            self.list_raw_action_3.append(action[3])
            self.list_raw_action_4.append(action[4])
            self.list_raw_action_5.append(action[5])
            self.list_raw_action_6.append(action[6])
            self.list_raw_action_7.append(action[7])
            self.list_raw_action_8.append(action[8])
            self.list_raw_action_9.append(action[9])
            self.list_raw_action_10.append(action[10])
            self.list_raw_action_11.append(action[11])

            self.list_orbit_target_joint_0.append(target_joints_orbit[0])
            self.list_orbit_target_joint_1.append(target_joints_orbit[1])
            self.list_orbit_target_joint_2.append(target_joints_orbit[2])
            self.list_orbit_target_joint_3.append(target_joints_orbit[3])
            self.list_orbit_target_joint_4.append(target_joints_orbit[4])
            self.list_orbit_target_joint_5.append(target_joints_orbit[5])
            self.list_orbit_target_joint_6.append(target_joints_orbit[6])
            self.list_orbit_target_joint_7.append(target_joints_orbit[7])
            self.list_orbit_target_joint_8.append(target_joints_orbit[8])
            self.list_orbit_target_joint_9.append(target_joints_orbit[9])
            self.list_orbit_target_joint_10.append(target_joints_orbit[10])
            self.list_orbit_target_joint_11.append(target_joints_orbit[11])

            self.list_ros2_target_joint_0.append(self.target_joints_ros2[0])
            self.list_ros2_target_joint_1.append(self.target_joints_ros2[1])
            self.list_ros2_target_joint_2.append(self.target_joints_ros2[2])
            self.list_ros2_target_joint_3.append(self.target_joints_ros2[3])
            self.list_ros2_target_joint_4.append(self.target_joints_ros2[4])
            self.list_ros2_target_joint_5.append(self.target_joints_ros2[5])
            self.list_ros2_target_joint_6.append(self.target_joints_ros2[6])
            self.list_ros2_target_joint_7.append(self.target_joints_ros2[7])
            self.list_ros2_target_joint_8.append(self.target_joints_ros2[8])
            self.list_ros2_target_joint_9.append(self.target_joints_ros2[9])
            self.list_ros2_target_joint_10.append(self.target_joints_ros2[10])
            self.list_ros2_target_joint_11.append(self.target_joints_ros2[11])

            if self.lowcmd_count == self.output_interval:
                self.check_output()
                breakpoint()

        if OUTPUTDISPLAY:
            #self.standup()
            self.update_subplot(self.lines4[0], action)
            self.update_subplot(self.lines5[0], target_joints_orbit)
            
            time.sleep(0.1)  # Simulate time delay for real-time update

        self.lin_vel_received = False
        self.ang_vel_received = False
        self.joint_received = False
    
    def check_hip(self):
        #orbit - ros2
        self.joint_pos_array_ros2 = self.orbit_to_ros2(self.joint_pos_array)

        print("hip-------")
        #ros2
        # print("default ", self.stand_up_joint_pos[0], self.stand_up_joint_pos[3], self.stand_up_joint_pos[6], self.stand_up_joint_pos[9])
        # print("target pos ", self.target_joints_ros2[0], self.target_joints_ros2[3], self.target_joints_ros2[6], self.target_joints_ros2[9])
        # print("Current ", self.joint_pos_array_ros2[0], self.joint_pos_array_ros2[3], self.joint_pos_array_ros2[6], self.joint_pos_array_ros2[9])
        print("Raw action ", self.ros2_raw_action[0], self.ros2_raw_action[3], self.ros2_raw_action[6], self.ros2_raw_action[9])

    def check_thigh(self):
        #orbit - ros2
        self.joint_pos_array_ros2 = self.orbit_to_ros2(self.joint_pos_array)

        print("thigh-------")
        #ros2
        # print("default ", self.stand_up_joint_pos[1], self.stand_up_joint_pos[4], self.stand_up_joint_pos[7], self.stand_up_joint_pos[10])
        # print("target pos ", self.target_joints_ros2[1], self.target_joints_ros2[4], self.target_joints_ros2[7], self.target_joints_ros2[10])
        # print("Current ", self.joint_pos_array_ros2[1], self.joint_pos_array_ros2[4], self.joint_pos_array_ros2[7], self.joint_pos_array_ros2[10])
        print("Raw action ", self.ros2_raw_action[1], self.ros2_raw_action[4], self.ros2_raw_action[7], self.ros2_raw_action[10])

    def check_calf(self):
        #orbit - ros2
        self.joint_pos_array_ros2 = self.orbit_to_ros2(self.joint_pos_array)

        print("calf-------")
        #ros2
        # print("default ", self.stand_up_joint_pos[2], self.stand_up_joint_pos[5], self.stand_up_joint_pos[8], self.stand_up_joint_pos[11])
        # print("target pos ", self.target_joints_ros2[2], self.target_joints_ros2[5], self.target_joints_ros2[8], self.target_joints_ros2[11])
        # print("Current ", self.joint_pos_array_ros2[2], self.joint_pos_array_ros2[5], self.joint_pos_array_ros2[8], self.joint_pos_array_ros2[11])
        print("Raw action ", self.ros2_raw_action[2], self.ros2_raw_action[5], self.ros2_raw_action[8], self.ros2_raw_action[11])

    def compute_mean_std(self):
        lin_vel_stacked_arrays = np.vstack(self.lin_vel_accumulated_arrays)
        ang_vel_stacked_arrays = np.vstack(self.ang_vel_accumulated_arrays)
        joint_pos_stacked_arrays = np.vstack(self.joint_pos_accumulated_arrays)
        joint_vel_stacked_arrays = np.vstack(self.joint_vel_accumulated_arrays)

        self.lin_vel_mean_array = np.mean(lin_vel_stacked_arrays, axis=0)
        self.lin_vel_std_array = np.std(lin_vel_stacked_arrays, axis=0, ddof=0)
        self.ang_vel_mean_array = np.mean(ang_vel_stacked_arrays, axis=0)
        self.ang_vel_std_array = np.std(ang_vel_stacked_arrays, axis=0, ddof=0)
        self.joint_pos_mean_array = np.mean(joint_pos_stacked_arrays, axis=0)
        self.joint_pos_std_array = np.std(joint_pos_stacked_arrays, axis=0, ddof=0)
        self.joint_vel_mean_array = np.mean(joint_vel_stacked_arrays, axis=0)
        self.joint_vel_std_array = np.std(joint_vel_stacked_arrays, axis=0, ddof=0)

        #print("lin_vel_stacked_arrays ", lin_vel_stacked_arrays)
        #print("lin_vel_mean_array ", self.lin_vel_mean_array)
        # print("lin_vel_std_array ", lin_vel_std_array)
        # print("ang_vel_stacked_arrays ", ang_vel_stacked_arrays)
        # print("ang_vel_mean_array ", ang_vel_mean_array)
        # print("ang_vel_std_array ", ang_vel_std_array)
        # print("joint_pos_stacked_arrays ", joint_pos_stacked_arrays)
        # print("joint_pos_mean_array ", joint_pos_mean_array)
        # print("joint_pos_std_array ", joint_pos_std_array)
        # print("joint_vel_stacked_arrays ", joint_vel_stacked_arrays)
        # print("joint_vel_mean_array ", joint_vel_mean_array)
        # print("joint_vel_std_array ", joint_vel_std_array)
        #breakpoint()

    def visualize_mean(self):
        #print("self.obs_accumulated_arrays: ", self.obs_accumulated_arrays)
        #print("self.action_accumulated_arrays: ", self.action_accumulated_arrays)
        # Stack the arrays vertically
        obs_stacked_arrays = np.vstack(self.obs_accumulated_arrays)
        action_stacked_arrays = np.vstack(self.action_accumulated_arrays)
        lin_vel_stacked_arrays = np.vstack(self.lin_vel_accumulated_arrays)
        ang_vel_stacked_arrays = np.vstack(self.ang_vel_accumulated_arrays)
        projected_gravity_stacked_arrays = np.vstack(self.projected_gravity_accumulated_arrays)
        velocity_commands_stacked_arrays = np.vstack(self.velocity_commands_accumulated_arrays)
        joint_pos_stacked_arrays = np.vstack(self.joint_pos_accumulated_arrays)
        joint_vel_stacked_arrays = np.vstack(self.joint_vel_accumulated_arrays)
        #height_scan_stacked_arrays = np.vstack(self.height_scan_accumulated_arrays)

        #print("action_stacked_arrays shape", action_stacked_arrays.shape) #(self.visualization_interval, 12)
        #print("action_stacked_arrays", action_stacked_arrays)
        #list_action_stacked_arrays = action_stacked_arrays.tolist() #2D list (self.visualization_interval, 12)
        #print("list_action_stacked_arrays[0]", list_action_stacked_arrays[0])
        #print("list_action_stacked_arrays", list_action_stacked_arrays)
        #breakpoint()

        # Calculate the mean along the first axis (axis=0)
        obs_mean_array = np.mean(obs_stacked_arrays, axis=0)
        action_mean_array = np.mean(action_stacked_arrays, axis=0)
        lin_vel_mean_array = np.mean(lin_vel_stacked_arrays, axis=0)
        ang_vel_mean_array = np.mean(ang_vel_stacked_arrays, axis=0)
        projected_gravity_mean_array = np.mean(projected_gravity_stacked_arrays, axis=0)
        velocity_commands_mean_array = np.mean(velocity_commands_stacked_arrays, axis=0)
        joint_pos_mean_array = np.mean(joint_pos_stacked_arrays, axis=0)
        joint_vel_mean_array = np.mean(joint_vel_stacked_arrays, axis=0)
        #height_scan_mean_array = np.mean(height_scan_stacked_arrays, axis=0)
        #print
        #print("Mean of obs accumulated arrays:", obs_mean_array)
        #print("Mean of action accumulated arrays:", action_mean_array)

        obs_action_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_mean/obs_action_average_{self.lowcmd_count}.png"
        each_obs_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_mean/each_obs_average_{self.lowcmd_count}.png"
        each_preset_obs_file_path = f"/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/check_mean/each_preset_obs_average_{self.lowcmd_count}.png"

        ##Visualization
        # Create a figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        # Plot the first graph (Obs average)
        ax1.plot(obs_mean_array)  # Provide custom x-values
        ax1.set_title('Obs Average')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 3, 6, 9, 12, 24, 36, 48]
        #custom_ticks = [0, 3, 6, 9, 12, 24, 36, 48, 234]
        ax1.set_xticks(custom_ticks)
        # Set custom x-axis tick labels font size
        ax1.set_xticklabels(custom_ticks, fontsize=8)

        # Plot the second graph (Action average)
        ax2.plot(action_mean_array)
        ax2.set_title('Action Average')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ax2.set_xticks(custom_ticks)
        # Add the mean value text to the corner of the subplot
        # (0.95, 0.95) corresponds to the top-right corner in normalized axis coordinates
        # Ensure the mean_value is included in the formatted string
        whole_action_mean = np.mean(action_mean_array)
        ax2.text(0.98, 0.1, f'Mean: {whole_action_mean:.2f}', transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Construct the file path using an f-string with the variable
        #obs_action_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/height_noise_0.5/obs_action_average_{self.lowcmd_count}_d5.png'
        #obs_action_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/action_scale_0.2/obs_action_average_{self.lowcmd_count}.png'
        #obs_action_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/specify_initial_pos_vel/obs_action_average_{self.lowcmd_count}.png'
        # Save the combined figure to a file
        plt.savefig(obs_action_file_path)
        # Clear the figure to free up memory
        plt.clf()


        # Create a figure with seven subplots arranged vertically (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        # Plot the graph
        axes[0, 0].plot(lin_vel_mean_array, color='blue')
        axes[0, 0].set_title('lin_vel average')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3]
        axes[0, 0].set_xticks(custom_ticks)
        # Set custom x-axis scale
        axes[0, 0].set_xlim([-0.2, len(lin_vel_mean_array) - 0.8])  # Set limits based on the number
        # Plot the graph
        axes[0, 1].plot(ang_vel_mean_array, color='green')
        axes[0, 1].set_title('ang_vel average')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3]
        axes[0, 1].set_xticks(custom_ticks)
        # Set custom x-axis scale
        axes[0, 1].set_xlim([-0.2, len(ang_vel_mean_array) - 0.8])  # Set limits based on the number
        # Plot the graph
        axes[1, 0].plot(joint_pos_mean_array, color='magenta')
        axes[1, 0].set_title('joint_pos average')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        axes[1, 0].set_xticks(custom_ticks)
        # Plot the graph
        axes[1, 1].plot(joint_vel_mean_array, color='red')
        axes[1, 1].set_title('joint_vel average')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        axes[1, 1].set_xticks(custom_ticks)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Construct the file path using an f-string with the variable
        #each_obs_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/height_noise_0.5/each_obs_average_{self.lowcmd_count}_d5.png'
        #each_obs_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/action_scale_0.2/each_obs_average_{self.lowcmd_count}.png'
        #each_obs_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/specify_initial_pos_vel/each_obs_average_{self.lowcmd_count}.png'
        # Save the combined figure to a file
        plt.savefig(each_obs_file_path)
        # Clear the figure to free up memory
        plt.clf()


        # Create a figure with seven subplots arranged vertically (1 row, 2 columns)
        fig, axes = plt.subplots(2, 1, figsize=(10, 15))
        # Plot the graph
        axes[0].plot(projected_gravity_mean_array, color='red')
        axes[0].set_title('projected_gravity (pre-set) average')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3]
        axes[0].set_xticks(custom_ticks)
        # Set custom x-axis scale
        axes[0].set_xlim([-0.2, len(projected_gravity_mean_array) - 0.8])  # Set limits based on the number
        # Plot the graph
        axes[1].plot(velocity_commands_mean_array, color='black')
        axes[1].set_title('velocity_commands (pre-set) average')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Average')
        # set custom x-axis ticks
        custom_ticks = [0, 1, 2, 3]
        axes[1].set_xticks(custom_ticks)
        # Set custom x-axis scale
        axes[1].set_xlim([-0.2, len(velocity_commands_mean_array) - 0.8])  # Set limits based on the number
        # # Plot the graph
        # axes[2].plot(height_scan_mean_array, color='purple')
        # axes[2].set_title('height_scan (pre-set) average')
        # axes[2].set_xlabel('Index')
        # axes[2].set_ylabel('Average')
        # # set custom x-axis ticks
        # custom_ticks = [0, 187]
        # axes[2].set_xticks(custom_ticks)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Construct the file path using an f-string with the variable
        #each_preset_obs_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/height_noise_0.5/each_preset_obs_average_{self.lowcmd_count}.png'
        #each_preset_obs_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/action_scale_0.2/each_preset_obs_average_{self.lowcmd_count}.png'
        #each_preset_obs_file_path = f'/home/koyo/unitree_ros2/unitree_ws/src/unitree_rl/data/specify_initial_pos_vel/each_preset_obs_average_{self.lowcmd_count}.png'
        # Save the combined figure to a file
        plt.savefig(each_preset_obs_file_path)
        # Clear the figure to free up memory
        plt.clf()

        #finish
        #rclpy.shutdown()

    def integrate_subs(self):
        if Z_NORMALIZATION:
            integrated_subs = np.concatenate([
            self.normalized_lin_vel ,  # lin_vel
            self.normalized_ang_vel ,  # ang_vel
            self.projected_gravity,  # projected gravity (flattened if necessary)
            self.velocity_commands,  # velocity commands (flattened if necessary)
            self.normalized_joint_pos,  # joint_pos (flattened if necessary)
            self.normalized_joint_vel,  # joint_vel (flattened if necessary)
            self.last_action,  # last_actions (flattened if necessary)
            #self.height_scan  # height_scan (flattened if necessary)
        ])
        else:
            lin_vel = np.array([self.lin_vel_x, self.lin_vel_y, self.lin_vel_z])
            ang_vel = np.array([self.ang_vel_x, self.ang_vel_y, self.ang_vel_z])

            integrated_subs = np.concatenate([
                lin_vel,  # lin_vel
                ang_vel,  # ang_vel
                #self.projected_gravity,  # projected gravity (flattened if necessary)
                self.velocity_commands,  # velocity commands (flattened if necessary)
                self.joint_pos_array,  # joint_pos (flattened if necessary)
                self.joint_vel_array,  # joint_vel (flattened if necessary)
                self.last_action,  # last_actions (flattened if necessary)
                #self.height_scan  # height_scan (flattened if necessary)
            ])

        #convert rank one into rank two
        integrated_subs = integrated_subs.reshape(1, -1) 

        return integrated_subs
    

    def standup(self):
        cmd_msg = LowCmd()

        dt = 0.002
        runing_time = 0.0

        while True:
            runing_time += dt
            if (runing_time < 1.0):
                # Stand up in first 3 second
                #print("stand up")
                # Total time for standing up or standing down is about 1.2s
                phase = np.tanh(runing_time / 0.4)
                for i in range(12):
                    cmd_msg.motor_cmd[i].q = phase * self.stand_up_joint_pos[i] + (
                        1 - phase) * self.stand_down_joint_pos[i]
                    cmd_msg.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                    cmd_msg.motor_cmd[i].dq = 0.0
                    cmd_msg.motor_cmd[i].kd = 3.5
                    cmd_msg.motor_cmd[i].tau = 0.0

                    #print("desired position when standing up ", cmd_msg.motor_cmd[i].q)
                    # cmd_msg.crc = self.crc.Crc(cmd_msg)

                #get_crc(cmd_msg)
                self.lowcmd_puber.publish(cmd_msg)  # Publish lowcmd message
                

            else:
                break


        
    def init_cmd(self):
        cmd_msg = LowCmd()
        # Initialize the motor_cmd list with 20 MotorCmd objects
        for i in range(20):
            cmd_msg.motor_cmd[i].mode = 0x01  # Set torque mode, 0x00 is passive mode
            cmd_msg.motor_cmd[i].q = 0.0
            cmd_msg.motor_cmd[i].kp = 0.0
            cmd_msg.motor_cmd[i].dq = 0.0
            cmd_msg.motor_cmd[i].kd = 0.0
            cmd_msg.motor_cmd[i].tau = 0.0


class ZNormalizer:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def normalize(self, x, mean, std):
        return (x - mean) / (std + self.eps)

    def denormalize(self, y, mean, std):
        return y * (std + self.eps) + mean

class MinMaxNormalizer: #-1 ~ 1
    def __init__(self, feature_dim):
        self.min_val = np.inf * np.ones(feature_dim)
        self.max_val = -np.inf * np.ones(feature_dim)
    
    def update(self, new_data):
        self.min_val = np.minimum(self.min_val, new_data)
        self.max_val = np.maximum(self.max_val, new_data)

    def normalize(self, x, xmin, xmax):
        return 2 * ((x - xmin) / (xmax - xmin)) - 1



def main(args=None):
    rclpy.init(args=args)                          # Initialize rclpy
    node = RlSubPub()                            # Create an instance of the TestSubPub class
    rclpy.spin(node)                               # Keeps the node running, processing incoming messages
    rclpy.shutdown()
   

if __name__ == '__main__':
    main()


# def orbit_to_ros2(self, obit):
#     ros2 = [obit[1], obit[5], obit[9], 
#             obit[0], obit[4], obit[8], 
#             obit[3], obit[7], obit[11], 
#             obit[2], obit[6], obit[10]]
#     return ros2

# orbit (for actuator and sensor)
# FL_hip 0 -> 3
# FR_hip 1 -> 0
# RL_hip 2 -> 9
# RR_hip 3 -> 6

# FL_thigh 4 -> 4
# FR_thigh 5 -> 1
# RL_thigh 6 -> 10    
# RR_thigh 7 -> 7

# FL_calf 8  -> 5
# FR_calf 9  -> 2
# RL_calf 10 -> 11
# RR_calf 11 -> 8
  
#　　　　🔽　from orbit

# #orbit -> ros2      
# ros2_action = [action[1], action[5], action[9], 
#                action[0], action[4], action[8], 
#                action[3], action[7], action[11], 
#                action[2], action[6], action[10]]


#　　　　🔽　to ros2

# ros2 (for actuator and sensor)
# FR_hip   0 <- 1
# FR_thig  1 <- 5
# FR_calf  2 <- 9

# FL_hip   3 <- 0
# FL_thig  4 <- 4
# FL_calf  5 <- 8

# RR_hip   6 <- 3
# RR_thig  7 <- 7
# RR_calf  8 <- 11

# RL_hip   9  <- 2
# RL_thig  10 <- 6
# RL_calf  11 <- 10


# orbit, default
# self._offset = np.array([0.0267, -0.0267,  0.0775, -0.0775,  
#                          0.5630,  0.5631,  0.5982,  0.5982,
#                         -1.3400, -1.3400, -1.3732, -1.3731])

# ros2
# self._ros2_offset = np.array([ -0.0267, 0.5631, -1.3400, 0.0267,
#                                 0.5630, -1.3400, -0.0775, 0.5982,
#                                -1.3731, 0.0775, 0.5982, -1.3732])

# #orbit
# target_joints = 
# #orbit -> ros2
# target_joints_ros2 = np.array([target_joints[1], target_joints[5], target_joints[9], target_joints[0],
#                                target_joints[4], target_joints[8], target_joints[3], target_joints[7],
#                                target_joints[11], target_joints[2], target_joints[6], target_joints[10])




# def ros2_to_orbit(self, ros2):
#     orbit = [ros2[3], ros2[0], ros2[9], ros2[6],
#              ros2[4], ros2[1], ros2[10], ros2[7],
#              ros2[5], ros2[2], ros2[11], ros2[8]]
#     return orbit

# ros2 (for actuator and sensor)
# FR_hip   0 -> 1
# FR_thig  1 -> 5
# FR_calf  2 -> 9

# FL_hip   3 -> 0
# FL_thig  4 -> 4
# FL_calf  5 -> 8

# RR_hip   6 -> 3
# RR_thig  7 -> 7
# RR_calf  8 -> 11

# RL_hip   9  -> 2
# RL_thig  10 -> 6
# RL_calf  11 -> 10

#      🔽　to orbit

# orbit (for actuator and sensor)
# FL_hip 0 <- 3
# FR_hip 1 <- 0
# RL_hip 2 <- 9
# RR_hip 3 <- 6 

# FL_thigh 4 <- 4 
# FR_thigh 5 <- 1
# RL_thigh 6 <- 10   
# RR_thigh 7 <- 7

# FL_calf 8  <- 5
# FR_calf 9  <- 2 
# RL_calf 10 <- 11
# RR_calf 11 <- 8



# #ros2 (joint_pos_list, joint_vel_list) -> obit        
#             joint_pos_list_orbit = [joint_pos_list[3], joint_pos_list[0], joint_pos_list[9], joint_pos_list[6],
#                                    joint_pos_list[4], joint_pos_list[1], joint_pos_list[10], joint_pos_list[7],
#                                    joint_pos_list[5], joint_pos_list[2], joint_pos_list[11], joint_pos_list[8]]

#             joint_vel_list_orbit = [joint_vel_list[3], joint_vel_list[0], joint_vel_list[9], joint_vel_list[6],
#                                    joint_vel_list[4], joint_vel_list[1], joint_vel_list[10], joint_vel_list[7],
#                                    joint_vel_list[5], joint_vel_list[2], joint_vel_list[11], joint_vel_list[8]]

#standup position for ros2
# self._ros2_offset = np.array([ 0.5651, -1.3358,  0.0734, -0.0247,  
#                                0.5651, -1.3755,  0.0247,  0.5918, 
#                               -1.3756, -1.3358, -0.0734,  0.5918])
#stadnd up position for orbit
#ros2 -> orbit
# self._obit_standup_pos = np.array([ -0.0247, 0.5651, -1.3358, 0.0247,
#                                     0.5651, -1.3358, -0.0734, 0.5918,                   
#                                     -1.3755, 0.0734,  0.5918, -1.3756])