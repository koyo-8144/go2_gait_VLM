#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState, SportModeState, LowCmd, IMUState, MotorState
from math import tanh
import time
import numpy as np
from unitree_sdk2py.utils.crc import CRC
import matplotlib.pyplot as plt

INFO_IMU = 1        # Set 1 to info IMU states
INFO_MOTOR = 1      # Set 1 to info motor states
INFO_VEL = 1        # set 1 to info vel states
INFO_FOOT_STATE = 0 # Set 1 to info foot states (foot position and velocity in body frame)

REALTIMEDIS = 0

class TopicCheck(Node):
    def __init__(self):
        super().__init__("topic_check")
        #ROS2 settings
        self.lowstate_topic = "/lowstate"
        self.sportstate_topic = "/sportmodestate"
        self.lowcmd_topic = "/lowcmd"

        # Subscribers
        self.lowstate_suber = self.create_subscription(LowState, self.lowstate_topic, self.lowstate_callback, 20)
        self.sportstate_suber = self.create_subscription(SportModeState, self.sportstate_topic, self.sportstate_callback, 20)

        # Publisher
        self.lowcmd_puber = self.create_publisher(LowCmd, self.lowcmd_topic, 10)

        #timer_period = 0.02 #callback to execute every 0.02 seconds (50 Hz)
        #timer_period = 0.1
        #timer_period = 0.25
        #timer_period = 0.5
        timer_period = 1.0
        self.timer_ = self.create_timer(timer_period, self.lowcmd_callback) #callback to execute every 0.02 seconds (50 Hz)
        #self.t = 0

        self.lowcmd_count = 0

        self.projected_gravity = np.array([-0.1, 0.0, -1.0])
        self.velocity_commands = np.array([0.9, -0.9, 0.4])

        # Flags to check if data is received
        self.lin_vel_received = False
        self.ang_vel_received = False
        self.joint_received = False

        self.crc = CRC()

        self.init_cmd()
        #self.standup()

        self.real_time_display()
        

    def lowstate_callback(self, data):
        if INFO_IMU:
            # Info IMU states
            self.imu = data.imu_state
            self.ang_vel_x = self.imu.gyroscope[0]
            self.ang_vel_y = self.imu.gyroscope[1]
            self.ang_vel_z = self.imu.gyroscope[2]
            self.ang_vel_received = True
            #self.get_logger().info(f"ang_vel_x: {self.imu.gyroscope[0]}; ang_vel_y: {self.imu.gyroscope[1]}; ang_vel_z: {self.imu.gyroscope[2]}")
     
        if INFO_MOTOR:
            motor = [None] * 12  # Initialize a list with 12 elements
            # Initialize empty lists to collect joint positions and velocities
            joint_pos_list = []
            joint_vel_list = []
            for i in range(12):
                motor[i] = data.motor_state[i]
                joint_pos = motor[i].q
                joint_vel = motor[i].dq
                # Append each joint position and velocity to the lists
                joint_pos_list.append(joint_pos)
                joint_vel_list.append(joint_vel)

                # self.get_logger().info(f"num: {i}")
                # self.get_logger().info(f"joint_pos : {joint_pos}")
                # self.get_logger().info(f"joint_vel : {joint_vel}")
            # Convert the lists into numpy arrays
            self.joint_pos_array = np.array(joint_pos_list)
            self.joint_vel_array = np.array(joint_vel_list)
            self.joint_received = True
            #print("self.joint_pos_array", self.joint_pos_array)
            #print("self.joint_vel_array", self.joint_vel_array)
            # breakpoint()

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

    def real_time_display(self):
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

    def update_subplot(self, line, new_data):
        line.set_ydata(new_data)
        # line.axes.relim()  # Recompute the data limits
        # line.axes.autoscale_view()  # Rescale the view
        plt.draw()
        plt.pause(0.1)  # Adjust pause time to control the update frequency

    def lowcmd_callback(self):
        if not (self.lin_vel_received and self.ang_vel_received and self.joint_received):
            # If data has not been received from both topics, do nothing
            return
        
        if REALTIMEDIS:
            lin_vel = np.array([self.lin_vel_x, self.lin_vel_y, self.lin_vel_z])
            ang_vel = np.array([self.ang_vel_x, self.ang_vel_y, self.ang_vel_z])

            print("upload subplots")

            self.update_subplot(self.lines1[0], lin_vel)
            self.update_subplot(self.lines1[1], ang_vel)

            self.update_subplot(self.lines2[0], self.joint_pos_array)
            self.update_subplot(self.lines2[1], self.joint_vel_array)

            self.update_subplot(self.lines3[0], self.projected_gravity)
            self.update_subplot(self.lines3[1], self.velocity_commands)
            
            time.sleep(0.01)  # Simulate time delay for real-time update

        # self.lowcmd_count += 1

        # cmd_msg = LowCmd()

        # if self.lowcmd_count % 2 == 0:
        #     cmd_msg.motor_cmd[0].q =  0.5 #float(ros_action[i])  # Target angular(rad)
        #     cmd_msg.motor_cmd[0].kp = 25.0 #1.0 # Position(rad) control kp gain
        #     cmd_msg.motor_cmd[0].dq = 0.0 # Target angular velocity(rad/s)
        #     cmd_msg.motor_cmd[0].kd = 0.5 #3.5 # Position(rad) control kd gain
        #     cmd_msg.motor_cmd[0].tau = 0.0  # target torque (N.m)
        #     cmd_msg.crc = self.crc.Crc(cmd_msg)
        #     self.lowcmd_puber.publish(cmd_msg)
        # else:
        #     cmd_msg.motor_cmd[0].q =  -0.5 #float(ros_action[i])  # Target angular(rad)
        #     cmd_msg.motor_cmd[0].kp = 25.0 #1.0 # Position(rad) control kp gain
        #     cmd_msg.motor_cmd[0].dq = 0.0 # Target angular velocity(rad/s)
        #     cmd_msg.motor_cmd[0].kd = 0.5 #3.5 # Position(rad) control kd gain
        #     cmd_msg.motor_cmd[0].tau = 0.0  # target torque (N.m)
        #     cmd_msg.crc = self.crc.Crc(cmd_msg)
        #     self.lowcmd_puber.publish(cmd_msg)

        # for i in range(12):
        #     cmd_msg.motor_cmd[i].q =  0.5 #float(ros_action[i])  # Target angular(rad)
        #     cmd_msg.motor_cmd[i].kp = 25.0 #1.0 # Position(rad) control kp gain
        #     cmd_msg.motor_cmd[i].dq = 0.0 # Target angular velocity(rad/s)
        #     cmd_msg.motor_cmd[i].kd = 0.5 #3.5 # Position(rad) control kd gain
        #     cmd_msg.motor_cmd[i].tau = 0.0  # target torque (N.m)
        #     #print("desired position when moving with DRL ", cmd_msg.motor_cmd[i].q)
        #     cmd_msg.crc = self.crc.Crc(cmd_msg)
        #     #breakpoint()    

        # self.lowcmd_puber.publish(cmd_msg)
        # self.get_logger().info("Publishing the low level command")

        self.lin_vel_received = False
        self.ang_vel_received = False
        self.joint_received = False

    def standup(self):
        cmd_msg = LowCmd()

        stand_up_joint_pos = np.array([
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
        ],
                                    dtype=float)

        stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
            1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ],
                                        dtype=float)
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
                    cmd_msg.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (
                        1 - phase) * stand_down_joint_pos[i]
                    cmd_msg.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                    cmd_msg.motor_cmd[i].dq = 0.0
                    cmd_msg.motor_cmd[i].kd = 3.5
                    cmd_msg.motor_cmd[i].tau = 0.0

                    #print("desired position when standing up ", cmd_msg.motor_cmd[i].q)
                    cmd_msg.crc = self.crc.Crc(cmd_msg)

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

def main(args=None):
    rclpy.init(args=args)                          # Initialize rclpy
    node = TopicCheck()                            # Create an instance of the TestSubPub class
    rclpy.spin(node)                               # Keeps the node running, processing incoming messages

    rclpy.shutdown()

    #if REALTIMEDIS:
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plots
   

if __name__ == '__main__':
    main()
