"""
robot.launch.py — Master launch file for the REAL ROBOT.

Brings up the same navigation and localization stack but with
real hardware driver nodes instead of Gazebo simulation.

Hardware drivers are included as placeholder nodes — replace
the package/executable names with your actual driver packages
once the hardware is integrated.
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    pkg_description = get_package_share_directory('robot_description')
    pkg_localization = get_package_share_directory('ev_robot_localization')
    pkg_navigation = get_package_share_directory('robot_navigation')

    xacro_file = os.path.join(pkg_description, 'urdf', 'robot.urdf.xacro')
    robot_description_content = Command(['xacro ', xacro_file])

    return LaunchDescription([
        # --- Arguments ---
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),

        # ==================================================
        # Robot State Publisher (loads URDF for TF tree)
        # ==================================================
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_content,
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        # ==================================================
        # HARDWARE DRIVERS — REPLACE WITH ACTUAL PACKAGES
        # ==================================================

        # --- USB Camera (Lenovo webcam via usb_cam) ---
        # Install: sudo apt install ros-humble-usb-cam
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device': '/dev/video0',
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv',
                'framerate': 30.0,
                'camera_name': 'camera',
            }],
            remappings=[
                ('image_raw', '/camera/image_raw'),
            ],
            output='screen',
        ),

        # --- IMU (BNO055 via serial or I2C) ---
        # PLACEHOLDER: Replace with your BNO055 ROS2 driver node
        # e.g.  package='bno055', executable='bno055_node'
        # The node should publish sensor_msgs/Imu on /imu
        # Node(
        #     package='bno055',
        #     executable='bno055_node',
        #     name='imu_driver',
        #     parameters=[{
        #         'serial_port': '/dev/ttyUSB0',
        #         'frame_id': 'imu_link',
        #     }],
        #     remappings=[('imu/data', '/imu')],
        #     output='screen',
        # ),

        # --- ESP32 Ultrasonic Sensor Bridge ---
        # PLACEHOLDER: Replace with your micro_ros_agent or
        # custom serial bridge node that publishes Range msgs
        # on /ultrasonic_front_left, etc.
        # Node(
        #     package='micro_ros_agent',
        #     executable='micro_ros_agent',
        #     name='micro_ros_agent_ultrasonics',
        #     arguments=['serial', '--dev', '/dev/ttyUSB1', '-b', '115200'],
        #     output='screen',
        # ),

        # --- ESP32 Wheel Encoder Bridge ---
        # PLACEHOLDER: Replace with your encoder node that
        # publishes nav_msgs/Odometry on /wheel_odom
        # Node(
        #     package='micro_ros_agent',
        #     executable='micro_ros_agent',
        #     name='micro_ros_agent_encoders',
        #     arguments=['serial', '--dev', '/dev/ttyUSB2', '-b', '115200'],
        #     output='screen',
        # ),

        # ==================================================
        # EKF Localization
        # ==================================================
        TimerAction(
            period=3.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(pkg_localization, 'launch', 'localization.launch.py')
                    ),
                    launch_arguments={'use_sim_time': use_sim_time}.items(),
                ),
            ],
        ),

        # ==================================================
        # Nav2 Navigation Stack
        # ==================================================
        TimerAction(
            period=6.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(pkg_navigation, 'launch', 'navigation.launch.py')
                    ),
                    launch_arguments={
                        'use_sim_time': use_sim_time,
                        'use_rviz': use_rviz,
                    }.items(),
                ),
            ],
        ),
    ])
