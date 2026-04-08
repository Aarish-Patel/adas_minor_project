"""
sim.launch.py — Master launch file for simulation.

Brings up the full autonomous EV robot stack in Gazebo:
  1. Gazebo + world + robot spawn + robot_state_publisher
  2. Static map → odom TF
  3. EKF localization
  4. Map server
  5. Nav2 navigation stack + RViz2

Motion is handled by the gazebo_ros_planar_move plugin (in the URDF)
which directly subscribes to /cmd_vel and publishes /wheel_odom.
No separate controller spawning needed.
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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    pkg_simulation = get_package_share_directory('robot_simulation')
    pkg_localization = get_package_share_directory('ev_robot_localization')
    pkg_navigation = get_package_share_directory('robot_navigation')

    map_yaml = os.path.join(pkg_navigation, 'maps', 'test_arena.yaml')

    return LaunchDescription([
        # --- Arguments ---
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),

        # ==================================================
        # 1) Gazebo + robot spawn + robot_state_publisher
        #    (planar_move plugin in URDF handles /cmd_vel)
        # ==================================================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_simulation, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # ==================================================
        # 2) Static map → odom transform
        # ==================================================
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

        # ==================================================
        # 3) EKF Localization (delayed for Gazebo)
        # ==================================================
        TimerAction(
            period=5.0,
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
        # 4) Map Server
        # ==================================================
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='nav2_map_server',
                    executable='map_server',
                    name='map_server',
                    parameters=[{
                        'yaml_filename': map_yaml,
                        'use_sim_time': True,
                    }],
                    output='screen',
                ),
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_map',
                    parameters=[{
                        'autostart': True,
                        'node_names': ['map_server'],
                        'use_sim_time': True,
                    }],
                    output='screen',
                ),
            ],
        ),

        # ==================================================
        # 5) Nav2 Navigation Stack + RViz2
        # ==================================================
        TimerAction(
            period=12.0,
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
