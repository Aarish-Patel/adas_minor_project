"""Launch Gazebo with the test arena world and spawn the EV robot."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_simulation = get_package_share_directory('robot_simulation')
    pkg_description = get_package_share_directory('robot_description')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_simulation, 'worlds', 'test_arena.world')
    xacro_file = os.path.join(pkg_description, 'urdf', 'robot.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_description_content = Command(['xacro ', xacro_file])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation clock',
        ),

        # --- Launch Gazebo server + client ---
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'world': world_file}.items(),
        ),

        # --- Robot State Publisher ---
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_content,
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        # --- Spawn robot entity into Gazebo ---
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'ev_robot',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.01',
            ],
            output='screen',
        ),
    ])
