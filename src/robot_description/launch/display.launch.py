"""Launch file for standalone URDF visualization in RViz2."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_description = get_package_share_directory('robot_description')

    use_sim_time = LaunchConfiguration('use_sim_time')
    xacro_file = os.path.join(pkg_description, 'urdf', 'robot.urdf.xacro')

    robot_description_content = Command(['xacro ', xacro_file])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_content,
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        # Joint State Publisher GUI — lets you drag joint sliders
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen',
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_description, '..', '..', '..',
                        'src', 'robot_navigation', 'rviz', 'nav.rviz')],
            output='screen',
        ),
    ])
