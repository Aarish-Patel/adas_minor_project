"""Launch the robot_localization EKF node for sensor fusion."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_localization = get_package_share_directory('ev_robot_localization')
    ekf_config = os.path.join(pkg_localization, 'config', 'ekf.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock',
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                ekf_config,
                {'use_sim_time': use_sim_time},
            ],
            remappings=[
                ('odometry/filtered', '/odom'),
            ],
        ),
    ])
