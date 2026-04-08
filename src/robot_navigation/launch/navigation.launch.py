"""Launch Nav2 navigation stack for the EV Robot."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_navigation = get_package_share_directory('robot_navigation')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    nav2_params = os.path.join(pkg_navigation, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_navigation, 'rviz', 'nav.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2',
        ),

        # --- Nav2 Bringup ---
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_nav2_bringup, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': nav2_params,
            }.items(),
        ),

        # --- RViz2 ---
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(use_rviz),
        ),
    ])
