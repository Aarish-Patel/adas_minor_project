import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('adas_project')
    
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'sim.launch.py')
        )
    )

    driver_node = Node(
        package='adas_project',
        executable='behavior_generator.py',
        name='aggressive_driver',
        arguments=['--mode', 'auto', '--driver', 'AGGRESSIVE'],
        output='screen'
    )

    supervisor_node = Node(
        package='adas_project',
        executable='supervisor.py',
        name='supervisor',
        output='screen'
    )
    
    scenario_node = Node(
        package='adas_project',
        executable='scenario_controller.py',
        name='scenario_controller',
        output='screen'
    )
    
    # We want Gazebo to spawn first before the driver model begins to ensure Lidar starts up
    delayed_driver = TimerAction(
        period=3.0,
        actions=[driver_node]
    )
    
    delayed_scenario = TimerAction(
        period=4.0,
        actions=[scenario_node]
    )
    
    delayed_supervisor = TimerAction(
        period=5.0, # Wait even longer for the supervisor to avoid instant resets as Gazebo settles
        actions=[supervisor_node]
    )

    return LaunchDescription([
        sim_launch,
        delayed_driver,
        delayed_scenario,
        delayed_supervisor
    ])
