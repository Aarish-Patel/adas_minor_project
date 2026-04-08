"""Launch file to spawn ros2_control controllers."""

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():

    controller_manager_name = '/controller_manager'

    # Spawn the joint state broadcaster first
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', controller_manager_name,
        ],
        output='screen',
    )

    # Spawn the Ackermann steering controller (after the broadcaster is up)
    ackermann_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'ackermann_steering_controller',
            '--controller-manager', controller_manager_name,
        ],
        output='screen',
    )

    # Delay Ackermann spawner to let joint_state_broadcaster settle
    delayed_ackermann = TimerAction(
        period=3.0,
        actions=[ackermann_controller_spawner],
    )

    return LaunchDescription([
        joint_state_broadcaster_spawner,
        delayed_ackermann,
    ])
