import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro
import tempfile

def generate_launch_description():
    pkg_share = get_package_share_directory('adas_project')
    world_path = os.path.join(pkg_share, 'worlds', 'two_lane.world')
    xacro_file = os.path.join(pkg_share, 'urdf', 'vehicle.xacro')

    doc = xacro.process_file(xacro_file)
    robot_desc = doc.toxml()

    urdf_path = os.path.join(tempfile.gettempdir(), 'adas_vehicle.urdf')
    with open(urdf_path, 'w') as f:
        f.write(robot_desc)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'adas_vehicle', '-file', urdf_path,
                   '-x', '-50', '-y', '46.5', '-z', '0.5'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
