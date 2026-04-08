from setuptools import setup
import os
from glob import glob

package_name = 'robot_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hsiraa',
    maintainer_email='aarishpatel30@gmail.com',
    description='Top-level bringup and demo nodes for the EV Robot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'autonomous_demo = robot_bringup.autonomous_demo:main',
            'cmd_vel_relay = robot_bringup.cmd_vel_relay:main',
        ],
    },
)
