import os
import yaml

param_file_path = '/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml'
output_path = os.path.expanduser('~/.gemini/antigravity/scratch/ev_robot_ws/src/robot_navigation/config/nav2_params.yaml')

with open(param_file_path, 'r') as f:
    params = yaml.safe_load(f)

# Add range sensor layer to local costmap
local_plugins = params['local_costmap']['local_costmap']['ros__parameters']['plugins']
if 'range_sensor_layer' not in local_plugins:
    local_plugins.insert(0, 'range_sensor_layer')

params['local_costmap']['local_costmap']['ros__parameters']['range_sensor_layer'] = {
    'plugin': 'nav2_costmap_2d::RangeSensorLayer',
    'topics': [
        '/ultrasonic_front_left',
        '/ultrasonic_front_center',
        '/ultrasonic_front_right',
        '/ultrasonic_rear_left',
        '/ultrasonic_rear_right'
    ],
    'phi': 0.15,
    'clear_threshold': 0.2,
    'mark_threshold': 0.8,
    'clear_on_max_reading': True
}

# Add range sensor layer to global costmap
global_plugins = params['global_costmap']['global_costmap']['ros__parameters']['plugins']
if 'range_sensor_layer' not in global_plugins:
    global_plugins.insert(0, 'range_sensor_layer')

params['global_costmap']['global_costmap']['ros__parameters']['range_sensor_layer'] = {
    'plugin': 'nav2_costmap_2d::RangeSensorLayer',
    'topics': [
        '/ultrasonic_front_left',
        '/ultrasonic_front_center',
        '/ultrasonic_front_right',
        '/ultrasonic_rear_left',
        '/ultrasonic_rear_right'
    ],
    'phi': 0.15,
    'clear_threshold': 0.2,
    'mark_threshold': 0.8,
    'clear_on_max_reading': True
}

# Change footprint for the EV robot (30x12cm)
# Let's say it's bounded by a box [-0.05, -0.075] to [0.25, 0.075]
robot_footprint = '[ [0.25, 0.075], [0.25, -0.075], [-0.05, -0.075], [-0.05, 0.075] ]'

params['local_costmap']['local_costmap']['ros__parameters']['footprint'] = robot_footprint
params['global_costmap']['global_costmap']['ros__parameters']['footprint'] = robot_footprint

# Remove robot_radius if present
params['local_costmap']['local_costmap']['ros__parameters'].pop('robot_radius', None)
params['global_costmap']['global_costmap']['ros__parameters'].pop('robot_radius', None)

# Update AMCL params
params['amcl']['ros__parameters']['base_frame_id'] = 'base_footprint'
# No laser scanner mapping for AMCL, but we keep it default or disable.
# Actually we might just run slam_toolbox because we don't have a map.
# Or if we provide a map, AMCL defaults to scan. We only have camera. So odometry and ultrasonics.

with open(output_path, 'w') as f:
    yaml.dump(params, f, default_flow_style=False)

print("Generated modified nav2_params.yaml")
