#!/usr/bin/env python3
"""
inference_node.py — Real-time Intent Prediction Engine.

This node hosts a Recursive Neural Network (specifically a GRU) that processes 
temporal windows of vehicle kinematics to predict the intent of the driver 
or surrounding actors.

Deep Learning Architecture:
- 2-Layer Gated Recurrent Unit (GRU).
- Input: 7 Features (Yaw, Accel, Steering, Velocity Gradient, Lane Dev, Effort, Range).
- Output: 4 categorical classes (Aggressive, LaneChange, EmergencyBrake, Normal).
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float32MultiArray, Float32
from gazebo_msgs.msg import ModelStates
import collections, time, os, math
import torch
import torch.nn as nn
import numpy as np

TRACK_CENTER = 48.5
NUM_FEATURES = 7
WINDOW_SIZE  = 10
CLASS_NAMES  = ['aggressive', 'lane_change', 'emergency_brake', 'normal']


class IntentModel(nn.Module):
    """
    PyTorch implementation of the Intent Prediction GRU.
    
    The model takes a window of temporal features and learns to map motion 
    patterns to specific driving intents.
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        # GRU Core: Handles time-series sequential dependencies
        self.gru  = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected Layers: Maps features to class scores
        self.fc1  = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(32, num_classes)

    def forward(self, x):
        """Standard Forward Pass through the RNN."""
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Recurrent pass
        out, _ = self.gru(x, h0)
        # We only care about the last output of the sequence (Final hidden state)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        return self.fc2(out)


class InferenceNode(Node):
    """
    ROS2 Node that converts sensor streams into intent probabilities.
    
    This node maintains a sliding window of features and executes an 
    asynchronous inference loop at 10Hz.
    """
    def __init__(self):
        super().__init__('inference_node')

        # ─── Initialization ──────────────────────────────────────────
        # Gather all kinematic and perceptual features
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Output channels for internal ADAS consumption
        self.prob_pub   = self.create_publisher(Float32MultiArray, '/intent_probabilities', 10)
        self.intent_pub = self.create_publisher(Float32, '/intent_probability', 10)

        # ─── Raw Input Buffers ───────────────────────────────────────
        self.yaw_rate     = 0.0
        self.accel_x      = 0.0
        self.steer_pos    = 0.0
        self.steer_rate   = 0.0
        self.joint_effort = 0.0
        self.velocity     = 0.0
        self.ego_x        = 0.0
        self.ego_y        = 0.0
        self.lane_deviation = 0.0
        self.min_distance = 100.0

        # Feature engineering state
        self.last_velocity = 0.0
        self.last_time = time.time()
        # Sliding window buffer (fixed size sequence for GRU)
        self.feature_window = collections.deque(maxlen=WINDOW_SIZE)

        # ─── Model Loading ───────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = IntentModel().to(self.device)

        model_path = 'intent_model.pt'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.get_logger().info(f'PyTorch: Loaded Weights from {model_path}')
        else:
            self.get_logger().error('PyTorch: Weights NOT FOUND. Node will output untuned priors.')

        self.model.eval()
        # Inference Loop (10Hz)
        self.timer = self.create_timer(0.1, self.run_inference)

    # ── callbacks ─────────────────────────────────────────────────────

    def imu_cb(self, msg):
        """Gathers lateral and longitudinal acceleration/rotation."""
        self.yaw_rate = msg.angular_velocity.z
        self.accel_x  = msg.linear_acceleration.x

    def joint_cb(self, msg):
        """Extracts steering kinematics and motor load."""
        if 'front_left_steer_joint' in msg.name:
            idx = msg.name.index('front_left_steer_joint')
            old = self.steer_pos
            self.steer_pos = msg.position[idx]
            self.steer_rate = (self.steer_pos - old) / 0.1
        if 'rear_left_wheel_joint' in msg.name:
            idx = msg.name.index('rear_left_wheel_joint')
            if len(msg.effort) > idx:
                self.joint_effort = msg.effort[idx]

    def scan_cb(self, msg):
        """Processes LiDAR for obstacle proximity feature."""
        mid = len(msg.ranges) // 2
        vals = [r for r in msg.ranges[mid-15:mid+15]
                if 0.1 < r < 200.0 and not math.isnan(r)]
        self.min_distance = min(vals) if vals else 100.0

    def model_cb(self, msg):
        """Computes track-absolute lane deviation."""
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            self.ego_x = msg.pose[idx].position.x
            self.ego_y = msg.pose[idx].position.y
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            self.velocity = math.sqrt(vx*vx + vy*vy)

            # Circular track geometry math
            cx, cy = self.ego_x, self.ego_y
            if cy > 0 and -100 <= cx <= 100:
                self.lane_deviation = cy - TRACK_CENTER
            elif cy <= 0 and -100 <= cx <= 100:
                self.lane_deviation = -(cy + TRACK_CENTER)
            elif cx > 100:
                self.lane_deviation = math.sqrt((cx-100)**2 + cy**2) - TRACK_CENTER
            else:
                self.lane_deviation = math.sqrt((cx+100)**2 + cy**2) - TRACK_CENTER

    # ── inference ─────────────────────────────────────────────────────

    def run_inference(self):
        """Main Compute Task: Features -> Tensor -> Model -> Probability."""
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: return
        
        # Calculate velocity derivative (Acceleration)
        vel_deriv = (self.velocity - self.last_velocity) / dt
        self.last_velocity = self.velocity
        self.last_time = now

        # Append current feature vector to temporal window
        self.feature_window.append([
            self.yaw_rate, self.accel_x, self.steer_rate,
            vel_deriv, self.lane_deviation, self.joint_effort,
            self.min_distance,
        ])

        # Execute if buffer is full (Needs WINDOW_SIZE samples for the GRU)
        if len(self.feature_window) == WINDOW_SIZE:
            arr = np.array(self.feature_window, dtype=np.float32)
            tensor = torch.tensor(arr).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 1. Full probability distribution
            prob_msg = Float32MultiArray()
            prob_msg.data = probs.tolist()
            self.prob_pub.publish(prob_msg)

            # 2. Aggressive Intent Scalar (Class 0)
            agg_msg = Float32()
            agg_msg.data = float(probs[0])
            self.intent_pub.publish(agg_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
