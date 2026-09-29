"""Vision algorithms for the (rear-facing) camera - all tested on synthetic frames (adas/vision/sim_camera.py):
rear_odometry (speed / yaw from the floor), rear_objects (ground-compensated residual blobs + looming time to
contact), quality (blur / brightness / vibration monitor), fusion (LiDAR clusters labelled by detections),
guidelines (reverse guidelines from the steering), detector (optional YOLO), plus the existing adas.markers (ArUco
pose for docking / parking) and adas.lane."""
