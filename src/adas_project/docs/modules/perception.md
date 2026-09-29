# Perception

## Purpose
Turn raw LiDAR scans (and optionally rear-camera frames) into what every other subsystem needs: obstacle points in the vehicle
frame, free distance along paths, the car's own motion and pose, and tracked moving objects. SRS: FR-001-008, NFR-01-03, NFR-11-12.

## Responsibilities
| Function | What it does |
|---|---|
| Scan conversion | relay scan (angle ° clockwise from ahead, range m) -> vehicle-frame points; mount offset (LiDAR 0.12 m ahead of the rear axle) and yaw offset from the tuning file; range limits |
| Blind-zone memory | remembers obstacles closer than the LiDAR's minimum range, moves them with ego-motion, prunes by distance travelled / age / count |
| Free space | free distance along the predicted swept body path (width + margin) for a curvature, forward or reverse (helpers in `adas/geometry.py`) |
| Speed / yaw | EKF: throttle-model prediction + LiDAR range-flow (RF2O-style) measurement; Mahalanobis gate; covariance -> speed uncertainty |
| Pose | keyframed ICP scan matching (keyframe every 1 m / 30°), dead reckoning between scans |
| Loop closure | submaps + correlative matching + pose-graph optimisation in a worker thread -> corrected world pose |
| Latency | online estimate of scan delay (range-flow speed vs model speed) -> excess over the 0.2 s budget |
| Steering calibration (online) | RLS on scan-matched heading change per distance -> servo centre and gain used by predictions |
| Tracking | cluster -> nearest-neighbour association -> velocity; "moving" only after ≥ 5 consistent scans at 0.15-2 m/s |
| Rear vision (optional) | ground-plane visual odometry, parallax blobs + looming time-to-contact, image quality / vibration, LiDAR-camera late fusion, guidelines, ghost footprint, floor patches |

## Inputs
| Input | Rate | Format |
|---|---|---|
| LiDAR scan | 5-10 Hz | list of (angle °, range m), sequence number, time |
| Commands actually sent | per packet | throttle PWM (+ forward), servo ° |
| Tuning | start | mount offsets, speed model, servo centre |
| Camera frame (optional) | 15 FPS | 320×240 BGR |

## Outputs (data objects, see `docs/architecture.md` § 3)
ObstacleSet (vehicle frame points incl. blind-zone points) · FreeSpace (per curvature / direction) · EgoState (v, w, sigma_v, pose,
pose_corrected) · LatencyEstimate (delay, excess) · TrackList · steering calibration (centre, k) · CameraState (optional).

## Dependencies
NumPy, SciPy (KD-tree, least squares, distance transform); OpenCV only for vision. No hardware, simulator or GUI imports.
Consumers: safety_decision (all outputs), planning (ObstacleSet, pose), integration (logging, HMI stream).

## Public interfaces
| Interface | Contract |
|---|---|
| `RelaySpeed` (`pi/relay_assists.py`) | `command(t, dt, physical, servo)` after every command sent; `on_scan(points, seq, t)` per new scan; properties `v`, `w`, `sigma_v`, `pose`, `pose_corrected`; `v_gate(direction)` conservative speed for braking; `latency` (DelayEstimator); `enable_slam()`, `reset_pose()` |
| `ObstacleMemory` (`adas/memory.py`) | advance with ego-motion, add scan points, `blind_points()` |
| `Tracker` (`adas/tracking.py`) | `update(points, t, v, omega)` -> tracks with position, velocity, moving flag |
| `SpeedEKF` (`adas/speed_ekf.py`) | `command`, `predict`, `correct(t, meas)`; state `v`, `w`, covariance `P` |
| `RangeFlow` (`adas/rf2o.py`), `Odometry` / `icp` (`pi/scanmatch.py`) | scan-to-scan motion |
| `PoseGraphMap`, `SlamService` (`adas/submaps.py`) | `add_scan(pose_est, pts)` / `submit(...)`; `correct(pose)` / `corrected(pose)` |
| `DelayEstimator` (`adas/latency.py`) | `push_model(t, v)`, `push_flow(t, v)`, `delay`, `excess` |
| `OnlineSteering` (`adas/online_steering.py`) | `update(t, v, pose)` -> centre, k |
| Vision (`adas/vision/*`) | `RearOdometry`, `RearObjects`, `LoomingTracker`, `ImageQuality`, `label_clusters`, `draw_reverse_assist`, `first_contact`, `floor_patches` |

Frames: vehicle (rear axle, x ahead, y left); relay scans are clockwise - convert only here. SI units inside.

## Test strategy
- Unit: `tests/test_scanmatch.py`, `test_latency.py`, `test_submaps.py`, `test_online_steering.py`, `test_vision.py`,
  `test_reverse_assist.py`, `test_uncertainty_margin.py`, tracking cases in `test_crossing.py`.
- Twin evaluations with ground truth: `sim/odometry_eval.py` (speed/yaw vs truth and vs a real log), `sim/slam_eval.py`,
  `sim/home_eval.py`, `sim/ekf_robust_eval.py`.
- Real data: `tools/replay_log.py` on recorded drives.

## Performance targets
| Metric | Target | Current |
|---|---|---|
| Scan processing | within one scan period | met |
| Speed RMSE | ≤ 5 cm/s | 3.4 cm/s real log, 4.2 twin |
| Yaw-rate RMSE | ≤ 3 °/s | 2.1 real log |
| Pose drift | ≤ 2 % of distance | 1.6-1.8 % |
| Loop closure | end error -80 % on a drifting loop | 8.0 -> 0.8 cm |
| Static objects falsely "moving" | 0 under 20 mm noise | 0 |
| Rear vision pipeline | ≥ 15 FPS on the Pi at 320×240 | 3.6 ms laptop, ~13 ms Pi estimate |

## Files
`adas/geometry.py`, `adas/memory.py`, `adas/tracking.py`, `adas/speed_ekf.py`, `adas/rf2o.py`, `adas/latency.py`, `adas/submaps.py`,
`adas/online_steering.py`, `adas/lidar_mount.py`, `adas/lidar_utils.py`, `adas/vehicle_params.py`, `adas/markers.py`, `adas/vision/*`,
`pi/scanmatch.py`, `RelaySpeed` in `pi/relay_assists.py`, `pi/rear_camera.py` (camera service), `tools/camera_calibrate.py`,
`tools/validate_camera.py`.

## Key parameters and car facts
| Item | Value |
|---|---|
| Body (vehicle frame) | front +0.28 m, rear −0.05 m (the gate model; body length ~0.33 m), width 0.20 m, wheelbase 0.20 m |
| LiDAR | RPLIDAR A3, ~10 Hz on the car (health: < 6 Hz = limp, no scan 1 s = fault), 0.12 m ahead of the rear axle; yaw offset from the tuning file (45.7° since 28 Sep after the ESP32 swap - the mount turned) |
| Steering model | curvature = −0.0656 1/m per servo degree from centre (fitted); centre ~87° (online estimate may adjust) |
| Speed model | v_max 0.896 m/s, dead-band 48.5 PWM (tuning); twin fitted v_max 0.83, dead-band 11 PWM |
| Blind-zone memory | blind radius 0.27 m, keep radius 1.2 m, max age 60 s, ≤ 700 points, ≤ 1.5 m travel, 2 cm voxel |
| Scan matching | keyframes every 1 m / 30°; ICP residual < 5 cm and ≥ 45 inliers else prediction; corridor ambiguity -> predicted along-track distance |
| SLAM | submap 14 nodes, node every 0.20 m / 12°, loop search ±0.45 m / ±14°, score ≥ 0.62 with margin over runner-up, soft-L1 optimiser |
| Car behaviour | vibrates strongly (inconsistent scans); stops almost instantly when the throttle is cut |
