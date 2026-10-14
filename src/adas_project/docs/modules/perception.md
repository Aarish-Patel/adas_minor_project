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
| Relocalisation | no pose guess (car lifted and dropped, restart in a saved map): global coarse-to-fine matching against every submap, refuses instead of guessing when the place is ambiguous, confirmed over two scans; saved / loaded maps |
| Latency | online estimate of scan delay (range-flow speed vs model speed) -> excess over the 0.2 s budget |
| Steering calibration (online) | RLS on scan-matched heading change per distance -> servo centre and gain used by predictions |
| Tracking | cluster (with real / cut ends) -> nearest-neighbour association (0.15 m gate) -> weighted line fits over the newest 3..8 samples -> chi-square test against the car's motion field with the ego error estimated from the other tracks; `moving` after 2 scans above the threshold; pieces of walls never moving |
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
| `RelaySpeed` (`pi/relay_assists.py`) | `command(t, dt, physical, servo)` after every command sent; `on_scan(points, seq, t)` per new scan; properties `v`, `w`, `sigma_v`, `pose`, `pose_corrected`, `lost`; `v_gate(direction)` conservative speed for braking; `latency` (DelayEstimator); `enable_slam()`, `reset_pose()`, `relocalise()`, `save_map(path)`, `load_map(path)` |
| `ObstacleMemory` (`adas/memory.py`) | advance with ego-motion, add scan points, `blind_points()`, `correct_pose(pose, stamp)` (re-anchored on the scan-matched pose after each scan), `clear()` |
| `Tracker` (`adas/tracking.py`) | `update(points, t, v, omega)` -> tracks with position, velocity, `moving`, `evidence` (chi-square statistic); `model="lrt"` (default) or `"legacy"`; feed it the EKF speed and yaw rate (`RelaySpeed.v`, `.w`) |
| `SpeedEKF` (`adas/speed_ekf.py`) | `command`, `predict`, `correct(t, meas)`; state `v`, `w`, covariance `P` |
| `RangeFlow` (`adas/rf2o.py`), `LocalMapOdometry` / `Odometry` / `icp` (`pi/scanmatch.py`) | pose: scan-to-local-map point-to-line ICP with degeneracy handling (ADR-034; the relay's manoeuvre frame and world pose); scan-to-scan `Odometry` kept for the bypass tools |
| `PoseGraphMap`, `SlamService` (`adas/submaps.py`) | `add_scan(pose_est, pts)` / `submit(pose_est, pts, t)`; `correct(pose)` / `corrected(pose)`; `relocalise(pts) -> Fix`, `rebase(fix, pose_est)`, `save(path)` / `load(path)`; service: `relocalise()`, `lost`, `last_fix`, `relocalisations`, `save`, `load`; `Relocaliser`, `kidnap_score` |
| `DelayEstimator` (`adas/latency.py`) | `push_model(t, v)`, `push_flow(t, v)`, `delay`, `excess` |
| `OnlineSteering` (`adas/online_steering.py`) | `update(t, v, pose)` -> centre, k |
| Vision (`adas/vision/*`) | `RearOdometry`, `RearObjects`, `LoomingTracker`, `ImageQuality`, `label_clusters`, `draw_reverse_assist`, `first_contact`, `floor_patches` |

Frames: vehicle (rear axle, x ahead, y left); relay scans are clockwise - convert only here. SI units inside.

## Test strategy
- Unit: `tests/test_scanmatch.py`, `test_latency.py`, `test_submaps.py` (loop closure, relocalisation, kidnap detector, saved maps,
  service), `test_tracking.py` (cluster ends, window fit, static never flagged, walker latency), `test_online_steering.py`,
  `test_vision.py`, `test_reverse_assist.py`, `test_uncertainty_margin.py`.
- Twin evaluations with ground truth: `sim/odometry_eval.py` (speed/yaw vs truth and vs a real log), `sim/slam_eval.py`,
  `sim/home_eval.py`, `sim/ekf_robust_eval.py`, `sim/reloc_eval.py` (relocalisation, `models/reloc.json`), `sim/tracking_eval.py`
  (moving test latency and false flags, tune / confirm seeds, `models/tracking_latency.json`).
- Timing: `tools/profile_perception.py` (per scan, laptop and Pi estimate, `models/perception_profile.json`).
- Real data: `tools/replay_log.py` on recorded drives; the kidnap detector's threshold was checked on ~10 000 consecutive real-car scan
  pairs (lowest agreement 0.65, threshold 0.30).

## Performance targets
| Metric | Target | Current |
|---|---|---|
| Scan processing (chain per scan, Pi 5 estimate = laptop × 3.5, `tools/profile_perception.py`, 187 scans of 719 points) | control-loop pieces (convert, memory, RF2O + EKF, ICP) ≤ 20 ms; everything within one scan period (100 ms) | control-loop share **15 ms median / 22 ms p95 / 33 ms max** on scan ticks (laptop 4.3 / 6.4 / 9.5); the LiDAR-thread tracker 3.3 ms; the SLAM worker 0.1 ms on 85 % of scans and **98 ms median / 243 ms max** on the 15 % with a loop search (it drops scans while busy, never blocks the loop): met with little margin, not yet measured on the Pi |
| Speed RMSE | ≤ 5 cm/s | 3.4 cm/s real log, 4.2 twin |
| Yaw-rate RMSE | ≤ 3 °/s | 2.1 real log |
| Pose drift | ≤ 2 % of distance | 1.6-1.8 % |
| Loop closure | end error -80 % on a drifting loop | 8.0 -> 0.8 cm (search 715 -> 28 ms on the laptop since the matcher is vectorised, same result) |
| Relocalisation (no pose guess, `sim/reloc_eval.py`, 20 random poses per room, seeds fixed) | correct pose within 3 cm / 1.5° when the place is unique; never a wrong fix; refuses when ambiguous | lounge, room, doorway: 60/60 fixes, median 0.3-0.5 cm / 0.1°, **0 wrong**; empty rectangle: 0/20 (symmetric: refused); corridor: 2/20 (ambiguous along its length), 0 wrong; 0.07-0.27 s per fix laptop (0.25-0.95 s Pi est.); with a saved map reloaded and the service: 0.3 cm after 2 scans |
| Kidnap detector | lift-and-drop noticed, no false alarm while driving | 92 % of random drops in the lounge / room / open twin rooms (0 % in a corridor); 0 alarms in ~10 000 real-car scan pairs |
| Static objects falsely "moving" (twin room with walls at grazing angles, 20 mm range noise, `sim/tracking_eval.py`, untouched seeds 3-8) | 0 | **new test (`lrt`): 0 of 13 000 static track-scans while a person crosses, 1 track (0.03 %) in 15 000 in static drives; old test: 15-19 % (700-860 of 4 700 per condition, 170-200 distinct static tracks) under the consumers' ≥ 5-scan rule**; at 8 mm 0, at 30 mm 0.1 % (1-9 track-scans of ~10 000 per condition) |
| Moving object recognised (person, 0.10 m radius, walking after standing still; median s from the first step) | as early as noise allows | `Track.moving` (new test): 0.3 m/s 1.3-1.5, 0.5 m/s 0.8-0.95, 1.0 m/s 0.45-0.6 (car still, straight, arc; 12/12 each). Old test under the same ≥ 5-scan rule: 0.9-0.95 / 0.7-0.8 / 0.6, but see the false rate above; no threshold/persistence setting of the old test reaches zero false flags (`models/tracking_legacy_grid.txt`). Not faster for slow walkers: the speed is below the noise for the first ~0.7 s |
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
| SLAM | submap 14 nodes, node every 0.20 m / 12°, loop search ±0.45 m / ±14°, score ≥ 0.62 with margin over runner-up, soft-L1 optimiser; search vectorised (integer cell shifts over a zero-padded float32 field) |
| Relocalisation (`PoseGraphMap.reloc`) | coarse level 0.4 m cells / 6° over the whole submap and all headings (60 scan points ≤ 5 m, top 24 kept) -> 0.2 m / 3° around them (top 4 per submap, 6 refined) -> 5 cm / 1° window ±0.15 m / ±4° + ICP; accept: score ≥ 0.60, ≥ 55 % of points within 10 cm of the map, margin ≥ 0.06 to a pose > 0.30 m or > 8° away; `Relocaliser`: 2 agreeing scans (0.15 m / 6°) within 3 s; lost mode gives up (new session) after 8 s |
| Kidnap detector (`kidnap_score`) | range-image agreement of consecutive scans after the best turn of ±25°: bins within 0.15 m; triggers below 0.30 (real drives ≥ 0.65, random drops in a room median 0.10) |
| Saved map | `.npz`: node poses, scans, edges, config; `SlamService.save/load`, `RelaySpeed.save_map/load_map` |
| Tracking (`Tracker`, `model="lrt"`) | gate 0.15 m; cluster gap 0.08 m, ≥ 3 points, ≤ 0.45 m; end = real edge if the neighbouring beam is ≥ 0.25 m further; position σ = √(20 mm²/n + 15 mm²), along the axis +0.7 per cut end (no information), +0.06 leaks across through the axis error; ego prior σ_v = 0.02 + 0.08\|v\|, σ_w = 0.005 + 0.15\|v\| (rad/s), refined from the other tracks; windows 3..8 samples (≥ 0.15 s); flag at χ² ≥ 18.4 (p ≈ 1e-4) twice in a row, keep at ≥ 9.2; residual speed ≥ 0.10 m/s; pieces cut at both ends never flagged |
| Car behaviour | vibrates strongly (inconsistent scans); stops almost instantly when the throttle is cut |
