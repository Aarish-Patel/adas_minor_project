# Interface Control Document - shared data types

| | |
|---|---|
| Document | ICD-ADAS-001 |
| Version | 1.0 |
| Date | 29 September 2026 |
| Status | The only source of truth for data exchanged between subsystems (`docs/modules/*`). |

A type listed here may only change together with every producer and consumer and a version bump below. Python types are given as the
realisation in the code (tuples, dataclasses, NumPy arrays, JSON dicts); a ROS 2 port maps them as in `docs/architecture.md` § 5.3.

## 0. Conventions

| Item | Rule |
|---|---|
| Vehicle frame `V` | origin rear-axle centre; x ahead, y left, z up; θ and curvature κ positive counter-clockwise (left) |
| LiDAR frame `L` | LiDAR centre, 0.12 m ahead of the rear axle on the centre line; angle in degrees **clockwise** from ahead (positive = right) |
| World frame `W` | pose of `V` at start or the last `ORIGIN` reset; corrected by loop closure |
| Manoeuvre frame `M` | pose of `V` when a plan was started |
| Camera frame `C` | rear camera, pinhole, yaw 180° (looking backwards) |
| Units | SI: m, s, rad, m/s, 1/m. Degrees only where stated. |
| Actuator units | servo ° (centre ≈ 87, + = right at the servo); throttle "physical" PWM −255..255, + = forward (the wire value is sign-flipped for the reversed motor: `M <wire>` with wire = −physical) |
| Speed shown | full-size km/h = m/s × 14 × 3.6 |
| Time | seconds, `time.time()` on the car; simulation time in the twin |
| Missing values | `None` / JSON `null`, never 0 |

---

## 1. LiDARScan
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| points | list of (angle, range) tuples | ° clockwise in `L`, m | only valid returns (range 0.2-12 m) |
| seq | int | - | increments per rotation; unchanged = no new scan |
| t | float | s | acquisition time |
Producer: LiDAR driver / `SimLidar`. Consumers: perception, logger. Rate 5-10 Hz (car ~10 Hz).
Conversion to `V`: x = 0.12 + r·cos(a), y = −r·sin(a) (a in rad, clockwise positive) - `RelayAssists.points_vehicle_frame`.

## 2. ObstaclePoints (detected obstacles)
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| points | ndarray (N, 2) float | m, `V` | current scan in `V` plus blind-zone memory points |
| blind | ndarray (K, 2) float | m, `V` | remembered points inside 0.27 m of the LiDAR |
Per scan. The system is geometric: an obstacle is a set of points, not a classified object.

## 3. Cluster (DetectedObject)
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| centre | (x, y) float | m, `V` | cluster centroid |
| radius | float | m | extent; clusters ≤ 0.45 m, ≥ 3 points, gap 0.08 m |
| label | str or None | - | from camera fusion when a camera is fitted, else None |
Per scan.

## 4. TrackedObject
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| id | int | - | stable across scans |
| hist | deque of (t, x, y), ≤ 8 | s, m, `V` at each scan | |
| radius | float | m | |
| vel_rel | (vx, vy) | m/s, `V` | as seen from the car |
| vel_obj | (vx, vy) | m/s, `V` | object's own velocity (ego-motion removed) |
| moving | bool | - | |
| moving_count | int | scans | consumers treat as moving only if ≥ 5 and 0.15 ≤ |vel_obj| ≤ 2.0 |
| missed | int | scans | dropped after 3 |
Producer: `Tracker.update`. Consumers: moving-object assist, gate closing speed, HMI. Per scan.

## 5. FreeSpace
| Field | Type | Unit | Notes |
|---|---|---|---|
| free(κ, direction) | float | m | travel distance (rear axle) until the body (+ margin) touches an obstacle along an arc; `inf` if clear within the horizon |
| arcs | 5 curvatures | 1/m | servo −20, −10, 0, +10, +20 ° from the current command (intent features) |
| back | float | m | reverse free distance on the current arc |
Computed on demand from ObstaclePoints (`PathGate.free_distance`, `adas.geometry`). Rate: per packet.

## 6. VehicleState (EgoState)
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| v | float | m/s | EKF speed, + forward |
| w | float | rad/s | yaw rate, + left |
| sigma_v | float | m/s | √P[0,0] of the EKF |
| v_model | float | m/s | throttle-model speed |
| v_gate(dir) | float | m/s | conservative speed for braking (max / min of model and EKF in the travel direction) |
| pose | (x, y, θ) | m, rad, `W` | front-end (scan matching + dead reckoning) |
| pose_corrected | (x, y, θ) | m, rad, `W` | after loop closure (= pose without SLAM) |
| latency.delay / .excess | float | s | estimated scan delay; excess over the 0.16 s budget, ≤ 0.4 s |
| steer_cal | {centre, k} | °, 1/m per ° | online steering calibration |
Producer: `RelaySpeed`. Rate: per command (prediction) and per scan (correction).

## 7. DriverCommand
| Field | Type | Unit | Notes |
|---|---|---|---|
| servo | float | ° | text line `A <servo> <servo>` (left, right channels) |
| wire | int | PWM | text line `M <wire>` |
| physical | float | PWM, + forward | derived: −wire for the reversed motor |
UDP port 4210, 50 Hz from `rc_controller.py`. Missing packets -> motor stopped by the relay; ESP32 stops on its own after 500 ms.

## 8. IntentState
| Field | Type | Unit | Notes |
|---|---|---|---|
| p_crash | float 0-1 or None | - | probability used for the takeover decision (v2, or the takeover-needed model if configured) |
| p_risk | float 0-1 or None | - | v3 risk, never below the physics floor (0.95 when the driver is inactive inside the stopping distance) |
| p_adapted | float 0-1 or None | - | online-recalibrated risk (shown once ≥ 30 outcomes seen) |
| trusted | bool | - | p_crash < threshold (0.5) and not stalled |
| attentive | bool | - | stick moved in the last second |
| stalled | bool | - | no progress: < 0.8 m in 5 s with throttle > 40 PWM; lasts 10 s |
| k_rate | float or None | 1/m per s | the attentive driver's curvature rate (predicted path) |
| reaction_m | float | m | this driver's usual reaction distance |
Producer: `RelayIntent.update` (20 Hz, 50 ms clock). Consumers: assists (hold / commit), gate (trusted FOS, predicted curvature), HMI.

## 9. RiskAssessment (PathPrediction + GateInfo)
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| pred / left / right | list of [angle °, range m] | `L` polar | predicted centre line and body edges of the swept path |
| hit | [angle, range] or None | `L` polar | predicted contact point |
| hit_m | float or None | m | distance to contact along the path |
| ttc | float or None | s | time to contact |
| state | "clear" / "limited" / "collision" | - | |
| action | str | - | "pass", "limited", "braking", "holding (…)", "stopped", "LiDAR lost" |
| free_m | float | m | free distance along the checked path |
| allowed_v | float | m/s | largest speed that still stops in the free distance |
| rss_min_m | float | m | RSS longitudinal safe distance at the current speed |
| warning_level | int 0-3 | - | 0 none, 1 advisory, 2 warning, 3 intervention |
Producer: `PathGate` + `RelayAssists`. Per packet. Consumers: throttle smoother, HMI, logger, Monte Carlo classifier.

## 10. Path (planned manoeuvre)
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| poses | ndarray (N, 4) | x, y m, θ rad, direction ±1 in `M` | forward / reverse legs; cusps where direction changes |
| goal | (x, y) and optional θ | `M` | |
| msg / state | str | - | "planning", "driving", "arrived", "path blocked - stopped", "stuck - …" |
| leg | (poses (K, 3) in `V`, direction) | | current leg for the gate |
Producer: planner service / `AutoNav` / `DrivingAssists`. On request; plan time bounded by budgets (0.2-8 s).

## 11. OccupancyGrid
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| logodds | ndarray (n, n) float32 | - | free −0.35 per ray pass, occupied +0.9 per hit, clipped ±4 |
| state | ndarray (n, n) uint8 | 0 unknown, 1 free, 2 occupied | |
| res, half | float | 0.05 m, 8 m | grid covers ±8 m around the `W` origin |
Producer: `adas/explore.OccupancyGrid` (exploration), per scan. Submaps (loop closure) keep points + a likelihood field instead
(`adas/submaps.Submap`: points in the submap frame, 0.05 m field, σ 0.08 m).

## 12. Zone
| Field | Type | Unit / frame | Notes |
|---|---|---|---|
| kind | "rect" / "circle" / "poly" | - | |
| geometry | x0, y0, x1, y1 / x, y, r / pts | m, `W` | |
| kph | float | full-size km/h | car limit = kph / 3.6 / 14 m/s |
Sent as `ZONES <json list>`; persisted by the GUI in `gui/zones.json` (not published).

## 13. ActuatorCommand
| Field | Type | Unit | Notes |
|---|---|---|---|
| servo | int | ° | clamped 35-145 |
| wire | int | PWM | `M <wire>`; physical after smoothing, gate and health limits |
Serial to the ESP32 per packet; `PING` -> `PONG` supervision.

## 14. HealthState
| Field | Type | Unit | Notes |
|---|---|---|---|
| state | "normal" / "limp" / "fault" | - | limp caps throttle at 120 PWM; fault holds the motor at 0 |
| causes | list of str | - | e.g. "LiDAR slow (4.2 Hz)", "ESP32 silent" |
| lidar_hz, link_hz, loop_p95_ms, temp_c | float | Hz, packets/s, ms, °C | |
1 Hz; a cause must be absent 2 s before recovery.

## 15. StateMessage (car -> HMI)
JSON object, 20 Hz over UDP to subscribers (and HTTP :8090). Keys:
| Key | Content |
|---|---|
| mode | "normal" / "override" |
| drive | v, w, v_model, pwm_in, pwm_out (physical PWM), servo, centre, t, steer_max_deg, steer_cal |
| gate | action, free_m, allowed speed (§ 9) |
| plan | pred, left, right, hit, hit_m, ttc, state, maneuver, line (all `L` polar) |
| nav | state, goal (`L` polar), msg |
| assist | enabled {name: bool}, info {name: text}, level, phase |
| intent | p_crash (shown risk), p_decision, trusted, attentive, reaction_m, stalled, calibrated, calib_a, calib_b |
| world | pose (corrected, `W`), zones, zone_kph, mode (eco / normal / sport), rss_min_m, loops |
| health, esp32 | § 14; ESP32 state, detail, reboots |
| _pts | decimated LiDAR points, `L` polar |

## 16. OperatorCommand (HMI -> car)
UDP text on 4210, reply "OK" or a message: `ASSIST <name|all> ON|OFF` · `MODE eco|normal|sport` · `ZONES <json>` · `ORIGIN` · `GOTO <x> <y> [heading°]`
(vehicle frame now) · `GOTO CANCEL` · `HOME` · `EXPLORE ON|OFF` · `PARK [PARALLEL|PERPENDICULAR]` · `FOLLOW_ON|OFF` · `ADAS_OVERRIDE_ON|OFF`
· `GUI_SUBSCRIBE <port>` (renew every 2 s, expires 6 s) · `PING`.

## 17. CameraState (optional)
| Field | Type | Unit | Notes |
|---|---|---|---|
| fps, pipeline_ms | float | Hz, ms | |
| odometry | {v, w, quality} | m/s, rad/s, 0-1 | visual odometry from the floor |
| objects | list of {id, foot (x, y), ttc} | m in `V`, s | looming objects behind the car |
| quality | blur, brightness, contrast, jitter_px, jitter_deg, vib_hz, degraded, why | | image / vibration health |
| contact_m, ttc_s, stop, patches | | m, s | reverse assist (ghost car) result |
HTTP :8091 `/state` 2 Hz; `/stream` MJPEG 15 FPS.

## 18. DriveLog record
Compressed JSONL on the Pi; one record per packet and per scan: time, driver command, command sent, gate info, assist info, intent,
EgoState, health; scans with sequence numbers; `event` records (goto, home, park, drive mode, faults) with named fields. Consumers:
`tools/replay_log.py`, twin identification, training.

## 19. SimulationResult
### 19.1 Scenario run (`sim.relay_scenarios.run`)
| Field | Type | Unit | Notes |
|---|---|---|---|
| trace | list of tuples | t s, x, y m, θ rad, v m/s, pwm, gated pwm, stick, servo-as-stick, gate action | world frame of the scenario |
| collided | bool | | |
| min_clear | float | m | minimum body clearance |
| x, y, th, v, t_end | float | | final state |
| infos | set of str | | assist messages seen |
| max_level | int | | highest warning level |
| latency | (delay, excess, n) | s | online latency estimator |
### 19.2 Monte Carlo drive (`sim.relay_mc.run`, one line of `models/mc_live/runs.jsonl`)
| Field | Type | Unit | Notes |
|---|---|---|---|
| seed, style, variant | int, str, str | | (seed, style) identifies the drive; variants share room, driver, lapses |
| crashed | bool | | |
| reached | float or None | s | time to goal (within 0.35 m) |
| min_clear | float | m | |
| interventions, false_positives (needless), episodes, progress | int | | episode = interventions < 2.5 s apart |
| lapses | int | | driver attention lapses |
| events | list | | (x, y, "ok"/"fp", kind, …) per intervention onset; kind ∈ evasive, steer, progress, gate:<action> |
| burden | dict | s | needless_s, needless_steer_s, needless_throttle_s, needed_* |
| goal, trace | (x, y); list of (x, y) | m | |
| shadow | list | | takeover features + label (decision-dataset runs only) |
### 19.3 Monte Carlo summary (`models/mc_live/status.json`, `models/mc_bound.json`)
Per variant: runs, crashes, reached_goal, interventions, episodes, progress_assists, needless, needless_takeovers / brakes / limits,
overridden / wheel-taken needlessly (s), median_time_s, min_clearance_cm; paired tests {a, b, key, sums, reduction_pct, p}.

## 20. ML sample and model file
| Item | Content |
|---|---|
| Tick vector | 12 floats: stick/30, throttle/255, v, free on 5 arcs /2.5, reverse free /2.5, TTC, free / stopping distance, reaction distance |
| Window | 32 × 12 (1.6 s), newest last; padded with the first tick |
| Label | 1 = contact or clearance < 2 cm within 2 s without ADAS |
| Dataset (.npz) | Z, y, tte, lapsed, v_true, clear, run, tick, run_family, run_style, run_crashed, run_seed, run_dr, dr_keys |
| Decision dataset (.npz) | X (flat features at takeover), y (needed), p_v3, seed, style, lapsed, ratio, t |
| Model JSON | kind (trees3 / mlp3 / gru3 / trees / regressor), temperature, weights or tree arrays (feature, threshold, left, right, leaf, value, baseline, max_depth) |

---

| Version | Date | Change |
|---|---|---|
| 1.0 | 29 Sep 2026 | First baseline |
