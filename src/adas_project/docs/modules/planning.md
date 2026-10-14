# Planning and Autonomy

## Purpose
Produce drivable paths for the evasive manoeuvre and for the autonomy modes (click-to-go, return to start, exploration, parking)
and track them, without ever blocking the control loop. SRS: FR-023, FR-030-034, NFR-06-07.

## Responsibilities
| Function | Method |
|---|---|
| Path search | Hybrid A* over (x, y, θ) with Ackermann arcs; forward-only first, reversing at 7× cost if needed; coarse lattice for goal poses; loops > 270° refused |
| Goal completion | Dubins (forward / reverse) analytic expansion for a goal pose; a single tangent arc for a goal point |
| Local fallback | MPPI when Hybrid A* has no path yet |
| Asynchronous service | planning jobs with a time budget; inline (twin, simulated Pi delay) or a worker process (car) |
| Path tracking | pure pursuit; speed from curvature and distance to stop; cusp handling for reversing legs; re-plan when blocked; stall -> re-plan (3×) -> give up |
| Closed-loop speed | feedforward from the throttle model + PI with anti-windup, only on ADAS-commanded throttle |
| Return to start | goal = world origin with the original heading, from the loop-closed pose |
| Exploration | log-odds occupancy grid, nearest reachable frontier (BFS over robot-radius-safe cells), tried goals blacklisted 60 s |
| Parking | parallel on a continuous lane: gaps from the scan, the viable slot nearest to the click, reverse S or a multi-stroke exit plan driven backwards (ADR-032/033), re-measured beside the gap; bay: ArUco marker + LiDAR wall, the stop follows the wall (ADR-036); straightening strokes, centring creep, stall -> re-plan from here |

## Inputs
Obstacle points in the vehicle frame (incl. blind-zone memory) · start pose · goal (point or pose) · pose updates while driving ·
measured speed · operator commands (GOTO / HOME / EXPLORE / PARK / cancel), throttle held as dead-man switch.

## Outputs
Path: array of poses with direction per point (+1 forward / −1 reverse), in the manoeuvre start frame · per-tick (curvature, target
speed) · status message (planning / driving / arrived / blocked / stuck) · current leg for the path gate.

## Dependencies
NumPy, SciPy. Consumes perception outputs. Used by safety_decision (evasive, autonomy legs inside `RelayAssists`).

## Public interfaces
| Interface | Contract |
|---|---|
| `HybridAStar` (`adas/hybrid_astar.py`) | `plan_to_point(points, start, goal, allow_reverse, max_nodes, budget_s, goal_heading, w_reverse, coarse)` -> path or None; `gave_up` reason |
| `PlanService`, `plan_point_job` (`adas/plan_service.py`) | `submit(fn, ...)` -> Job (`ready(dt)`, `result()`); `budget(s)`; `plan_point_job(..., goal_heading, reverse_first)` |
| `AutoNav` (`adas/autonav.py`) | `start(goal, pts, heading, reverse_first)`; `step(dt, pose, pts, v, held)` -> (kappa, v) or None; `cancel(why)`; `leg()`; `active`, `state`, `msg` |
| `dubins.sample / sample_reverse` | poses along the shortest path to a goal pose |
| `adas/mppi.py` | sampled local control |
| `SpeedController` (`adas/speed_control.py`) | `pwm(v_target, v_measured, dt)`, `reset()` |
| `Explorer`, `OccupancyGrid`, `pick_frontier` (`adas/explore.py`) | `step(t, pose, pts, nav_active, goto)`; `done`, `msg` |
| `find_bays`, `park_goal`, `find_parallel_slots`, `park_goal_parallel` (`adas/park.py`) | bay / slot detection and goal pose |
| `home_goal(pose)` (`pi/relay_assists.py`) | world origin as a vehicle-frame goal |
| `plan_parallel(strip, click, car, pose, points)` (`adas/park_select.py`) | gaps from `points` in the lane band (`find_gaps`), viable slots (`choose_slots`: >= car + 0.30 m, slid toward the click), the nearest with a manoeuvre: reverse S or `_tight_gap` exit plan; `refine_parallel` re-measures alongside |
| `plan_perpendicular(marker, car, pose, points)` (`adas/park_select.py`) | ArUco rear-wall marker (`adas/markers.MarkerObs`) -> bay centre line -> reverse path aligned with it; `detect_rear_marker`, `fuse_markers` give the marker pose from camera frames |
| `ParkingPlan` | `selected_space_id` (label), `parking_goal` (x, y, heading°, rear axle), `planned_path` ((N, 4) x, y, θ, direction), `slot` (4 corners, parallel), `request` (lane + click for the re-plan, or the bay's wall); `None`s when nothing can be planned |

## Test strategy
- Unit: `tests/test_planning_safety.py` (doorway, reverse, boxed, click-to-go), `test_explore_park.py`, `test_speed_control.py`,
  `test_submaps.py` (return to start).
- Twin evaluations: `sim/autonav_eval.py` (6 goals), `sim/explore_eval.py`, `sim/park_eval.py`, `sim/home_eval.py`,
  `sim/speed_ctrl_eval.py`; planner timing `tools/plan_bench.py`.

## Performance targets
| Metric | Target | Current |
|---|---|---|
| Plan time | ≤ 0.5 s typical, bounded | 1-440 ms twin |
| Click-to-go | 6/6 goals, ≤ 8 cm | 6/6, 4-6 cm |
| Return to start | ≤ 30 cm, ≤ 15° | 11 cm mean in a bare room |
| Exploration | ≥ 90 % floor, 0 contacts | 97.7-100 % |
| Parking | ≤ 5 cm, ≤ 5°, 0 contacts, never stuck (`sim/parking_validation.py`, deterministic twin) | 68-71/86 pass; 0 contacts, 0 stuck; median 2.7 cm, p90 6 cm, max 9.8 cm; heading median 1.7° |
| Click-to-go scenarios | all pass, ≤ 1 steering reversal / m | 26/26 (`python -m sim.autonav_eval suite 2`), worst 0.56 / m |
| Speed tracking | ≤ 4 cm/s RMSE with ±30 % model error | 2.2-3.3 cm/s |

## Files
`adas/hybrid_astar.py`, `adas/dubins.py`, `adas/mppi.py`, `adas/plan_service.py`, `adas/autonav.py`, `adas/explore.py`, `adas/park.py`,
`adas/speed_control.py`; autonomy glue (`goto`, `explore`, `park`, `_navigate`) in `pi/relay_assists.py`.

## Key parameters
| Parameter | Value |
|---|---|
| Grid resolution / heading bins | 0.05 m; 7.5° (15° when reversing) |
| Planner steering limit | κ ≤ 1.5 1/m (tightest reliable turn ~0.65 m radius) |
| Budgets | evasive 0.20 s, back-off 2.0 s, click-to-go 2.0 s, parking 4× click-to-go; simulated Pi slowdown `PI_COMPUTE_FACTOR` |
| Reverse cost | `GOTO_W_REVERSE` 6 -> 1 m backwards costs 7 m forwards |
| Arrival | 8 cm; heading tolerance 20° (parking 8°); straightening strokes 7 cm at 0.10 m/s, ≤ 10 |
| Cruise / tracking | 0.30 m/s cruise, pure-pursuit look-ahead 0.30 m, lateral-acceleration speed limit ~0.12 g |
| Re-plan | every 0.5 s if the rest of the path is blocked; stall 0.8 s -> re-plan, ≤ 3 |
| Speed PI | kp 90 PWM per m/s, ki 140, integrator ±45 PWM, conditional integration, reset on zero target / direction change |
| Exploration | grid 0.05 m, max range 4 m, robot radius 0.16 m, frontier ≥ 4 cells, min goal distance 0.5 m, blacklist 60 s, re-plan 1 s |
| Parking | bay: width ≥ car + 2×6 cm, depth ≥ 0.40 m, reverse-in nose out; parallel: length ≥ car + 2×9 cm (in practice ≥ ~3.4 car lengths = 1.1 m for the 0.4 m turning radius) |
| Loops | Dubins paths turning > 270° refused (a looping approach was seen on the car) |
| Parallel parking (`adas/park_select.py`) | κ 1.5 1/m (S tried at 80 % first); exact-rectangle clearance 4 cm (13 cm tried first), strokes end 5 cm off; gap >= car + 0.30 m, lane >= car width + 8 cm; slot <= 1.6 m; goal 7 cm off the kerb; re-measure within 0.9 m of the reversing point, re-select among gaps within 1.5 m; arrive 3 cm / 4°, centring to 1.5 cm along |
| Perpendicular (ArUco) | rear bumper 6 cm from the wall, 30 cm straight reverse on the marker centre line, marker pose fused over several frames; forward staging pose 45 cm further out when the bay cannot be reached by one reverse move |
