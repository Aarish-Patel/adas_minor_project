# Algorithm choices and references

Each component uses a published method. This file records what is used, why, and what it replaces.

## 1. Path planning for evasive steering and autonomy: Hybrid A*

**Problem seen:** the offset-lattice evasive planner (swerve to an offset, pass, return) got stuck in the doorway
scenario at full throttle: all its hand-shaped manoeuvres collided, so it gave up and braked.

**Choice: Hybrid A\*** (Dolgov, Thrun, Montemerlo, Diebel, *Path planning for autonomous vehicles in unknown
semi-structured environments*, IJRR 29(5), 2010 - Stanford "Junior", DARPA Urban Challenge).
- Searches the car's continuous state (x, y, heading) with kinematically feasible motion primitives (arcs at
  several steering angles), so every path it returns can be driven by the car.
- Occupancy grid built from the LiDAR scan + obstacle memory, inflated by the body half-width + margin.
- Heuristic = max(obstacle-free non-holonomic distance (Dubins/Reeds-Shepp), obstacle-aware 2D grid distance).
- Analytic expansion (Dubins, or Reeds-Shepp when reversing is allowed) to finish exactly on the goal.
- Goal = rejoin the driver's desired path: any pose on the line ahead of the obstacle, heading along the line.
- Reversing primitives only in "unstuck" mode (the user's option 1: back off / move away from the line and come
  back later), never in normal evasive steering.
- Re-planning while driving: incremental re-use of the search (Incremental Generalized Hybrid A*, arXiv 2508.13392,
  2025) - re-plan from the current pose when new scans invalidate the path.
- Tracking the path: pure pursuit on the planned curvature; speed from the curvature and free distance.

**Click-to-go autonomy (`adas/autonav.py`):** the same Hybrid A* with a position goal (any final heading, as in
goal-region variants used for parking/navigation). The analytic expansion is the single circular arc from a node
through the goal - exactly what pure pursuit (Coulter, CMU-RI-TR-92-01, 1992) will drive - forward, or backward
when the goal is behind (Reeds-Shepp-style reversing, Reeds & Shepp 1990). The planner runs in a worker thread, as a
planner node does in ROS 2 Nav2, and the car holds still while it plans. Supervision follows remote-parking
practice (Tesla Smart Summon, UNECE R79 remote control manoeuvring): the operator holds a dead-man control (the
throttle), releasing it stops the car, steering or braking hands back at once, and after the manoeuvre ends the car
stays stopped until the operator lets go. The path brake still has the last word.

**Quick alternatives when execution gets stuck:** MPPI - Model Predictive Path Integral control (Williams et al.,
ICRA 2016 / T-RO 2018, "Aggressive driving with MPPI", AutoRally). Samples hundreds of control sequences through the
car model and weights them by cost; cheap enough on the Pi at ~300 samples x 20 steps. Used as the local
fallback when Hybrid A* has no solution or the car deviates.

## 2. Braking and steering corrections: control barrier function safety filter

**Problem seen:** heuristic braking rules (cones, then speed caps) either interrupted normal driving or were late.

**Choice: minimally-invasive CBF safety filter** (Ames et al., control barrier functions; applied to shared vehicle
control in Talbot, Suminaka, Thompson, Lew, Orosz, Subosits, *Control Barrier Functions for Shared Control and
Vehicle Safety*, ACC 2025; non-convex obstacles: *Minimal Intervention Shared Control with Guaranteed Safety under
Non-Convex Constraints*, arXiv 2507.02438, 2025).
- Barrier h(x) = free distance along the predicted swept path - stopping distance (reaction + braking at the
  measured deceleration).
- Each tick solve a tiny QP: the command closest to the driver's (throttle, steering) such that
  dh/dt >= -alpha * h. When the driver is safe the filter returns exactly their command (zero interruptions);
  otherwise it changes it as little as possible - first steering, then throttle.
- This replaces the hand-tuned speed-cap formula in `pi/path_gate.py` with a principled guarantee.

## 3. LiDAR localisation (why speed and turn rate from the LiDAR were poor, and the fix)

**Why they were poor:** scan-to-scan point-to-point ICP with fixed correspondence gates, velocity by
differencing noisy poses (noise is amplified by 1/dt), no motion model, no de-skewing of the ~0.1 s rotation, and
points thinned to 190.

**Choices:**
- **RF2O - range-flow planar odometry** (Jaimez, Monroy, Gonzalez-Jimenez, ICRA 2016): estimates the sensor's
  planar velocity (vx, vy, omega) directly from the range-flow constraint of every beam, dense, no
  correspondences, ~1 ms per scan. Gives the linear and angular velocity we are missing.
- **KISS-ICP ideas** (Vizzo et al., RA-L 2023, *In Defense of Point-to-Point ICP*): constant-velocity motion
  prediction, scan de-skewing, voxel subsampling, adaptive correspondence threshold, robust kernel - for our
  pose tracking.
- **Correlative scan matching** (Olson, ICRA 2009) for robust matching when the initial guess is poor (fast turns),
  and **submap scan matching as in Cartographer** (Hess, Kohler, Rapp, Andor, ICRA 2016) for a drift-free room
  map to localise in (enables point-to-point navigation and return-to-start).
- **Fusion:** an EKF with the car model (fitted motor + steering) as the process model and RF2O/ICP as
  measurements -> smooth, lag-free speed and yaw rate for braking, prediction and the GUI.

## 4. Driver intent

Current: `adas/intent.py` (learned intent estimator) feeding the warnings. References for the upgrade:
- CARPAL: confidence-aware intent recognition for parallel autonomy (arXiv 2003.08003).
- Uncertainty-aware driver trajectory prediction for parallel autonomy (arXiv 1901.05105).
- CNN-GRU driver steering intention prediction for shared control (2025).
Plan: GRU over the recent stick/throttle/speed history + scene features -> distribution over the driver's
intended path; the safety filter and warnings use the predicted path distribution instead of the current stick
only. Measured in the Monte Carlo as intent-aware vs not (crashes, interruptions, warning lead time).

### Result (sim/relay_mc.py, sim/train_intent_net.py)
Learned crash-risk model (MLP on stick history, stick activity, time since the stick moved, LiDAR free distance on
five candidate arcs, and an online personal reaction-distance profile), trained on simulated drivers, tested on
held-out rooms: AUC 0.86-0.88. Used only to decide whether to take over the STEERING; braking stays pure physics.
48 paired drives (lapsing + late-but-competent drivers), counterfactual ground truth:

| | crashes | reach goal | interventions | needless steering takeovers | needless speed limits |
|---|---|---|---|---|---|
| no ADAS | 13 | 35 | - | - | - |
| brake only | 0 | 32 | 188 | 0 | 117 |
| ADAS (brake + evasive) | 0 | 47 | 99 | 17 | 43 |
| ADAS + learned intent (threshold 0.5) | 0 | 47 | 92 | 7 | 53 |

Needless takeovers -59%: 10 runs better, 0 worse, one-sided Wilcoxon signed-rank p = 0.0008. Threshold sweep
0.05 -> 0.9: takeovers 13 -> 4, crashes 0 at every threshold (the stopping-distance brake is the safety net).

## 5. Camera plan (one camera; front-facing chosen - it helps every forward feature, rear only helps reversing)

| Use | Method | Feeds |
|---|---|---|
| What each obstacle is (person, chair, box, wall) | YOLO nano (v8n/v11n) via NCNN on the Pi 5 (~10 fps at 320x240) | class-specific margins (people wider, predicted motion), GUI labels |
| Obstacles the LiDAR plane misses (table tops, low objects, drop-offs) | Depth Anything V2 Small monocular depth (2024) at low rate, or a Hailo AI HAT | extra points for the safety filter and planner |
| LiDAR-camera fusion | project LiDAR points into the image (extrinsics from an ArUco/checkerboard calibration) | depth for detections, labels for LiDAR clusters |
| Lane / tape lines | existing `adas/lane.py` | lane keeping assist |
| Parking bay + signs | existing ArUco (`adas/markers.py`, `adas/isa.py`) | auto-park, speed limits |
| Visual odometry | optical flow on the ground plane | fused into the EKF (better turn rate) |
| Driver view in the GUI | predicted path, collision X, planned manoeuvre drawn in perspective on the video (AR overlay) | "what might happen" visualisation |

## 6. Feature ideas (kept up to date while observing the car)
- Room map (Cartographer-style submaps) shown live; click a goal on the map -> Hybrid A* drives there.
- Return-to-start and "explore the room" autonomy using the map.
- Auto-park between two boxes with LiDAR only (Reeds-Shepp parallel/perpendicular parking).
- Guardian mode (driver drives, filter protects) vs Chauffeur mode (autonomy drives, driver supervises).
- Risk heat-map around the car; time-to-collision gauge; intervention counter.
- Replay mode: scrub a drive log, see scans, commands, predictions and decisions at any moment.
- Monte Carlo lab: hundreds of random scenario drives, crash/intervention rates, intent-aware vs not.
- Learned policy (RL, e.g. PPO) trained in the simulator as a comparison to the classical planner.
