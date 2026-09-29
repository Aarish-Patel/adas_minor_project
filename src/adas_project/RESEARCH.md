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

**Done so far (least-restrictive filter in `pi/path_gate.py`)** - the gate only reacts to what the car can actually
do before its next decision (Hsu, Hu & Fisac, *The Safety Filter: A Unified View of Safety-Critical Control in
Autonomous Systems*, Annual Review of Control, Robotics, and Autonomous Systems 2024; Wabersich & Zeilinger,
predictive safety filter, Automatica 2021):
- the swept path is a composite: any steering command of the last 0.25 s for the command-delay distance (fitted
  0.12 s delay + servo slew), then the current command - each widened by the steering model's error (0.08 1/m +
  15 %, the fitted left/right gains differ by ~15 %). The old fixed +-0.35 1/m band is gone;
- speed-dependent protective field, as AGV laser scanners switch field size with speed (ISO 3691-4): body margin
  1.2 cm up to 0.15 m/s, 2 cm up to 0.4 m/s, 3 cm above - so the car can slow and pass a tight gap it fits;
- braking unchanged: stopping distance x 1.3 must fit the free distance; active braking above the physical limit.
Result (96 paired Monte Carlo drives, 0 crashes before and after): brake-only needless speed limits 117/48 drives
-> 79/96 drives; needless brakes -> ~0; a 32 cm gap for the 20 cm car now passes untouched.

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

**Implemented** (`adas/rf2o.py`, `adas/speed_ekf.py`, in the relay as `pi/relay_assists.RelaySpeed`; evaluation
`python -m sim.odometry_eval` -> `reports/odometry.png`). Speed error while moving (RMSE / median, cm/s):

| | twin | twin, car 20 % slower than its model | real drive (vs smoothed scan matching) |
|---|---|---|---|
| throttle model (the relay until now) | 7.5 / 0.0 | 10.3 / 8.9 | 3.5 / 2.8 |
| ICP differentiated (earlier speed measurements) | 4.1 / 1.1 | 3.4 / 1.1 | 6.5 / 4.4 |
| RF2O range flow | 3.7 / 0.7 | 3.3 / 0.7 | 3.9 / 2.2 |
| **EKF: car model + RF2O** | **3.6 / 0.6** | **3.1 / 0.5** | **3.4 / 2.1** |

Yaw rate from the EKF: 2.1 deg/s RMSE on the real drive (the relay had none). RF2O costs 0.5 ms per scan.
On a steady drive on a flat floor the fitted throttle model is about as good as the LiDAR; the EKF matters when
the model is wrong - coasting and braking, a sagging battery, a carpet. Found on the way: while the gate brakes, the
throttle model believes the car stops almost at once (0.83 -> 0.03 m/s in 0.2 s) while range flow shows it still
at 0.6-0.8 m/s. The EKF clips acceleration to the fitted limits and re-initialises from the LiDAR after two gated
outliers. For braking the relay uses the more conservative of the two estimates, so this can only brake earlier.

## 4. Driver intent

Current: `adas/intent.py` (learned intent estimator) feeding the warnings. References for the upgrade:
- CARPAL: confidence-aware intent recognition for parallel autonomy (arXiv 2003.08003).
- Uncertainty-aware driver trajectory prediction for parallel autonomy (arXiv 1901.05105).
- CNN-GRU driver steering intention prediction for shared control (2025).
Plan: GRU over the recent stick/throttle/speed history + scene features -> distribution over the driver's
intended path; the safety filter and warnings use the predicted path distribution instead of the current stick
only. Measured in the Monte Carlo as intent-aware vs not (crashes, interruptions, warning lead time).

### Result (sim/relay_mc.py, sim/mc_stats.py, sim/train_intent_net.py)
Learned crash-risk model (MLP on stick history, stick activity, time since the stick moved, LiDAR free distance on
five candidate arcs, and an online personal reaction-distance profile), trained on simulated drivers, tested on
held-out rooms: AUC 0.86-0.88. Used only to decide whether to take over the STEERING; braking stays pure physics.
The relay, the Monte Carlo and the scenario checks all run the same class (`pi/relay_assists.RelayIntent`).

96 paired drives (48 random rooms x lapsing / late-but-competent drivers), the car's own relay code on the twin
(least-restrictive gate, LiDAR+model speed EKF, swerve trigger 1.2 s / 0.7 s attentive, intent model retrained on
the same noisy 720-beam sensing the relay sees: held-out AUC 0.91), counterfactual ground truth ("needless" = the
driver alone would not have crashed within 2 s):

| | crashes | reach goal | interventions | needless takeovers | needless brakes | needless limits | wheel taken needlessly | overridden needlessly |
|---|---|---|---|---|---|---|---|---|
| no ADAS | 26 | 70 | - | - | - | - | - | - |
| brake only | 0 | 85 | 174 | 0 | 5 | 74 | 0 s | 64 s |
| ADAS (brake + evasive) | 0 | 93 | 98 | 14 | 1 | 17 | 83 s | 101 s |
| **ADAS + learned intent** | **0** | **94** | 96 | **8** | 2 | 27 | **65 s** | **80 s** |

Intent-aware vs the same ADAS without intent, paired one-sided Wilcoxon signed-rank:
- needless steering takeovers 14 -> 8 (6 drives better, 0 worse, p = 0.007);
- time the wheel was taken needlessly 83 -> 65 s (p = 0.037); time overridden needlessly 101 -> 80 s (p = 0.030);
- 0 crashes. The price: a swerve held back for an attentive driver can become a brief speed trim by the physics
  brake (needless limits 17 -> 27, mostly mild).
- Safety limit on the hold (last point to steer, Brannstrom 2010): for a driver NOT moving the stick the hold ends
  where the swerve must start before the brake would act; the shorter attentive trigger (0.7 s) needs an active
  stick. Without it the retrained model let a frozen-stick driver reach the box at full speed before swerving.
- Trust-threshold sweep with the previous model (0.5 / 0.7 / 0.9): takeovers 9 / 8 / 6, crashes 0 throughout.
- If the ADAS is tuned to help early (swerve at 1.6 s), intent is significantly better on every severity measure:
  takeovers 34 -> 13 (p = 0.0001), wheel taken needlessly 171 -> 103 s (p = 0.0009), overridden needlessly
  185 -> 117 s (p = 0.002), throttle removed 64 -> 39 throttle-s (p = 0.008).
- Remaining needless speed limits happen with the driver's own path inside 1.3x its stopping distance (the brake's
  safety factor): the driver was faster than they could stop and swerved late - physics, not a tuning choice.
- Tried and rejected: a later soft cap for trusted drivers (FOS 1.1 / 1.0) - fewer limits but more brakes and
  swerves (21 -> 22 needless). Earlier result (48 drives, old gate): takeovers 17 -> 7, p = 0.0008.

### v3: trained on the randomised digital twin (28 Sep; section 7, `sim/twin_intent_data.py`, `sim/train_intent_torch.py`)
Why v2 predicted badly in general (the user saw it at full speed into a wall):
- it was trained only on moments within 1.2 m of an obstacle, but the relay asks it every tick;
- its view ahead stopped at 1.5 m, although at full speed the car covers ~1.7 m in the 2 s the label looks ahead;
- it looked forward only, and it saw one fixed car with a clean sensor and drivers who slow down.

v3 uses a 12-number tick vector (stick, throttle, speed, free distance on five arcs up to 2.5 m ahead and on the
current arc behind, time to contact, free distance in stopping distances, the driver's reaction distance) over a
1.6 s window. It was trained on 3000 randomised-twin drives (543k labelled ticks) and tested on 600 new drives.
Candidates were trained on the RTX 4060: gradient-boosted trees 0.862 validation AP, MLPs 0.847-0.851, GRUs
0.845-0.846. All plateau after ~30 epochs, so the limit is the data, not the model size. The trees were chosen:
24k nodes, 0.22 ms per tick in numpy on the laptop.

| held-out test (600 drives) | v2 | v3 |
|---|---|---|
| average precision / AUC | 0.42 / 0.73 | 0.86 / 0.96 |
| recall / false alarms at 0.5 | 0.07 / 0.007 | 0.77 / 0.039 |
| calibration error (ECE) | 0.12 | 0.004 |
| crash drives never warned / warned >= 1 s before | 54 % / 8 % | 6 % / 62 % |
| safe drives with a false alarm >= 0.25 s | 4 % | 17 % |

v3 is better in every situation slice (`reports/intent_v3.png`). Its weakest slices are turning (AP 0.65), and
false alarms within 0.5 m (19 %) and while reversing (10 %).

**But better prediction did not mean fewer needless interventions.** In the Monte Carlo on the car's own code
(96 paired drives, lapsing and late drivers, `models/mc_intent_v2_v3.json`), there were 0 crashes with either
model. Needless takeovers went 53 (ADAS) -> 25 (v2) / 26 (v3). Time overridden needlessly went 128 s -> 65 s (v2) /
86 s (v3), and trust thresholds 0.7 / 0.85 did not close the gap. v2 was trained on exactly these Monte Carlo
drivers, and almost never raises its risk (recall 7 %). With the physics brake as a backstop, "trust nearly
everyone" is a good takeover policy there.
Decision: v2 keeps deciding steering takeovers; v3 supplies the P(crash) the driver sees and the warnings.
Next step: learn the takeover decision itself ("would this intervention be needless?") from the Monte Carlo
counterfactual labels, instead of thresholding a crash probability.

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

## 7. Training ML models with the digital twin (TODO M4)

**The problem.** A driver-intent / crash-risk model needs thousands of near-crashes and crashes to learn from.
Collecting those on the real car is slow, risky and breaks things (it cost an ESP32 on 28 Sep). The standard answer
in robotics and automated driving is to generate the data in a simulator. A model trained on one exact simulator,
though, learns that simulator's quirks and fails on the real machine (the "reality gap"). The field closes the gap
with a digital twin that is (1) identified from the real system, (2) randomised around that identification, and
(3) checked against real data in a loop.

**What the literature does** (references from my background knowledge, not a fresh literature search):
| Method | Key reference | Idea |
|---|---|---|
| Digital twin | Grieves & Vickers 2017; Tao et al., IEEE TII 2019 | a virtual copy kept consistent with the physical system through its data |
| System identification | Ljung, *System Identification* (1999) | fit the simulator's parameters (speed response, delays, steering gain) to logged real behaviour |
| Domain randomisation | Tobin et al., IROS 2017 | randomise the simulator's appearance/sensing so the real world looks like one more variation |
| Dynamics randomisation | Peng et al., ICRA 2018 | randomise masses, friction, delays, so the policy works across the whole range, including the real car |
| Automatic DR | OpenAI et al. 2019 (Rubik's cube) | widen the randomisation ranges automatically while the model still performs |
| Real-to-sim-to-real loop | Chebotar et al., ICRA 2019 (SimOpt); Ramos et al., RSS 2019 (BayesSim) | use real rollouts to update the simulator's parameter distribution, retrain, repeat |
| Sim-vs-real agreement metric | Kadian et al., "Sim2Real Predictivity", IEEE RA-L 2020 | does ranking models in simulation predict their ranking on the real robot? |
| Small-scale car platforms | Balaji et al., ICRA 2020 (AWS DeepRacer); O'Kelly et al. 2020 (F1TENTH) | 1/10-1/18 scale cars trained in simulation with randomisation, then run on the real car |
| Safety-critical scenario generation | Feng et al., *Nature* 2023 (dense RL for AV testing) | over-sample the rare dangerous moments that natural driving almost never produces |
| Calibrated probabilities | Guo et al., ICML 2017 | temperature scaling so "P = 0.8" really means 80 % |

**How this project applies it:**
1. *Identified twin.* `sim/hw_sim.py` uses the car model fitted from the real logging drive (`sim/log_fit.py`:
   top speed, dead-band, motor lag, 0.12 s command delay, steering gain) and the braking measured on 28 Sep. Its
   accuracy is measured by replaying real commands (`sim/twin_report.py`: path error 4.5 cm median over 3 s).
2. *Randomised twin* (`sim/twin_intent_data.py`). Every training run draws a different car and sensor around the fit:
   top speed ±15 %, dead-band 6-18 PWM, motor lag 20-80 ms, coast 2.5-5 m/s², brake 5-10 m/s², command delay
   60-200 ms, steering gain ±15 %, servo centre ±3°, LiDAR range noise 4-20 mm, dropout 0-12 %, yaw-offset error
   ±2°, vibration jitter up to 1.2° per scan, and a speed estimate with 0-100 ms lag, 1-5 cm/s noise and ±10 %
   scale error. The real car's vibration and inconsistency (TODO N) are one more draw from these ranges.
3. *Scenario coverage instead of one driver type.* Random rooms with seven driver styles, plus scripted
   approaches: straight at walls and boxes, alongside a wall, on a curve, reversing, through gaps. Each reacts
   (brake, coast, steer away, stop, or not at all) at a random distance from far too late to comfortably early.
   This over-samples the dangerous moments, in the spirit of Feng et al.
4. *Labels from the twin itself.* Every tick gets its ground truth by letting the run continue with no ADAS: did
   the car touch something, or pass within 2 cm while moving, within 2 s? No hand labelling is involved.
5. *Held-out evaluation by run and by situation.* The test runs use new rooms and newly drawn cars. Metrics are
   reported per slice (situation family, speed, free distance, reversing, turning, lapsed vs attentive, time to
   contact), so a good average cannot hide a situation where the model fails.
6. *Randomisation ablation* (evidence that step 2 matters): a model trained on the nominal twin only vs one
   trained with randomisation, both tested on cars drawn from wider ranges than either saw.
   **Result** (`sim/intent_ablation.py`, `models/intent_ablation.json`, `reports/intent_ablation.png`): the same
   gradient-boosted trees, trained on the same number of ticks (253k) from 1500 nominal-twin drives vs randomised
   drives, tested on 400 drives each:

   | tested on | trained on | AP | AUC | calibration error | crash drives never warned | warned >= 1 s before |
   |---|---|---|---|---|---|---|
   | nominal twin | nominal twin | 0.68 | 0.97 | 0.022 | 6.7 % | 61 % |
   | nominal twin | randomised twins | 0.68 | 0.96 | 0.015 | 5.6 % | 60 % |
   | **1.6x wider, unseen cars** | nominal twin | **0.63** | **0.87** | **0.120** | **8.0 %** | **43 %** |
   | **1.6x wider, unseen cars** | randomised twins | **0.85** | **0.96** | **0.018** | **2.0 %** | **60 %** |

   On the car it was trained for, randomisation costs nothing. On cars it has never seen, the model trained on one
   twin loses a quarter of its precision, becomes badly calibrated, and warns late. The randomised one keeps its
   performance. That is the expected behaviour on the real car, whose exact parameters no twin knows.
7. *Real-to-sim loop (next, when the car runs again).* Real drive logs are scored by the model. The twin's
   randomisation ranges are then re-centred on what the logs show, following SimOpt / BayesSim. The model is
   retrained, and the sim-vs-real agreement of candidate models is reported, as in Kadian et al. (TODO B15).

## 8. What production ADAS displays show, and what this HMI takes from them (TODO Q4, 29 Sep)

Sources read (web, 29 Sep): Tesla FSD visualisation reference (notateslaapp.com, "All Tesla FSD Visualizations and What
They Mean"); Tesla FSD architecture write-ups (thinkautonomous.ai, occupancy networks); Huawei Qiankun ADS 3.0 and
HarmonyOS cockpit coverage (carnewschina.com 2024-04-24, huaweicentral.com); XPeng XNGP (electrek.co, xpeng.com);
NHTSA *Human Factors Design Guidance for Level 2 and Level 3 Automated Driving Concepts* (DOT HS 812 555, 2018 - the PDF
returned 403, so only its abstract-level points are used); ISO 15005 / UN R157 (10 s takeover time) as cited in the
review literature; Wikipedia "Surround-view system" and "Backup camera".
Limits of this research: Huawei and XPeng publish capabilities (LiDAR + 3 mm-wave radars + 11 cameras + 12 ultrasonics,
BEV + GOD obstacle network, CAS 2.0 front/rear/side collision avoidance, parking-space-to-parking-space navigation,
AR-HUD, DMS) but no public description of the pixel-level UI; what is copied below is therefore the *common pattern*
documented for Tesla and the generic surround-view / reversing camera practice, not Huawei's exact screens.

Features documented, and what this project does with each:

| Feature seen in production HMIs | Source | Here |
|---|---|---|
| Planned path as a ribbon; darker where the car will accelerate, faded where it will decelerate/stop; chevrons when slowing | Tesla | planned-path ribbon in the 3D scene (done); accel/decel shading (Q5) |
| Other objects coloured by relevance: grey normal, blue in the planned path, red = action required, brake-light cue | Tesla | object colouring by path relevance and RSS/TTC (Q5) |
| Unclassified obstacle shown as a neutral 'debris' pile rather than hidden | Tesla | LiDAR clusters drawn as neutral blocks, moving ones flagged (Q5) |
| Proximity arcs around the car, grey -> yellow -> red with distance (ultrasonic arcs) | Tesla, all parking assists | proximity arcs from the LiDAR ring (Q5) |
| Reversing camera with steering-dependent guidelines, distance bars and a warning strip | Backup-camera standard practice (US FMVSS 111, 2018) | done (guidelines, `adas/vision/guidelines.py`) |
| **Predicted-position 'ghost' of the car while reversing, hazards in the swept path, STOP strip, puddle/hole warning** | user request 29 Sep; production 'trajectory + object in path' | done: ghost footprints at 0.3/0.6/1.0 m, swept-body contact distance and time, floor-patch warning (Q1-Q2) |
| Surround/bird's-eye and 'transparent chassis' parking views | Huawei, all 360 systems | needs several cameras; one rear webcam only - the LiDAR map already gives the top-down view |
| Collision avoidance front/rear/side; stability control at speed | Huawei CAS 2.0 / XMotion | emergency braking + evasive steer + steering envelope (existing) |
| Parking-to-parking navigation; valet | Huawei | click-to-go with orientation, return to start (existing / new) |
| Driver-state and takeover escalation: staged warnings, takeover within a defined time | NHTSA L2/L3 guidance, UN R157 | intent-based risk -> staged warning -> intervention (existing); takeover-time is not applicable (driver always in control) |
| Speed-limit sign and system-status icons | all | done (cluster limit sign, mode chips, safety pill) |
| Formal safe-distance readout | Mobileye RSS | RSS distance in the safety pill (done, N7) |

Design rules taken from the NHTSA/ISO guidance: keep glance time short (few, large elements on the main view), show
system mode and limits at all times (mode chips, limit sign, steering range), colour escalation green -> amber -> red
with one meaning per colour, and keep detail (diagnostics, tuning) in drawers away from the driving view.
