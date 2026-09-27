# Task list

Every task, asked for or noticed, is written here first and checked off when done. Unfinished work keeps its
current state in the note. Algorithm choices and paper citations are in `RESEARCH.md`.

## A. Blocking the next car session
- [ ] A1. **Deploy to the Pi.** The Pi still runs the version whose obstacle memory made a phantom wall and blocked
  forward driving. Deploy `adas/` and `pi/`, restart `rc-relay`, check a forward drive first.

## B. Safety and planning (the car must be impossible to crash, with minimal false interruptions)
- [x] B1. **Research** (done: RESEARCH.md sections 1-2) planning/safety algorithms and pick them (RESEARCH.md): Hybrid A* for the evasive path,
  a minimally-invasive safety filter (control barrier functions) for braking/steering corrections, fallback
  manoeuvres.
- [x] B2. (done, awaiting the user's sim drive: Hybrid A* `adas/hybrid_astar.py`, pure pursuit, speed cap 0.40 m/s, reverse legs, wait-and-retry keeping the original line; doorway at full throttle passes through the relay, no crash) **Evasive steer gets stuck in the doorway scenario, especially at full throttle** (user report). Replace the
  offset lattice with a Hybrid A* (or equivalent researched) planner over the LiDAR occupancy grid, aimed at the
  driver's desired path; re-plan quickly when execution gets stuck; speed limited to what the plan can do.
- [x] B3. (done: 40 cm box now swerves, 13 cm clearance) Wide obstacles trigger too late (40 cm box -> brakes instead of swerving). Plan earlier when the needed
  offset is large. (Likely solved by B2.)
- [x] B4. (no longer reproduces: `sim/relay_scenarios.py` doorway at full throttle on the relay code - through
  the door, 6 cm closest, 0 ticks of throttle cut while evading; kept as a regression scenario) Brief throttle cut
  mid-manoeuvre in the doorway run (t = 3.6 s).
- [ ] B5. (12 -> 9 needless of 44 by triggering evasive steer only on real contact courses; narrowing the gate's steering-slop band made it worse (13), reverted. Remaining: 5 gate speed-limits, 4 evasive)  Measure false-positive interruptions: count brakes/limits/swerves in normal driving (sim Monte Carlo +
  real logs) and tune down.
- [ ] B16. **Minimise speed cuts and needless interventions until only extreme, unpredictable driving gets one**
  (user request). Normal and reasonably sloppy driving must never be slowed, braked or steered. Target in the
  Monte Carlo (counterfactual ground truth): needless speed limits and needless takeovers close to 0 with crashes
  still 0; then confirm on real logs. Was: 53 needless speed limits and 7 needless takeovers in 48 drives.
  Step 1 done: the gate sweeps only the steering the car can be on before the next decision (the commands of the
  last 0.3 s + a model error of 0.08 1/m + 15 %, instead of a fixed +-0.35 1/m band) and uses a speed-dependent
  protective field (1.2 cm margin up to 0.15 m/s, 2 cm up to 0.4, 3 cm above, as AGV scanners do). 48 drives,
  crashes still 0: brake-only interventions 188 -> 88 (needless limits 117 -> 37, needless brakes 6 -> 0);
  ADAS 99 -> 54 (limits 43 -> 7, brakes 5 -> 0, takeovers 17 -> 12); ADAS+intent 93 -> 48 (limits 53 -> 14,
  brakes 6 -> 0, takeovers 7), 48/48 reach the goal. 32 cm gap now passes untouched.
  Step 2 done: composite sweep (earlier command only for the command-delay distance, then the current one);
  swerve trigger 1.6 -> 1.2 s (0.9 -> 0.7 s attentive); Monte Carlo now measures the burden (seconds overridden,
  wheel taken, throttle removed) and where each needless intervention happened (free / stopping distance),
  `python -m sim.mc_stats`. 96 drives, 0 crashes: ADAS needless takeovers 34 -> 12, overridden needlessly
  185 s -> 81 s (73 s with intent). Tried and rejected: later soft cap for trusted drivers (more brakes).
  Remaining: needless limits are all inside 1.3x the stopping distance (late swerves) - the next lever is a
  better speed estimate (C5) so FOS 1.3 can come down safely, or steering-based avoidance at the last point to
  steer (B8/B9).
- [x] B6. (done: `python -m sim.relay_scenarios` on the relay code incl. the relay's intent model, 11/11 pass -
  evasive box, doorway full throttle, driver already avoiding, full speed at a wall, centring, centring yields,
  limiter, narrow won't fit, 32 cm gap fits (passes untouched after B16), proximity, no slowing beside a wall)
  Corridor centring, limiter, narrow gap and side/rear alerts re-checked since the path gate replaced the cone gate.
- [x] B7. (done: `VP` = measured body 20 cm wide / front 0.28 / rear -0.05 from the tuning file, steering = the
  fitted 0.0656 rad/m per servo degree about the calibrated centre, the same as the path gate; the old turn-radius
  table is only a fallback) `pi/path_predict.py` `VP` still had the old body (14 cm wide).

- [ ] B8. (partly: the gate is now a least-restrictive filter over the reachable steering with a speed-dependent
  protective field - B16; still to do: steering correction instead of braking when a nearby arc is safe, i.e. the
  QP over (steering, throttle)) CBF safety filter (RESEARCH.md 2) replacing the heuristic speed caps in `pi/path_gate.py`.
- [ ] B9. MPPI local fallback when Hybrid A* has no path or execution deviates.
- [x] B10. (done inside B2: the planner adds reverse legs when no forward path exists) "Unstuck" mode with reversing (Reeds-Shepp) - the user's option 1.

- [x] B11. (checked on the relay code with the fitted speed model: full throttle at a wall now stops 23 cm short,
  was 0.75 m; regression scenario in `sim/relay_scenarios.py`) Brake hold stopped 0.75 m short of a wall at full speed.
- [x] B13. (fixed) Car froze next to a box: points already inside the body margin blocked every direction, even
  backing away. The path gate now ignores close points the motion moves away from (same margin box as the sweep).
- [x] B12. (done in J1) Unit tests for the Hybrid A* planner and the evasive state machine (doorway, boxed, reverse case).

## C. Localisation and car model
- [ ] C1. **Research accurate 2D LiDAR odometry** (research done: RESEARCH.md section 3 - RF2O + KISS-ICP + EKF; implementation NOT done) (linear and angular velocity from the LiDAR is poor now):
  range-flow / ICP variants, filter fusion with the commands (RESEARCH.md), then implement and measure on logs.
- [x] C2. (done: held-out 3 s error physics 5.5 cm vs plain fit 5.7 cm vs physics+ML 5.8 cm - ML correction not useful on one 2-min drive, so the selection keeps whichever wins; found a steering asymmetry: left 0.0151 vs right 0.0113 rad per servo degree. More varied logs needed for the ML part) Physics-informed ML car model (`adas/car_model.py`, `sim/car_model_eval.py`, uncommitted). First version lost
  to the plain fit (15-19 cm vs 5.7 cm). Rewritten, NOT re-run. Commit only if it beats the plain fit.
- [ ] C3. (speed model done: `pi/car_model.json` from the logging drive, loaded by the relay and the Monte Carlo via `apply_car_model`; min clearance 4 -> 6 cm. Still to do: curvature/servo centre in the relay, the ML model when C2 passes) Use the model everywhere: relay speed estimate, gate speed caps, curvature for prediction/planning,
  the simulator's virtual car. (Fixes the relay's speed-model mismatch.)
- [ ] C4. Start-of-drive calibration run (panel): drive the legs, fit on the Pi, save `pi/car_model.json`, show the
  fit. Runs only when the user asks.

- [ ] C5. RF2O range-flow odometry for speed and yaw rate; EKF fusing it with the car model.
- [ ] C7. (LOW PRIORITY - premise disproved: the -3.7 cm twin bias was NOT scan skew; it was the same with the car
  standing still (-3.7) and moving (-3.6), and worst on grazing beams. It was a map artefact, fixed in D3b.
  Skew is not measurable at the logged speeds; it only matters near full speed, ~8 cm per rotation at 0.8 m/s.)
  Scan de-skewing (KISS-ICP style) in the odometry and the simulated LiDAR.
- [x] D3b. Twin LiDAR map built properly: log-odds occupancy with free-space carving (Moravec & Elfes; Probabilistic
  Robotics ch. 9) instead of "cells hit 3 times". Median |sim - real| 3.7 -> 1.1 cm, bias -3.7 -> -0.6 cm, 86% of
  9928 beams within 5 cm, bias now uniform in every direction.
- [ ] D7. Show the twin report figures in the dashboard (Diagnostics mode) and as slides.
- [ ] C6. Room map (Cartographer-style submaps) for localisation, point-to-point navigation, return-to-start.

- [x] B14. (done: relay samples the stick on a 50 ms clock, features from live scans, P(crash) < 0.5 -> no evasive takeover; dashboard 'Driver intent' card with risk bar, trust state, learned reaction distance) Put the learned intent model + driver profile into the relay on the car (features from live scans,
  `pi/intent_net.json`), shown on the dashboard (risk gauge, "driver is avoiding it" message).
- [ ] B15. Retrain the intent model on real drive logs as they accumulate (the relay records stick + scans).

## D. Simulator / digital twin
- [ ] D1. (v1 done: the /dash 3D view shows the live relay state for the car or the simulator, incl. true walls, predicted path ribbon, contact X, manoeuvre, line. Still to do: world-fixed map frame, 3D obstacle models instead of LiDAR strokes) **3D view of the car simulator next to the car GUI**: one drive shown in 3D (world, car with the measured
  body and live steering, LiDAR rays, predicted path, collision X, manoeuvre + original line) and in the 2D GUI.
- [x] D2. (done v1: `python -m sim.twin_report` -> reports/twin_path.png: 48 x 3 s replays of the real logging drive, twin ends 4.5 cm from the car median, 9.6 cm 90th) **Twin accuracy evidence**: replay a real log's commands in the simulator, overlay simulated vs real
  (scan-matched) path with the error; the digital-twin testing cycle page.
- [x] D3. (done v1: reports/twin_lidar.png: static occupancy map from even scans, simulated beams at held-out scan poses, median |sim-real| 3.7 cm over 9953 beams; now 1.1 cm with the carved occupancy map, see D3b) **Real LiDAR vs simulated LiDAR demo**: take real scans from a log, rebuild the room, raycast the simulated
  LiDAR from the same poses, show both overlaid with the range error statistics.
- [x] D4. (done v1: `python -m sim.relay_mc 24` on the car's own decision code -> reports/monte_carlo_relay.png: no ADAS 17/24 crashes, ADAS 0/24 crashes and 24/24 reach the goal, 39 interventions of which 12 needless) **Visual Monte Carlo** of real-world scenarios (random rooms, obstacles, pedestrians, driver lapses) on the
  car's own relay code, side by side runs.
- [x] D5. (done via D8) (was: run, but NO difference yet: the steering-trend intent predictor in `pi/path_gate.py` never changed a decision - 12 needless interventions in both. Next: a learned intent model (GRU over stick/throttle/speed history, RESEARCH.md 4) predicting the driver's path distribution, and a driver model that telegraphs intent (gradual steering) so the comparison is fair) **Intent-aware vs not intent-aware** comparison on the same Monte Carlo runs (crashes, interruptions,
  warning lead time).
- [x] D8. (done: learned crash-risk intent model, AUC ~0.87 on held-out rooms; intent decides steering takeovers only. 48 paired drives: needless takeovers 17 -> 7 (-59%), 10 better / 0 worse, Wilcoxon p = 0.0008, crashes 0; brake-only baseline added. RESEARCH.md section 4) **Intent-aware must show a significant, genuine difference**
- [ ] D6. GUI "connection lost" flicker while the planner runs.

## E. GUI (EV-grade frontend)
- [ ] E1. (v1 done: `pi/dash/index.html` at /dash on the car and in the simulator - 3D scene, speed/gear/throttle/steering cluster, time-to-contact ring, mode chip, alert banner, assist toggles, events, camera slot. Still to do: modes page (Drive/Assist/Autonomy/Diagnostics/Replay), intent bars, map - to be built in the native GUI, E3) Redesign: EV-style dashboard (speed, gear/direction, ADAS state, predicted path, alerts, camera slot),
  useful modes (Drive, Assist, Autonomy, Diagnostics, Replay), all relevant information visible.
- [ ] E3. **Native (locally running) GUI instead of web GUIs** (user request), above all for the 3D simulator /
  digital twin: browser rendering and HTTP polling add lag. Follow the field's norm (Gazebo, CARLA, Webots and
  RViz are native apps): pick a native Python 3D toolkit, measure frame rate and input-to-screen latency against
  the web dashboard, and build the remaining E1 modes there. Keep the web page only for viewing from a phone.
- [ ] E2. Show "what might happen": predicted path, time to collision, intent probabilities, planned manoeuvre,
  alternatives considered.

## F. Autonomy (show everything the car can do)
- [x] F1a. Click-to-go (done: `adas/autonav.py` - Hybrid A* to a point with a single-arc analytic expansion and
  Reeds-Shepp-style reverse arcs, planner in a worker thread, pure pursuit + curvature/distance speed profile,
  re-plan when blocked. Relay: `GOTO x y` / `/api/goto/x/y`, operator holds the throttle as the dead-man switch,
  steer/brake hands back, holds after arrival until the throttle is released. 2D GUI: click the map. Twin
  (`python -m sim.autonav_eval` -> reports/autonav.png): 6/6 goals reached - doorway, side goal, furnished room,
  point beside the car, 52 cm gap, reverse in a corridor - 4-7 cm from the goal, 0 crashes, plans 1-440 ms.
  End-to-end through the relay in the simulator: doorway and corridor both arrive.)
- [ ] F1. (click-to-go done, F1a; the rest still to do) List and expose all autonomy modes in the GUI: obstacle avoidance run, follow-the-leader, return to start,
  explore/map the room, point-to-point navigation (click a goal on the map), auto-park (with camera later).

## G. Camera (one camera, front or rear) - plan first
- [x] G1. Write the camera plan (done: RESEARCH.md section 5) (RESEARCH.md): what it adds to every existing and future feature.

## H. Keep improving
- [ ] H1. (ongoing: RESEARCH.md section 6) Keep a running list of new feature ideas (autonomy, ML, visualisation) as the car is observed.

## I. Verification
- [ ] I1. User drives the laptop simulator and reports issues; fix from `logs/sim/`.
- [ ] I2. Same on the real car.

## J. Housekeeping
- [ ] J1. (partly done: tests/test_planning_safety.py - Hybrid A* doorway/reverse/boxed, click-to-go (6), gate beside-wall/head-on/not-frozen/phantom/latch, intent model, relay scenarios on the twin (2), Monte Carlo smoke; 58 tests pass. Still: dashboard) Tests for the new pieces (path gate, memory pruning, planner, hw simulator).
- [x] J2. (done: STATUS.md rewritten for the current state - what is/isn't on the car, calibration in use, results;
  README.md leads with the digital twin and its checks, the older simulator kept below) Update `STATUS.md` and `README.md`.
- [ ] J3. Results/slides from sim + car logs.
- [ ] J4. Calibrations only when the user asks (LiDAR yaw 63.4 deg current).
