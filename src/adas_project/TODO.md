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
- [ ] B4. Brief throttle cut mid-manoeuvre in the doorway run (t = 3.6 s): find the cause in the sim log.
- [ ] B5. (12 -> 9 needless of 44 by triggering evasive steer only on real contact courses; narrowing the gate's steering-slop band made it worse (13), reverted. Remaining: 5 gate speed-limits, 4 evasive)  Measure false-positive interruptions: count brakes/limits/swerves in normal driving (sim Monte Carlo +
  real logs) and tune down.
- [ ] B6. Corridor centring, limiter, narrow gap and side/rear alerts not re-checked since the path gate replaced the
  cone gate.
- [ ] B7. `pi/path_predict.py` `VP` still has the old body (14 cm wide): switch users to `pi/relay_assists.car_params`.

- [ ] B8. CBF safety filter (RESEARCH.md 2) replacing the heuristic speed caps in `pi/path_gate.py`.
- [ ] B9. MPPI local fallback when Hybrid A* has no path or execution deviates.
- [x] B10. (done inside B2: the planner adds reverse legs when no forward path exists) "Unstuck" mode with reversing (Reeds-Shepp) - the user's option 1.

- [ ] B11. Brake hold stops 0.75 m short of a wall at full speed (old speed model in the relay) - fixed by C3.
- [x] B13. (fixed) Car froze next to a box: points already inside the body margin blocked every direction, even
  backing away. The path gate now ignores close points the motion moves away from (same margin box as the sweep).
- [ ] B12. Unit tests for the Hybrid A* planner and the evasive state machine (doorway, boxed, reverse case).

## C. Localisation and car model
- [ ] C1. **Research accurate 2D LiDAR odometry** (research done: RESEARCH.md section 3 - RF2O + KISS-ICP + EKF; implementation NOT done) (linear and angular velocity from the LiDAR is poor now):
  range-flow / ICP variants, filter fusion with the commands (RESEARCH.md), then implement and measure on logs.
- [ ] C2. Physics-informed ML car model (`adas/car_model.py`, `sim/car_model_eval.py`, uncommitted). First version lost
  to the plain fit (15-19 cm vs 5.7 cm). Rewritten, NOT re-run. Commit only if it beats the plain fit.
- [ ] C3. (speed model done: `pi/car_model.json` from the logging drive, loaded by the relay and the Monte Carlo via `apply_car_model`; min clearance 4 -> 6 cm. Still to do: curvature/servo centre in the relay, the ML model when C2 passes) Use the model everywhere: relay speed estimate, gate speed caps, curvature for prediction/planning,
  the simulator's virtual car. (Fixes the relay's speed-model mismatch.)
- [ ] C4. Start-of-drive calibration run (panel): drive the legs, fit on the Pi, save `pi/car_model.json`, show the
  fit. Runs only when the user asks.

- [ ] C5. RF2O range-flow odometry for speed and yaw rate; EKF fusing it with the car model.
- [ ] C7. Scan de-skewing (KISS-ICP style): the car moves ~3 cm during one 0.1 s LiDAR rotation (measured as a
  -3.7 cm bias in the twin LiDAR check). De-skew real scans in the odometry, and add the same skew to the
  simulated LiDAR so the twin reproduces it.
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
- [x] D3. (done v1: reports/twin_lidar.png: static occupancy map from even scans, simulated beams at held-out scan poses, median |sim-real| 3.7 cm over 9953 beams; the -3.7 cm bias is scan skew from the car moving during the 0.1 s rotation - see C7) **Real LiDAR vs simulated LiDAR demo**: take real scans from a log, rebuild the room, raycast the simulated
  LiDAR from the same poses, show both overlaid with the range error statistics.
- [x] D4. (done v1: `python -m sim.relay_mc 24` on the car's own decision code -> reports/monte_carlo_relay.png: no ADAS 17/24 crashes, ADAS 0/24 crashes and 24/24 reach the goal, 39 interventions of which 12 needless) **Visual Monte Carlo** of real-world scenarios (random rooms, obstacles, pedestrians, driver lapses) on the
  car's own relay code, side by side runs.
- [x] D5. (done via D8) (was: run, but NO difference yet: the steering-trend intent predictor in `pi/path_gate.py` never changed a decision - 12 needless interventions in both. Next: a learned intent model (GRU over stick/throttle/speed history, RESEARCH.md 4) predicting the driver's path distribution, and a driver model that telegraphs intent (gradual steering) so the comparison is fair) **Intent-aware vs not intent-aware** comparison on the same Monte Carlo runs (crashes, interruptions,
  warning lead time).
- [x] D8. (done: learned crash-risk intent model, AUC ~0.87 on held-out rooms; intent decides steering takeovers only. 48 paired drives: needless takeovers 17 -> 7 (-59%), 10 better / 0 worse, Wilcoxon p = 0.0008, crashes 0; brake-only baseline added. RESEARCH.md section 4) **Intent-aware must show a significant, genuine difference**
- [ ] D6. GUI "connection lost" flicker while the planner runs.

## E. GUI (EV-grade frontend)
- [ ] E1. (v1 done: `pi/dash/index.html` at /dash on the car and in the simulator - 3D scene, speed/gear/throttle/steering cluster, time-to-contact ring, mode chip, alert banner, assist toggles, events, camera slot. Still to do: modes page (Drive/Assist/Autonomy/Diagnostics/Replay), intent bars, map) Redesign: EV-style dashboard (speed, gear/direction, ADAS state, predicted path, alerts, camera slot),
  useful modes (Drive, Assist, Autonomy, Diagnostics, Replay), all relevant information visible.
- [ ] E2. Show "what might happen": predicted path, time to collision, intent probabilities, planned manoeuvre,
  alternatives considered.

## F. Autonomy (show everything the car can do)
- [ ] F1. List and expose all autonomy modes in the GUI: obstacle avoidance run, follow-the-leader, return to start,
  explore/map the room, point-to-point navigation (click a goal on the map), auto-park (with camera later).

## G. Camera (one camera, front or rear) - plan first
- [x] G1. Write the camera plan (done: RESEARCH.md section 5) (RESEARCH.md): what it adds to every existing and future feature.

## H. Keep improving
- [ ] H1. (ongoing: RESEARCH.md section 6) Keep a running list of new feature ideas (autonomy, ML, visualisation) as the car is observed.

## I. Verification
- [ ] I1. User drives the laptop simulator and reports issues; fix from `logs/sim/`.
- [ ] I2. Same on the real car.

## J. Housekeeping
- [ ] J1. Tests for the new pieces (path gate, memory pruning, planner, hw simulator).
- [ ] J2. Update `STATUS.md` and `README.md`.
- [ ] J3. Results/slides from sim + car logs.
- [ ] J4. Calibrations only when the user asks (LiDAR yaw 63.4 deg current).
