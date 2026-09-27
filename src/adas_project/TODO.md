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
- [ ] B2. **Evasive steer gets stuck in the doorway scenario, especially at full throttle** (user report). Replace the
  offset lattice with a Hybrid A* (or equivalent researched) planner over the LiDAR occupancy grid, aimed at the
  driver's desired path; re-plan quickly when execution gets stuck; speed limited to what the plan can do.
- [ ] B3. Wide obstacles trigger too late (40 cm box -> brakes instead of swerving). Plan earlier when the needed
  offset is large. (Likely solved by B2.)
- [ ] B4. Brief throttle cut mid-manoeuvre in the doorway run (t = 3.6 s): find the cause in the sim log.
- [ ] B5. Measure false-positive interruptions: count brakes/limits/swerves in normal driving (sim Monte Carlo +
  real logs) and tune down.
- [ ] B6. Corridor centring, limiter, narrow gap and side/rear alerts not re-checked since the path gate replaced the
  cone gate.
- [ ] B7. `pi/path_predict.py` `VP` still has the old body (14 cm wide): switch users to `pi/relay_assists.car_params`.

- [ ] B8. CBF safety filter (RESEARCH.md 2) replacing the heuristic speed caps in `pi/path_gate.py`.
- [ ] B9. MPPI local fallback when Hybrid A* has no path or execution deviates.
- [ ] B10. "Unstuck" mode with reversing (Reeds-Shepp) - the user's option 1.

## C. Localisation and car model
- [ ] C1. **Research accurate 2D LiDAR odometry** (research done: RESEARCH.md section 3 - RF2O + KISS-ICP + EKF; implementation NOT done) (linear and angular velocity from the LiDAR is poor now):
  range-flow / ICP variants, filter fusion with the commands (RESEARCH.md), then implement and measure on logs.
- [ ] C2. Physics-informed ML car model (`adas/car_model.py`, `sim/car_model_eval.py`, uncommitted). First version lost
  to the plain fit (15-19 cm vs 5.7 cm). Rewritten, NOT re-run. Commit only if it beats the plain fit.
- [ ] C3. Use the model everywhere: relay speed estimate, gate speed caps, curvature for prediction/planning,
  the simulator's virtual car. (Fixes the relay's speed-model mismatch.)
- [ ] C4. Start-of-drive calibration run (panel): drive the legs, fit on the Pi, save `pi/car_model.json`, show the
  fit. Runs only when the user asks.

- [ ] C5. RF2O range-flow odometry for speed and yaw rate; EKF fusing it with the car model.
- [ ] C6. Room map (Cartographer-style submaps) for localisation, point-to-point navigation, return-to-start.

## D. Simulator / digital twin
- [ ] D1. **3D view of the car simulator next to the car GUI**: one drive shown in 3D (world, car with the measured
  body and live steering, LiDAR rays, predicted path, collision X, manoeuvre + original line) and in the 2D GUI.
- [ ] D2. **Twin accuracy evidence**: replay a real log's commands in the simulator, overlay simulated vs real
  (scan-matched) path with the error; the digital-twin testing cycle page.
- [ ] D3. **Real LiDAR vs simulated LiDAR demo**: take real scans from a log, rebuild the room, raycast the simulated
  LiDAR from the same poses, show both overlaid with the range error statistics.
- [ ] D4. **Visual Monte Carlo** of real-world scenarios (random rooms, obstacles, pedestrians, driver lapses) on the
  car's own relay code, side by side runs.
- [ ] D5. **Intent-aware vs not intent-aware** comparison on the same Monte Carlo runs (crashes, interruptions,
  warning lead time).
- [ ] D6. GUI "connection lost" flicker while the planner runs.

## E. GUI (EV-grade frontend)
- [ ] E1. Redesign: EV-style dashboard (speed, gear/direction, ADAS state, predicted path, alerts, camera slot),
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
