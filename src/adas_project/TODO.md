# Remaining tasks (as of 2026-09-28)

## Blocking the next car session
1. **Deploy to the Pi.** The Pi still runs the version whose obstacle memory created a phantom wall and blocked
   forward driving. Deploy `adas/` and `pi/` (commits since `32fdb5a`: memory pruning, brake latch, lattice
   evasive planner, scan-matching odometry, predicted-path overlay, GUI), restart `rc-relay`, and check a forward
   drive before anything else.

## In progress (uncommitted)
2. **Physics-informed ML car model** (`adas/car_model.py`, `sim/car_model_eval.py`). First version was worse than
   the plain fit on held-out legs (15-19 cm vs 5.7 cm, 3 s open loop). Rewritten (motor ODE fitted by output error,
   smooth ML corrections on the steady-speed curve and servo linkage) but NOT re-run. Next: run
   `python -m sim.car_model_eval`, compare, commit only if it beats the plain fit.
3. **Use the model everywhere** once it is good: relay speed estimate, the gate's speed caps (`pwm_for_speed`),
   curvature from servo for prediction/planning, and the simulator's virtual car (`sim/hw_sim.py`).
4. **Start-of-drive calibration run**: turn the panel's Logging drive into "calibration drive": drive the legs,
   fit the model on the Pi, save `pi/car_model.json`, show the fit on the GUI. (Runs only when the user asks.)

## Known issues to fix
5. Evasive steer triggers too late for wide obstacles (40 cm box -> brakes instead of swerving). Plan earlier when
   the needed offset is large.
6. Brief throttle cut mid-manoeuvre in the doorway run (t = 3.6 s) - find the cause in the sim log.
7. GUI "connection lost" flicker while the planner runs (planner on the relay thread starves the GUI server).
8. Relay speed model (hand-measured 0.896 m/s / 48.5 PWM) disagrees with the logs -> speed-cap overshoot. Fixed
   by task 3.
9. Corridor centring, limiter, narrow gap and side/rear alerts not re-checked since the path gate replaced the
   cone gate.
10. `pi/path_predict.py` `VP` still has the old body (14 cm wide); anything still using it should use
    `pi/relay_assists.car_params`.

## Verification the user asked for
11. User drives the laptop simulator (`python tools/sim_car.py --world doorway`, then
    `python rc_controller.py --ip 127.0.0.1`, GUI http://localhost:8090/) and reports issues; fix from the sim
    logs in `logs/sim/`.
12. Then the same on the real car, with its drive logs.

## Later
13. Tests for the new pieces (path gate, memory pruning, lattice planner, hw simulator) in `tests/`.
14. Update `STATUS.md` and `README.md` (simulator instructions, assists, path gate).
15. Results/slides: add evasive-steer and braking results from sim + car logs.
16. Calibrations only when the user asks (LiDAR yaw 63.4 deg is the current value; speed/turn not re-done).
