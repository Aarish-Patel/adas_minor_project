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

## Digital twin for the evaluator (requested, NOT done)
17. **3D view of the car simulator, alongside the car GUI.** Today the two are separate: the 3D viewer
    (`python server.py`, :8765) runs the older simulator ADAS pipeline, NOT the car's relay, so it does not show
    the path gate, brake latch, obstacle memory, lattice evasive steer or scan-matching odometry. Needed:
    - stream the virtual car (`sim/hw_sim.py`: true pose, world, LiDAR scan) and the relay's plan/assist/gate
      state from `tools/sim_car.py` to the 3D viewer, so one drive shows in both: 3D (world, car, LiDAR rays,
      predicted path, collision X, manoeuvre + original line) and the 2D car GUI (:8090)
    - 3D car model with the measured body (20 x 44 cm, wheelbase 20 cm) and steering from the servo angle
    - worlds shared between both (`sim/hw_worlds.py`), including `log:` rooms rebuilt from real scans
18. **Accuracy to the real car, shown as evidence:** the virtual car's motor/steering must come from the fitted
    model (task 2/3), and a "twin check" page: replay a real drive log's commands in the simulator and overlay the
    simulated path on the real (scan-matched) path, with the error. That is the digital-twin / testing-cycle
    demo: real drive -> log -> fit -> simulate -> compare -> fix -> redeploy.

## Later
13. Tests for the new pieces (path gate, memory pruning, lattice planner, hw simulator) in `tests/`.
14. Update `STATUS.md` and `README.md` (simulator instructions, assists, path gate).
15. Results/slides: add evasive-steer and braking results from sim + car logs.
16. Calibrations only when the user asks (LiDAR yaw 63.4 deg is the current value; speed/turn not re-done).
