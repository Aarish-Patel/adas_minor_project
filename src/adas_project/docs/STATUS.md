# Project status (intent-aware ADAS on an RC car: Raspberry Pi 5 + RPLIDAR A3 + ESP32)

Last updated 28 Sep 2026. The task list with every open item is `TODO.md`; algorithm choices and references are in
`RESEARCH.md`.

## Where things are

| Piece | Where | State |
|---|---|---|
| Pi | `192.168.1.6`, user `pi` | powered off by the user ("the car has issues") |
| ESP32 | `192.168.1.7` | firmware unchanged |
| Safety relay (the ADAS on the car) | `pi/wifi_drive_safety.py`, systemd `rc-relay` | **the Pi runs an OLD version** (obstacle-memory phantom wall); deploy = TODO A1 |
| Control panel | `pi/control_panel.py`, systemd `rc-panel`, port 8080 | deployed |
| 2D live view / 3D EV dashboard | relay port 8090: `/` and `/dash` | on the laptop simulator; the car gets them with A1 |
| Laptop digital twin | `python tools/sim_car.py --world doorway` + `python rc_controller.py --ip 127.0.0.1` | works: the unchanged relay drives a virtual car + LiDAR |

Only one process may own the LiDAR/ESP32 at a time. Never leave a stray `rc_controller.py` running during tests.

## Calibration in use

| Item | Value | Source |
|---|---|---|
| LiDAR yaw offset | 63.4 deg on the Pi (`pi/tuning_real_car.json` there is authoritative; the laptop copy still says 90) | front/multi-object calibration |
| Servo straight-ahead | 87 (fit 86.76) | logging drive |
| Steering | 0.0656 rad/m of curvature per servo degree; left side ~15 % stronger than right | logging drive (`sim/log_fit.py`, `sim/car_model_eval.py`) |
| Motor | v_max 0.83 m/s, dead-band 11 PWM, tau 0.03 s, command delay 0.12 s (`pi/car_model.json`) | logging drive |
| Body | 20 x 44 cm, front 0.28 / rear -0.05 m from the rear axle, LiDAR 0.12 m ahead of it | ruler |

Calibrations are only run when the user asks.

## What the ADAS does now (relay code; measured on the digital twin)

| Feature | Code | Result |
|---|---|---|
| Path-predicted emergency braking | `pi/path_gate.py` | body swept along the steering the car can be on (last 0.25 s of commands, then the current one, + model error); speed-dependent margin; stopping distance x 1.3; obstacle memory for the LiDAR's blind 20 cm. 0 crashes in 96 random drives; full speed at a wall stops 23 cm short; a 32 cm gap for the 20 cm car passes untouched |
| Evasive steering | `adas/assists.py` + `adas/hybrid_astar.py` | Hybrid A* round the obstacle and back to the driver's line, reverses first if too close, re-plans; doorway at full throttle passes (6 cm closest) |
| Learned driver intent | `adas/intent_net.py`, `pi/relay_assists.RelayIntent` | held-out AUC 0.91; needless steering takeovers 14 -> 8 (p = 0.007), needless override time 101 -> 80 s (p = 0.030); braking never suppressed |
| Speed + yaw rate | `adas/rf2o.py`, `adas/speed_ekf.py`, `pi/relay_assists.RelaySpeed` | EKF of the car model + LiDAR range flow: 3.1 cm/s RMSE even with the car 20 % off its model (throttle model: 10.3); braking uses the more conservative estimate |
| Other assists | `adas/assists.py` | centring, speed-vs-steering limiter, narrow gap, side alerts, dead-man: 11/11 scenarios (`python -m sim.relay_scenarios`) |
| Click-to-go autonomy | `adas/autonav.py` | click the 2D map: Hybrid A* to the point, dead-man throttle, hands back on steer/brake; 6/6 twin goals, 4-7 cm from the goal |
| Digital twin accuracy | `sim/twin_report.py` | motion 4.5 cm median over 48 x 3 s replays of a real drive; LiDAR 1.1 cm median over 9928 beams |

Monte Carlo (96 paired drives, `python -m sim.relay_mc 48` then `python -m sim.mc_stats`): no ADAS 26 crashes;
brake-only 0 crashes / 85 goals; ADAS 0 / 93; ADAS + intent 0 / 94. Needless override time 185 s (old ADAS,
before this session's gate work) -> 101 s (ADAS) -> 80 s (ADAS + intent).

## Not verified on the car yet
Everything in the table above except the relay's basic LiDAR braking. Next car session: deploy (A1), then the user
drives and tries to fool it (I2).

## Known limitations
- The brake keeps a 1.3 safety factor on stopping distance with an assumed 1.2 m/s^2 deceleration; at full speed
  it stops ~35 cm short. Closer needs a measured braking deceleration (a calibration - only when asked).
- The intent model is trained on simulated drivers; it should be retrained on real drives (B15).
- One scan plane: obstacles above/below it are invisible; nothing closer than 20 cm to the LiDAR is measured (memory
  covers it for obstacles seen earlier).
- The web GUIs are being replaced by a native GUI for the 3D view (E3).

## Commands
```bash
python -m unittest discover -s tests -t .      # 58 tests, ~2 min, no car needed
python tools/sim_car.py --world doorway        # digital twin: then  python rc_controller.py --ip 127.0.0.1
python -m sim.relay_scenarios                  # 11 assist scenarios on the relay code
python -m sim.relay_mc 48 && python -m sim.mc_stats   # Monte Carlo + paired statistics
python -m sim.autonav_eval                     # click-to-go on 6 goals
python -m sim.twin_report                      # digital-twin accuracy vs a real drive log
```
