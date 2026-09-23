# Project status (RC-car ADAS: Raspberry Pi 5 + RPLIDAR A3M1 + ESP32 car)

Last updated after the final-grip recalibration, the obstacle-bypass work and the control panel.

## Where things are

| Piece | Where | State |
|---|---|---|
| Pi | `192.168.1.6` (DHCP, has changed once: give it a static IP), user `pi` | reachable |
| ESP32 | `192.168.1.7` | reachable |
| Safety relay (drive with LiDAR stops) | `pi/wifi_drive_safety.py`, systemd `rc-relay` | deployed; stopped whenever a test/panel job owns the LiDAR |
| Control panel (buttons) | `pi/control_panel.py` + `.html`, systemd `rc-panel`, **http://192.168.1.6:8080/** | deployed; only the LiDAR-calibration button has been exercised so far |
| Live LiDAR view | port 8090 (relay GUI, or the running script's `pi/test_gui.py`) | works |
| Simulator + 3D viewer | `python server.py` -> http://localhost:8765 | works; new: Bypass (B) and real-car profile (K) |
| Laptop controller | `rc_controller.py` (ESP32_IP = the Pi relay) | works |

Only ONE process may own the LiDAR/ESP32 at a time (relay, a test script, or the panel's job).

## Current calibration (real car, final wheel grips) - `pi/tuning_real_car.json`

| Item | Value | Source |
|---|---|---|
| LiDAR yaw offset | 90.0 deg (object dead-centre in front) | `lidar_front_cal.py` / earlier refine runs |
| Servo straight-ahead | 87.0 (fit 86.9, t = 17.7, 20/20 arcs) | `center_fine.py` |
| Steering gain | 0.068 rad/m of path curvature per servo degree (about 24 deg -> radius 0.6 m) | `turn_test.py`, old centre - re-measure |
| Speed model | v_max 0.896 m/s, deadband 48.5 PWM (0.27 m/s at PWM 115) | sustained LiDAR runs |
| Stopping | rolled 0-7 cm after cutting throttle at 0.25-0.36 m/s | `brake_run.py`; faster speeds NOT measured |
| Body extents from the LiDAR | front 0.16, rear 0.17, left/right 0.10 m | ruler |

## What works (verified on the car)
- Obstacle bypass (`pi/bypass_run.py`, controller `pi/bypass_core.py`, scan-matching odometry `pi/scanmatch.py`):
  detects the obstacle, picks the more open side (0.87 m vs 0.35 m), passes it, and starts rejoining the line.
  Speeds and steering are ramped (PWM step 8/tick, servo slew 60 deg/s). **It has never finished the return on the car**:
  the room only gives ~2.7 m and the independent front stop (0.28 m) ended both runs, 0.26 m off the line at +26 deg.
  Needs ~3.5 m of clear runway. Nothing has been touched.
- LiDAR-based safety relay (see the git log M1-M17), calibration tools, live GUI.

## What works (simulator only)
- Everything in `README.md` (14/14 self-test scenarios, Monte Carlo, faults, parking, lane keeping, follow, signs).
- **Real-car profile** (`sim/real_car.py`): the simulator configured from the measurements above. All 14 scenarios pass with it,
  also at +/-25 % speed error. Stops about 10 cm short of a wall at any speed.
- **Bypass in the simulator** (`sim/bypass_driver.py`, `sim/bypass_eval.py`): 60 random obstacle/wall/gap cases x both car profiles:
  60/60 correct, 0 crashes, median final lateral error 0.5 cm, median clearance 6.7 cm (real profile), 10 correct refusals.

## Changed since the last real run - NOT yet on the Pi (deploy + verify at the next car session)
- `pi/scanmatch.py`: the scan-to-start-scan correction now also accepts weaker matches (residual < 0.055, inliers >= 55) but only
  applies y and heading from them. In simulation this fixed odometry drift in the return phase (lateral error 7-16 cm -> under 2 cm).
  It is untested on real scans. The old strict behaviour is the `strict` branch in `Odometry.update`.

## Known limitations
- An obstacle within ~16 cm of a wall is clustered with the wall (region-grow radius) and may be refused although the other side is open.
- Steering gain is only measured to ~24 servo degrees; full lock (radius ~0.26 m) is an extrapolation in the simulator.
- Stopping distance above 0.36 m/s and turn gain at the new servo centre are not measured.
- The creep-into-obstacle fix (M16) and moving-object tracking / follow mode (M16/M17) have never run on the real car.
- No webcam: ArUco parking, traffic-sign ISA and lane keeping are simulator-only.

## Test commands
```bash
python -m unittest discover -s tests -t .      # 23 tests, ~90 s, no car needed
python -m sim.selftest                         # the 14 scenarios, 6 s
python -m sim.run_scenarios --real             # the same scenarios with the real-car profile
python -m sim.bypass_eval 60                   # bypass Monte Carlo, both profiles
python -m sim.report                           # rebuild the Results tab (use --run to re-run everything: minutes)
python -m pi.sim_bypass                        # standalone bypass simulator (ray cast + real controller)
```
On the car: open the control panel (http://192.168.1.6:8080/) and use the buttons. Suggested order: LiDAR front, speed + stopping,
turning, then Drive with safety stops, then the obstacle-avoidance demo (with ~3.5 m runway).

## Next steps
1. Car: run the three calibrations from the panel and press Apply on each result.
2. Deploy `pi/scanmatch.py` + `pi/bypass_core.py` + `pi/bypass_run.py` (already on the Pi are the older versions) and rerun the bypass with runway.
3. Car: automated pass/fail braking and failsafe tests (stopping distance at 2 speeds, stop time after a WiFi cut).
4. Intent-aware warning and follow-the-leader on the car (needs the printed cones and a leader body).
5. Webcam for parking; results slides from `reports/index.html` and the Results tab.
