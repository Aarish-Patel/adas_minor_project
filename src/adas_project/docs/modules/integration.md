# Integration (car runtime, hardware, communication, deployment)

## Purpose
Run the ADAS on the car: connect sensors, driver, HMI and actuators; supervise hardware; degrade safely; record everything; deploy and
operate the Pi services. SRS: FR-041, FR-042, FR-048, NFR-08, NFR-13-14, NFR-21, HW-01-09.

## Responsibilities
| Function | What it does |
|---|---|
| Relay main loop | receive driver packets (UDP 4210), read the newest scan, run perception -> intent -> `RelayAssists.process` -> path gate -> smoother -> health limits, send to the ESP32, publish state |
| LiDAR acquisition | native streamer (`pi/lidar_stream/`) or Python driver; newest-scan reader with sequence numbers |
| ESP32 link | supervised serial: steer / motor frames, PING, reboot banner detection, automatic re-open |
| ESP32 firmware | servo and ESC PWM; motor off if no valid command for 500 ms |
| Health monitor | normal / limp / fault from LiDAR rate, link rate, loop time, ESP32 state, temperature |
| Operator commands | assists, modes, zones, origin, goto / home / explore / park, follow, override, subscribe |
| State publishing | JSON state to UDP subscribers (≥ 20 Hz) and HTTP :8090 |
| Logging | per-packet / per-scan drive logs (JSONL, compressed) + events |
| Control panel | HTTP :8080: start drive / calibration / demo jobs, show and apply calibration results (only on confirmation) |
| Services | systemd units `rc-relay.service`, `rc-panel.service` |
| Driver client | `rc_controller.py`: gamepad / keyboard -> UDP (or serial for bench tests) |

## Inputs
LiDAR USB stream · driver UDP packets · operator UDP commands · HTTP requests · tuning files (`pi/tuning_real_car.json`,
`pi/car_model.json`) · models (`pi/intent_net.json`, `pi/intent_v3.json`).

## Outputs
Serial frames to the ESP32 · StateMessage (UDP / HTTP) · drive logs on the Pi (`~/rc_car/logs/…`) · calibration results.

## Dependencies
All other subsystems (their classes are instantiated here). Python: NumPy, SciPy, pyserial; `rplidar-roboticia` on the Pi.

## Public interfaces
| Interface | Contract |
|---|---|
| UDP 4210 (in) | driver lines `A <servo> <servo>` / `M <pwm>` and operator text commands (see gui.md) -> "OK" / message |
| UDP push (out) | StateMessage keys: `mode`, `drive`, `gate`, `plan`, `nav`, `assist`, `intent`, `world` (pose, zones, zone_kph, mode, rss_min_m, loops), `health`, `esp32`, points |
| HTTP :8090 / :8080 / :8091 | state + web dashboard / control panel API (`/api/state`, `/api/start`, `/api/stop`, `/api/apply`) / rear camera (`/stream`, `/state`) |
| Serial (ESP32) | text frames for steer and motor, PING / PONG; 500 ms failsafe |
| `EspLink` (`pi/esp_link.py`) | `send`, `status()`, `healthy()` |
| `HealthMonitor` (`pi/health.py`) | rates and temperature in -> `state`, `causes`, limits |
| Drive log (`pi/drive_log.py`) | records + `event(name, **fields)`; read by `tools/replay_log.py`, `tools/stream_log.py`, training |

## Operating procedures
- Deploy: copy the repository to the Pi (`~/rc_car`), restart `rc-relay` / `rc-panel` - **only with the user's OK**; back up first
  (`~/rc_car_backup_<date>.tgz`). Benchmarks run in `~/rc_bench`, never in `~/rc_car`.
- Calibrations only when the user asks; results applied only via "Apply".
- Pull logs: `tools/pull_logs.py`; analyse / replay: `tools/analyze_drive_log.py`, `tools/replay_log.py`.
- Laptop twin in place of the car: `python tools/sim_car.py --world room` (same ports, so the GUI and `rc_controller.py` work unchanged).

## Test strategy
Unit: `tests/test_esp_link.py`, `test_health.py`, `test_logging.py`, `test_run_all.py`, `test_real_profile.py`. Integration on the twin
through the same relay classes (monte_carlo.md). Real-car checklist: LiDAR centred, ESP32 PONG, straight line, turns, gate stop at a
wall, evasive, click-to-go - before new features are trusted (SRS status Met-S -> Met).

## Performance targets
| Metric | Target | Current |
|---|---|---|
| Control loop | ≥ 50 Hz, 95th percentile loop time ≤ 20 ms | ~50 Hz packets |
| Scan-to-command latency | ≤ 200 ms | ~120 ms command delay + scan |
| ESP32 failsafe | ≤ 500 ms | 500 ms |
| Relay start | ≤ 20 s | ~60 s (slow LiDAR USB open, open issue A8) |
| Pi temperature | < 80 °C sustained | 74 °C seen - cooler required |
| Relay memory | ≤ 500 MB | not measured |

## Files
`pi/wifi_drive_safety.py` (relay main), `pi/esp_link.py`, `pi/health.py`, `pi/drive_log.py`, `pi/log_drive.py`, `pi/control_panel.py`,
`pi/control_panel.html`, `pi/calibrate.py`, `pi/lidar_front_cal.py`, `pi/lidar_steering_diag.py`, `pi/center_*.py`, `pi/speed_run.py`,
`pi/turn_test.py`, `pi/brake_run.py`, `pi/passthrough.py`, `pi/estop.py`, `pi/rear_camera.py`, `pi/lidar_stream/`, `pi/*.service`,
`pi/devices.py`, `pi/link.py`, `pi/runtime.py`, `pi/main.py`, `pi/hil.py`, `pi/run_all.py`; `ESP32_RC/ESP32_RC.ino`; `rc_controller.py`;
`tools/pull_logs.py`, `tools/stream_log.py`, `tools/analyze_drive_log.py`, `tools/relay_latency.py`, `tools/pi_parts_bench.py`,
`tools/drivetrain_report.py`. Early experiments: `pi/legacy/` (not used).
