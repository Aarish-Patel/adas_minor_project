# Pi deployment plan and real-car test checklist (PREPARED, NOT RUN)

Nothing here is executed without the user's explicit OK, and only after the car is repaired (rear wheel shaft, 28 Sep). Backups and
benchmarks follow the rules in `docs/modules/integration.md` (benchmarks only in `~/rc_bench`, never `~/rc_car`).

## 1. Before touching the Pi (laptop)
1. `python -m unittest discover -s tests` and `python -m sim.relay_scenarios` green (30 Sep: 160/161 - `test_submaps` return-to-start is
   flaky, see TODO section Y; scenarios 21/21).
2. Commit; `python tools/publish.py` (dry run) to confirm what GitHub has versus what will be deployed.
3. Hardware check by the user: shaft repaired, ESP32 (WROOM-32D) powered from its own 5 V regulator or a PD supply that gives the Pi
   5 A (the 600 mA USB limit is the suspected killer of the first board), active cooler fitted (K7), motors OFF.

## 2. What changed since the 28 Sep 22:15 deploy (commit cfb7e55; the Pi still runs it)
| Area | Files | Effect on the car |
|---|---|---|
| Drivetrain protection | `pi/relay_assists.py`, `pi/path_gate.py` | closing speed only for tracked moving objects, brake pulse 80 PWM / 0.2 s, no latch release mid-pulse, reverse lockout above 0.12 m/s, smoother drops to zero on a direction change |
| Steering realism | `pi/relay_assists.py` | lock capped at 0.40 m radius; envelope shrinks with speed (38 deg up to 0.4 m/s ... 9.5 deg at 0.83 m/s) |
| World pose + zones | `pi/zones.py`, `adas/submaps.py`, relay | keyframed scan-matched pose, speed-limit zones (km/h scaled 1:14), submaps + loop closure for HOME |
| Moving obstacles | `adas/crossing.py`, `relay_assists.py` | yield / pass / swerve for tracked movers ('moving' assist) |
| Online adaptation | `adas/online_steering.py`, `online_calibration.py`, `latency.py` | live servo centre / gain (+-4.5 deg / +-25 %), risk recalibration, scan-latency estimate widens the brake margin |
| Speed control | `adas/speed_control.py`, `speed_ekf.py` | closed-loop speed on autonomous legs; uncertainty-aware brake margin |
| Modes / autonomy | `adas/explore.py`, `adas/park.py`, `adas/rss.py`, relay | drive modes, EXPLORE, PARK, HOME, RSS need in the state |
| Intent | `adas/conformal.py`, `pi/intent_v3.json`, `pi/needless_v1.json` (optional, only via `RC_NEED_PATH`) | v2 decides takeovers, v3 supplies the shown risk; conformal warning threshold |
| Rear camera | `adas/vision/*`, `pi/rear_camera.py` | optional separate process (:8091); no camera fitted -> nothing changes |
| Cleanup | 18 scripts moved to `pi/legacy/`, old reports and backups deleted | the Pi copy must not keep stale files: sync with delete, except the car's own json (section 4) |
| ESP32 firmware | `ESP32_RC/ESP32_RC.ino` | unchanged since the replacement board was flashed - do not re-flash |

## 3. Protocol additions the new GUI needs (UDP 4210, text)
The old Pi relay ignores or mishandles these, so the Pi must be updated before the Qt GUI (`python -m gui.dashboard --host <Pi>`) is used on it.
| Command | Meaning | Reply |
|---|---|---|
| `MODE eco\|normal\|sport` | drive mode (throttle rise rate, margins) | `OK` / `BAD MODE` |
| `HOME` | return to the world origin with the original heading (loop-closed pose) | `OK` |
| `EXPLORE ON\|OFF` | frontier exploration of the room | `OK` |
| `PARK [auto\|perpendicular\|parallel]` | back into the nearest bay / slot beside the car | result text |
| `ZONES <json>` | speed-limit zones (rect / circle / polygon, km/h full size), world frame | `OK <n>` / `BAD ZONES` |
| `ORIGIN` | world frame starts at the car now | `OK` |
| `GOTO x y [heading_deg]` / `GOTO CANCEL` | click-to-go; the arrival heading is new | `OK` |
| `GUI_SUBSCRIBE <port>` | stream subscription (already on the Pi) | `OK` |

Stream (JSON, ~20 Hz) keys the GUI reads: `world` {pose, zones, zone_kph, mode, rss_min_m, loops}, `health`, `esp32`, `assist`, `intent`
(risk v3 + trust), `nav`, `plan`, `gate`.

## 4. Deployment steps (each after the user's OK)
1. Reachability and hardware: `ssh pi@<host> 'ls /dev/serial/by-id; uptime; vcgencmd measure_temp'`.
2. Backup: `ssh pi@<host> 'tar czf ~/rc_car_backup_$(date +%Y%m%d_%H%M).tgz -C ~ rc_car'` (pattern as before, e.g.
   `rc_car_backup_20261002_1030.tgz`); write the exact name into `integration.md` and `STATUS.md`.
3. Keep the car's own files: `pi/tuning_real_car.json` (LiDAR yaw 45.7 deg), `cal_results.json`, `panel_params.json`. `pi/car_model.json`
   was copied in on 28 Sep - ask the user whether the car's copy or the repo copy stays; never overwrite silently.
4. Upload `adas/`, `pi/` (minus the keep list and `drive_logs*`), `requirements.txt`; delete files that are no longer in the repo from `pi/`.
5. Offline checks on the Pi, no motors: `python3 -c "import pi.wifi_drive_safety"`, `python3 -m unittest tests.test_esp_link tests.test_health`,
   then `tools/relay_latency.py` in `~/rc_bench` (28 Sep reference: 13 ms median / 20 ms 95th).
6. `sudo systemctl restart rc-relay` (LiDAR open ~1 min, issue A8); watch `~/relay.log`: LiDAR >= 6 Hz, ESP32 OK, health OK.
7. GUI connection test with NO motor commands: stream arrives, `PING` -> `PONG`, `MODE normal`, `ZONES []`, `ORIGIN`, `EXPLORE OFF`, `GOTO CANCEL`.
8. Rollback: `sudo systemctl stop rc-relay; rm -rf ~/rc_car; tar xzf ~/rc_car_backup_<name>.tgz -C ~; sudo systemctl start rc-relay`.
9. Only then the user switches the motors on; the first drive is checklist tests 0-2.

## 5. Real-car checklist (Met-S -> Met)
Rules: the user drives and places obstacles; no calibration unless the user asks; car on blocks for the first throttle test; stop at any odd
sound from the drivetrain. A test passes only if the acceptance number in `docs/requirements.md` is met on the car; record the drive-log
name, then set the status column to Met and add the numbers to `REPORT.md`.

| # | Test (on the car) | Closes | Pass criterion |
|---|---|---|---|
| 0 | Object dead ahead (LiDAR yaw), ESP32 PONG, straight line 2 m | FR-001, HW | object within +-2 deg; PONG; drift < 5 cm over 2 m |
| 1 | Forward drive, no obstacle, ADAS on, 3 laps | regression of the phantom-wall bug | no throttle cut, no false brake |
| 2 | Full throttle at a wall: brake-only, then ADAS | FR-007, FR-020 | stops without contact; FCW shown >= 0.5 s before the brake |
| 3 | Scan-latency estimate at rest and driving (`latency` in the state) | FR-007 | estimate stable and within 0.1 s of the expected ~0.2 s scan-to-motor delay |
| 4 | Box 0.5 m ahead at full throttle; doorway | FR-023 | 5/5 no contact, control handed back |
| 5 | Person / rolling object crossing at 0.3-0.5 m/s; a late crosser | FR-004, FR-024 | speed error < 0.1 m/s; no contact; no needless wait |
| 6 | Click-to-go, 6 goals (doorway, side, behind, 52 cm gap, reverse in corridor, pose goal) | FR-030, FR-034 | >= 5/6 within 8 cm / 20 deg, 0 contacts; speed RMSE <= 4 cm/s |
| 7 | Excursion >= 1.5 m, then HOME, three times | FR-006, FR-031 | ends <= 30 cm and <= 15 deg from the start |
| 8 | EXPLORE a bare room, then PARK (perpendicular, parallel) | FR-032, FR-033 | >= 90 % of the floor mapped, 0 contacts; parked <= 10 cm / 10 deg |
| 9 | Speed-limit zones of 10 and 20 km/h | speed-zone requirement in requirements.md | throttle capped, slows before entering |
| 10 | Normal driving with the risk display; a lapse case (stick frozen at a wall) | FR-011, FR-016 | no needless takeover in normal driving; warning before contact |
| 11 | Faults: cover / unplug the LiDAR, kill Wi-Fi, unplug the ESP32 USB | FR-046, health | motor stops within 0.5 s, FAULT shown, relay recovers by itself |
| 12 | Rear camera (only if fitted): guidelines, ghost car, object distance +-5 cm | FR-008 | as stated in requirements.md |
| 13 | After every session: `tools/pull_logs.py`, `tools/drivetrain_report.py` | drivetrain safety | no reverse pulses > 80 PWM, no reversals above 0.12 m/s |

Not part of this list: the braking-deceleration measurement (C9) - only when the user asks (calibration rule).
