# Architecture

The system is layered like a production ADAS stack (perception -> prediction -> planning -> control -> HMI), with the same
code running on the Pi 5 (`pi/`) and in the laptop digital twin (`sim/`). `adas/` has no hardware imports.

```
 LiDAR (RPLIDAR, 5-10 Hz)       driver (Xbox pad / keyboard, UDP to the relay)       rear webcam (optional, MJPEG :8091)
        |                                   |                                                   |
        v                                   v                                                   v
+------------------- pi/wifi_drive_safety.py  (the relay: one loop per driver packet, ~50 Hz) ---------------------+
|  PERCEPTION      pi/relay_assists.RelaySpeed   speed / yaw / world pose: throttle model + LiDAR range flow (EKF),  |
|                  adas/latency.py               scan-matched pose; online scan-latency estimate                     |
|                  pi/path_gate.ObstacleMemory   blind-ring memory;  adas/tracking.py  moving-object tracks           |
|  PREDICTION      pi/path_gate.PathGate         swept path at the current steering, free distance along it           |
|                  pi/relay_assists.RelayIntent  driver-intent risk (v2 decides takeovers, v3 supplies risk/warnings) |
|  PLANNING        adas/assists.DrivingAssists   evasive steer (Hybrid A* via adas/plan_service.py, MPPI fallback)    |
|                  adas/autonav.py               click-to-go / return-to-start;  adas/crossing.py  moving obstacles   |
|  CONTROL         pi/relay_assists.RelayAssists nudge, centring, limiter, narrow gap, proximity, zones, drive modes,  |
|                                                realistic steering envelope, online steering calibration            |
|                  pi/path_gate.PathGate.decide  last word: brake / hold / cap by stopping distance (never suppressed)|
|                  pi/relay_assists.ThrottleSmoother  rate limit, reverse lockout (safety cuts bypass)                |
|  SUPERVISION     pi/health.py (normal / limp / fault),  pi/esp_link.py (PING, reboot detection),  ESP32 failsafe    |
+--------------------------------------------------+-------------------------------------------------------------------+
                                                   | commands "A <servo> <servo>", "M <pwm>" (serial)
                                                   v
                                          ESP32 (ESP32_RC/ESP32_RC.ino): servo + motor PWM, 0.5 s command failsafe

 state + logs:  pi/drive_log (JSONL), GUI state on UDP :4210 (relay -> GUI) and HTTP :8090, control panel :8080
 HMI (laptop):  gui/dashboard.py -> gui/ev.py (EV-style window, drawers) + gui/controls.py + gui/theme.py (dark/light)
                gui/lab_windows.py + gui/lab_tabs.py (Monte Carlo and ML training labs)
```

## Rules the layers keep
1. **The driver has the wheel.** Every assist is opt-in or minimal; the order of escalation is nudge -> speed limit -> evasive
   manoeuvre -> brake. The brake gate is pure physics and is never switched off by the learned model.
2. **The learned model may only hold interventions back, never cause them.** Intent decides whether the evasive steer takes
   over; its trust is withdrawn when the car is stuck (`RelayIntent._progress`).
3. **Safety cuts bypass comfort filters.** The throttle smoother never delays a brake, hold or health cut.
4. **One code path for car and simulation.** `sim/relay_scenarios.py`, `sim/relay_mc.py` and `tools/replay_log.py` instantiate the
   same `RelayAssists`, `PathGate`, `RelaySpeed` and `RelayIntent` as the relay.
5. **Faults degrade, not crash.** Health monitor -> limp (reduced speed) -> fault (stop); ESP32 link is supervised; the twin
   injects dropouts, frozen/late scans, command loss and spurious points (`sim/relay_scenarios.py` fault scenarios).

## Data and model files
| file | what | used by |
|---|---|---|
| `pi/tuning_real_car.json` | mount, servo, speed model (calibrations, only when asked) | relay, twin |
| `pi/car_model.json` | fitted motor lag, delay, decel | speed EKF, twin |
| `pi/intent_net.json` | v2 intent model (decides takeovers) | RelayIntent |
| `pi/intent_v3.json` | v3 trees (risk shown, warnings) | RelayIntent |
| `models/*.json` | evaluation results (Monte Carlo, ablation, repeatability, conformal, latency) | reports, labs |

## Where to look
`docs/REPORT.md` (methods, problems, results), `RESEARCH.md` (algorithm choices with references), `TODO.md` (open work),
`STATUS.md` (what is on the car and what is not), `README.md` (commands).
