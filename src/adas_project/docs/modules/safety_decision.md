# Safety and Decision (prediction, gate, assists, arbitration)

## Purpose
Decide, every control cycle, what command reaches the motor and servo: the driver's own, a softened one, a steering correction, a
manoeuvre, or a brake - the least intrusive response that keeps the car out of contact. Owns the safety argument. SRS: FR-010,
FR-012-015, FR-020-027, NFR-15-16.

## Responsibilities
| Function | Rule |
|---|---|
| Collision prediction | swept path at the current steering (plus recent steering for the command delay), free distance, time to contact, stopping distance `BASE + v·REACTION + v²/(2·DECEL)` × FOS |
| Path gate (last word) | brake / hold / cap throttle when free distance < stopping distance; hold released only when the driver lets go or the path clears; margins widened by scan latency excess and speed uncertainty (95th percentile) |
| Warning stages | collision warning (≤ 1.6 s to contact) -> speed limited -> emergency brake; streamed to the HMI |
| Assists (opt-in) | evasive steer, steering correction (nudge), corridor centring, speed/steer limiter, narrow gap, side/rear proximity, moving obstacles (pass / yield / stop / away) |
| Always on | realistic speed-dependent steering envelope, speed-limit zones, drive modes (Eco cap 55 %), throttle smoother + reverse lockout |
| Intent use | the intent model may hold back a takeover (trusted driver); trust withdrawn when stuck; never touches the gate |
| Commitment | a manoeuvre is not dropped for < 0.9 s of released throttle or < 0.35 s of counter-steer; after a real override, stay back 6 s unless the brake envelope is reached; a held car is freed after 2 s grace |
| RSS | formal longitudinal safe distance, displayed |

Order inside one packet: steering envelope -> nudge -> `RelayAssists.process` (autonomy / evasive / assists / crossing / zones / mode)
-> path gate -> throttle smoother (safety cuts bypass) -> health limits.

## Inputs
Driver command (servo °, PWM) · perception outputs (ObstacleSet, FreeSpace, EgoState, LatencyEstimate, TrackList) · IntentState
(risk, trusted, attentive, stalled, k-rate) · planner Paths · zones, drive mode, assist switches from the HMI · HealthState.

## Outputs
Rewritten command lines (`A <servo> <servo>`, `M <pwm>`) · GateInfo (action, free distance, contact point) · assist info / level
· warnings · planning requests.

## Dependencies
perception (data), planning (paths via PlanService), ml (intent models, loaded by `RelayIntent`). Used by integration (relay) and
monte_carlo (twin) - same classes.

## Public interfaces
| Interface | Contract |
|---|---|
| `PathGate` (`pi/path_gate.py`) | `on_scan(points_vehicle, seq)`; `decide(dt, physical, delta, v_est, closing, intent_k_rate, trusted, leg)` -> (physical, braking); `free_distance(delta, direction)`; `allowed_speed(free, fos)`; `nudge(...)`; attributes `info`, `memory`, `delay_source`, `uncertainty_source` |
| `RelayAssists` (`pi/relay_assists.py`) | `process(lines, points, seq, now)` -> lines; `set(name, on)`; `set_mode(name)`; `goto / explore / park`; `limit_servo(servo, v)`; `planned_leg()`; `set_tracks(tracks)`; `info`, `level`, `crossing` |
| `DrivingAssists` (`adas/assists.py`) | `update(dt, pts, stick, pwm, v)` -> (stick, pwm, level); phases PLANNING / EXECUTE / WAIT / BACKOFF; intent fields `intent_hold`, `intent_attentive`, `intent_k_rate`, `intent_stalled`, `intent_commit`; `trigger_reason` |
| `RelayIntent` (`pi/relay_assists.py`) | `update(now, servo, physical, v, points)`; `gate_trust`, `gate_k_rate`, `trusted`, `stalled`; `feedback_event(now)`; `gui()` |
| `ThrottleSmoother` | `step(target, dt, emergency, v)` |
| `adas/crossing.py` `decide(...)` | path-velocity decision for moving objects |
| `adas/rss.py` | `longitudinal_min_distance(v_rear, v_front)`, `lateral_min_distance`, `margin` |
| `pi/zones.py` `SpeedZones` | zones in the world frame, `limit_ahead(pose, v)` (full-size km/h) |

## Test strategy
- Unit: `tests/test_planning_safety.py` (gate, latch, memory, evasive state machine, stuck recovery), `test_crossing.py`,
  `test_zones.py`, `test_modes_rss_home.py`, `test_uncertainty_margin.py`, `test_intent_v3.py` (physics floor, float32 parity).
- Scenario suite (must stay all-pass): `python -m sim.relay_scenarios` - walls, boxes, doorway, corridor, nudge, narrow gaps,
  moving obstacles, fault injection.
- Repeatability on randomised cars: `python -m sim.repeat_scenarios`.
- Needless-intervention statistics: see monte_carlo.md.

## Performance targets
| Metric | Target | Current |
|---|---|---|
| Contacts in scenario suite / Monte Carlo | 0 | 0 / 0 |
| Stop position from full speed | 2-8 cm | 4-7 cm |
| Safe rate on randomised cars | 100 % | 100 % (420 runs) |
| Gate decision time | ≪ one packet (20 ms) | met |
| Needless interventions vs plain ADAS | significant reduction | -41 % (p = 9e-5) |
| Nudge success on randomised cars | ≥ 80 % | 55 % (known limitation) |

## Invariants (do not break)
1. The gate runs last and cannot be disabled except by the explicit ADAS override.
2. Learned models only withhold takeovers; they never suppress the brake.
3. Safety cuts bypass the throttle smoother.
4. The relay, twin and Monte Carlo use these exact classes.

## Files
`pi/path_gate.py`, `pi/relay_assists.py` (RelayAssists, RelayIntent, ThrottleSmoother, DRIVE_MODES, home_goal), `pi/zones.py`,
`pi/path_predict.py`, `adas/assists.py`, `adas/crossing.py`, `adas/rss.py`, `adas/aeb.py` (speed model), `adas/servo.py`,
`adas/config.py`. Older first-simulator logic (`adas/pipeline.py`, `acc.py`, `isa.py`, `lane.py`, `parking.py`, `warning.py`,
`adaptive.py`, `intent.py`) is not on the car's relay path except `acc.py` (follow the leader).

## Key parameters (values in code; change only with evidence from the twin AND a note here)
| Parameter | Value | Meaning |
|---|---|---|
| `FOS` / `TRUSTED_FOS_FLOOR` | 1.3 / 1.15 | stopping distance must fit this many times into the free distance (trusted attentive driver: 1.15) |
| `BASE_M` | 0.05 m | standoff kept at walking pace |
| `REACTION_S` | 0.20 s | scan + relay + motor delay budget (fitted command delay 0.12 s + scan period) |
| `DECEL` | 4.0 m/s² | deceleration on a throttle cut (the real car stops almost instantly: 1-2 cm drift) |
| `BODY_MARGIN_M`, `MARGIN_LEVELS` | 0.03 m; ≥0.15 m/s +1.2 cm, ≥0.40 m/s +2 cm | speed-dependent protective field |
| `K_WINDOW_S`, `KAPPA_ERR_ABS/REL` | 0.25 s; 0.08 1/m + 15 % | composite swept path: any steering of the last 0.25 s, widened by steering-model error |
| `HORIZON_M` | 1.6 m | prediction horizon |
| `CREEP_V`, `CREEP_MIN_M` | 0.10 m/s, 0.06 m | creep allowed while more than 6 cm is free (parking, nosing up) |
| `BRAKE_OVER_V`, `BRAKE_GAIN`, `BRAKE_PWM_MAX`, `BRAKE_MAX_S` | 0.15 m/s, 300 PWM per m/s, 80, 0.2 s | active reverse pulse; small because a cut alone stops the car |
| `LATCH_RELEASE_M` | 0.08 m | hold releases once free distance grows this much (and only after the pulse, standing, path clear) |
| Closing speed | only for tracks flagged moving | noise on static objects caused brake hammering (real log 22:11) |
| `SCAN_LOST_S` | 0.5 s | no new scan -> no throttle |
| Latency budget / cap | nominal 0.16 s, window 1.4 s, excess ≤ 0.4 s | online delay estimator |
| Uncertainty | σ_nominal 0.03 m/s, z = 1.645 | speed planned at v + 1.645·max(0, σ − 0.03) |
| Nudge | κ steps 0.1…0.7 1/m, min 0.2 m/s, TTC 1.0 s, must gain ≥ 0.5 m free | steering correction |
| Evasive trigger | TTC 1.2 s (attentive 0.7 s), v ≥ 0.18 m/s, confirm 0.15 s, contact margin 3 cm, last-point-to-steer + 0.25 m | `AssistConfig` in `adas/assists.py` |
| Evasive execution | speed ≤ 0.40 m/s, reverse 0.15 m/s, back-off 0.30 m (≤ 3 tries), stall re-plan after 0.6 s (≤ 3), retry pause 1.5 s, max 15 s / 2.5 m past | |
| Driver override | stick > 0.45 at once (without commitment); with commitment counter-steer 0.35 s or throttle released 0.9 s | |
| Commitment / respect / stuck | respect 6 s after override; held car freed after 2 s; stall = < 0.8 m in 5 s with throttle > 40 PWM, trust withdrawn 10 s | `RelayIntent._progress`, `STALL_*` |
| Moving objects | tracked ≥ 5 scans, 0.15-2.0 m/s; swerve held 1.4 s; arcs ±1.5, ±1.0, ±0.5 1/m away from the object first | |
| Steering envelope | `KAPPA_REAL` 2.5, `LAT_ACCEL_REAL` 6.0 m/s² full-size (≈0.6 g) at 1:14; servo travel 57° right / 53° left | |
| Throttle smoother | rise 600 PWM/s (Eco 320, Sport 950), fall 1500 PWM/s, direction change -> 0 at once, reverse lockout above 0.12 m/s | drivetrain protection after the rear shaft broke |
| Zones | full-size km/h / 3.6 / 14 = car m/s; look-ahead 0.5 s; lowest limit wins | 50 km/h -> 0.99 m/s |
| Intent trust | P(crash) < 0.5 (v2) | `INTENT_TRUST`, `RC_TRUST` |

The relay (`pi/wifi_drive_safety.py`) also contains an older cone-based front/rear stop (`CONE_DEG`, `REACTION_TIME_S`, ...) kept as a
fallback path; the path gate is the primary safety function.
