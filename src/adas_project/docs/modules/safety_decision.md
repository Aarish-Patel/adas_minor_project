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
