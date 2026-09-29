# Machine Learning (driver intent)

## Purpose
Estimate whether the driver, left alone, will contact or nearly contact something within 2 s, so the ADAS can hold back needless
takeovers and show a meaningful risk. SRS: FR-011, FR-012, FR-016, FR-047, NFR-10.

## Responsibilities
| Function | What it does |
|---|---|
| Features | per tick (50 ms): stick, throttle, speed, free distance on five steering arcs forward + reverse, time to contact, free distance in stopping distances, driver's reaction distance; a 32-tick (1.6 s) window, flattened at fixed lags |
| Models | v2 (decides takeovers), v3 gradient-boosted trees (displayed risk, warnings); candidate: "takeover needed" classifier |
| Physics floor | risk ≥ 0.95 when the driver is inactive and the free way < stopping distance |
| Driver profile | online reaction distance of this driver |
| Online recalibration | Platt scaling tracked by a Kalman-filter logistic regression from observed safety events (display only) |
| Threshold selection | conformal risk control for the warning threshold |
| Data generation | twin drives with 15 randomised car parameters, seven driver styles, labels from continuing the drive without ADAS |
| Training | scikit-learn (trees), PyTorch + CUDA (MLP, GRU), temperature scaling, held-out and sliced evaluation, export to JSON |

## Inputs
Offline: twin drives (`data/intent/*.npz`, git-ignored), takeover decisions (`data/needless/*.npz`). Online: stick / throttle history,
speed, LiDAR points (via `RelayIntent`).

## Outputs
Model files `pi/intent_net.json` (v2), `pi/intent_v3.json` (v3), `pi/needless_v1.json` (candidate) · risk, trust, stalled, recalibrated
risk (IntentState) · reports `models/intent_v3_report.json`, `intent_ablation.json`, `conformal_intent.json`, `online_adapt.json`,
`intent_optimise.json`, `mc_bound.json`, `needless_report.json`.

## Dependencies
Car: NumPy only. Training: scikit-learn, PyTorch (GPU optional), the twin (monte_carlo.md). Consumer: safety_decision (`RelayIntent`).

## Public interfaces
| Interface | Contract |
|---|---|
| `IntentNet(path)` (`adas/intent_net.py`) | loads v2 / v3 / trees / MLP / GRU; `risk(window)`, `logit3(window)`, `crash_probability(f)` |
| `tick_vector`, `window_of`, `flat_features(_batch)`, `physics_floor(_batch)` | feature construction - identical on car and in training |
| `DriverProfile` | `update(servo_hist, free_now)`, `reaction_distance` |
| `OnlineCalibrator` (`adas/online_calibration.py`) | `observe(t, logit)`, `event(t)`, `probability(logit)`, `active` |
| `conformal_risk_threshold`, `late_warning_loss` (`adas/conformal.py`) | threshold with a stated late-warning bound |
| `RelayIntent(assist, path, trust, risk_path, need_path, need_tau)` | online use (see safety_decision.md) |
| Model JSON | `kind` (trees3 / mlp3 / gru3 / trees / regressor), `temperature`, weights or tree arrays |

Trees are fitted on float32 features; inference casts to float32 (tested).

## Workflow
1. `python -m sim.twin_intent_data` - generate randomised twin data (or from the training lab).
2. `python -m sim.train_intent_torch` - train all candidates on the GPU, pick by validation AP, write `models/intent_v3.json` + report.
3. `python -m sim.intent_ablation` - randomisation ablation; `python -m sim.conformal_intent`; `python -m sim.online_adapt_eval`.
4. Decider changes only after `python -m sim.intent_optimise confirm` / `sim.mc_bound` show a paired improvement on untouched seeds.
5. Takeover-needed model: `python -m sim.needless_data`, then `python -m sim.train_needless`, then compare with `RC_NEED_PATH`.

## Test strategy
Unit: `tests/test_intent_v3.py` (numpy = training framework, float32 parity, physics floor), `test_conformal.py`,
`test_online_calibration.py`. Evaluation on held-out drives and on unseen wider cars; decisions validated only in the Monte Carlo.

## Performance targets
| Metric | Target | Current (v3) |
|---|---|---|
| Held-out AP / AUC | ≥ 0.80 / ≥ 0.90 | 0.84-0.86 / 0.96 |
| Calibration error | ≤ 0.03 | 0.010 |
| AP on unseen wider cars | ≥ 0.80 | 0.85 |
| Inference on the Pi | ≤ 2 ms, NumPy only | met (laptop ≪ 1 ms) |
| Model size | ≤ 5 MB | ~1 MB |
| Online recalibration | ECE −20 % | −37 % nominal, −28 % unseen |

## Known findings
v3 predicts better but does not decide better than v2; an oracle intent does not reduce interventions (arbitration, not prediction,
is the limit) - `docs/decisions.md` ADR-010, ADR-024.

## Files
`adas/intent_net.py`, `adas/online_calibration.py`, `adas/conformal.py`, `RelayIntent` in `pi/relay_assists.py`,
`sim/twin_intent_data.py`, `sim/intent_data.py`, `sim/train_intent_torch.py`, `sim/train_intent_net.py`, `sim/intent_ablation.py`,
`sim/conformal_intent.py`, `sim/online_adapt_eval.py`, `sim/intent_optimise.py`, `sim/needless_data.py`, `sim/train_needless.py`,
model files in `pi/` and `models/`.
