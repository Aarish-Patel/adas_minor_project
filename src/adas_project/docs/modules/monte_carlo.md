# Simulation and Validation (digital twin, scenarios, Monte Carlo)

## Purpose
Prove the ADAS works - and how often it intervenes needlessly - without risking the car: run the car's own relay classes on a
digital twin, across fixed scenarios, randomised cars, injected faults and thousands of randomised drives. SRS: FR-043-046,
NFR-18-19, all success metrics.

## Responsibilities
| Function | What it does |
|---|---|
| Vehicle model | kinematic Ackermann car with fitted motor lag, command delay, deceleration, steering gain (`sim/fitted_car.json`, from real logs) |
| LiDAR model | ray casting, 720 beams, range noise, dropout, dark (no-return) objects, latency |
| Worlds | doorway, room, corridor, gap, open, lounge, parking, parallel, random Monte Carlo rooms, worlds from real logs |
| Drivers | scripted and stochastic drivers: lapsing, late, good, distracted, aggressive (+ more in data generation) |
| Scenario suite | 21 pass/fail cases with specifications, incl. moving obstacles and fault injection |
| Repeatability | every scenario on N randomised cars: safe % and spec % |
| Fault injection | LiDAR dropout, frozen scans, late scans, 30 % command loss, spurious points |
| Monte Carlo | driver-only, brake-only, ADAS, ADAS + intent, oracle on identical drives; counterfactual judge (same driver, no ADAS, 2 s, 2 cm) |
| Statistics | paired one-sided Wilcoxon, tune seeds vs untouched confirmation seeds, episodes and progress assists |
| Log replay | real recorded scans and commands through the current code |

## Inputs
Relay classes (safety_decision, perception, planning), intent models (ml), fitted car model, seeds, driver styles, environment
variables for experiments (`RC_TRUST`, `RC_INTENT_PATH`, `RC_NEED_PATH`, `RC_STALL`, `RC_STUCKWAIT`, `RC_RESPECT`, `RC_DEFER`, `RC_COMMIT`).

## Outputs
`models/relay_scenarios.json`, `repeatability.json`, `mc_live/` (runs.jsonl + status.json for the lab), `mc_bound.json`,
`intent_optimise.json`, per-evaluation JSON and `reports/*.png`.

## Dependencies
NumPy, SciPy, matplotlib; runs the car's classes directly. The lab GUI (gui.md) reads `models/mc_live/`.

## Public interfaces
| Interface | Contract |
|---|---|
| `relay_scenarios.run(world, driver, seconds, assists, start, seed, stop_when, hook, fault, cmd_loss)` | one drive on the relay code; returns trace, collided, min clearance, infos, latency |
| `relay_mc.run((seed, variant, style))` | one Monte Carlo drive; returns crashed, reached, interventions, false_positives (needless), episodes, progress, events, burden, trace |
| `relay_mc.counterfactual(car, driver, t0)` | "crash" / "near-miss" / None |
| `relay_mc --live <rooms> <variants> <styles> [model]` | streaming run for the lab |
| `mc_bound.main(n_seeds, first_seed)` | all systems + oracle, avoidable-intervention bound, paired tests |
| `hw_sim.VirtualCar`, `SimLidar`, `hw_worlds.WORLDS` | twin building blocks |

Drives are identified by (seed, style); variants on the same key share room, driver and lapses.

## Test strategy
- `tests/test_planning_safety.py`, `test_project.py`, `test_run_all.py` smoke-test the twin; scenario and evaluation scripts print
  pass/fail and write JSON.
- Rules: tune on seeds 0-23, confirm on untouched seeds (24+); report sample sizes and p-values; differences of a few drives are noise
  (planner compute time enters the simulated Pi delay).

## Performance targets
| Metric | Target | Current |
|---|---|---|
| Scenario suite | all pass | 21/21 |
| Repeatability safe rate | 100 % | 100 % of 420 |
| Monte Carlo crashes (any ADAS) | 0 | 0 |
| Twin vs real log path error | ≤ 5 cm | see `models/twin_report.json` |
| Monte Carlo throughput | 72 drives × 4 systems in ≤ 45 min on 8 workers | ~40 min |

## Commands
`python -m sim.relay_scenarios` · `python -m sim.repeat_scenarios` · `python -m sim.relay_mc --live 24 off,brake-only,adas,adas+intent
lapsing,aggressive,late` · `python -m sim.mc_bound 24 24` · `python -m sim.intent_optimise confirm` · `python tools/replay_log.py <log>`
· `python tools/sim_car.py --world room` (twin relay for the GUI).

## Files
Current twin and validation: `sim/hw_sim.py`, `sim/hw_worlds.py`, `sim/world.py`, `sim/fitted_car.json`, `sim/relay_scenarios.py`,
`sim/repeat_scenarios.py`, `sim/relay_mc.py`, `sim/mc_bound.py`, `sim/mc_stats.py`, `sim/intent_optimise.py`, `sim/*_eval.py`,
`sim/twin_report.py`, `sim/log_fit.py`, `sim/log_synth.py`, `sim/real_car.py`, `sim/profile_relay.py`, `tools/sim_car.py`,
`tools/replay_log.py`, `tools/scripted_driver.py`. First simulator (browser viewer, kept for its scenarios): `sim/simulator.py`,
`sim/car_sim.py`, `sim/lidar_sim.py`, `sim/scenarios.py`, `sim/run_scenarios.py`, `sim/monte_carlo.py`, `sim/session.py`,
`sim/viewer.py`, `server.py`, `web/`.

## Methodology details
| Item | Value |
|---|---|
| Time step / scan period | 0.05 s control, 0.1 s LiDAR (720 beams, 8 mm noise, 4 % dropout) |
| Drive length | ≤ 30 s; goal reached within 0.35 m |
| Counterfactual | fork the simulation at the intervention onset, same driver (incl. lapses) continues 2 s with no ADAS; crash or clearance < 2 cm = needed |
| Intervention kinds | `evasive`, `steer` (nudge / centring), `gate:limited`, `gate:braking`, `gate:holding`, `progress` (freeing a held car) |
| Episode | interventions < 2.5 s apart |
| Burden | seconds overridden, seconds the wheel was taken, share of throttle removed - needless vs needed |
| Driver styles (Monte Carlo) | lapsing (lapses every ~5 s for 1-2 s), late (reacts at 0.7-0.9 m, decisive), good (avoids early), distracted (lapses every ~3 s for 1.5-3 s), aggressive (PWM 200-250, reacts at 0.8-1.0 m); all stop and re-steer at 0.22 m |
| Oracle | intent that knows the counterfactual every 0.2 s - the upper bound for any intent model |
| Seeds | 0-23 tuning, 24-47 confirmation; decision data 100-499 |
| Non-determinism | planner compute time enters the simulated Pi delay: re-runs differ by a few drives; use ≥ 72 paired drives |
| Randomisation for repeatability | same 15 parameters as the ML data (level 1 = full ranges); contrast checks against "no assist" skipped under randomisation |

## Latest results (untouched seeds 24-47, 72 drives)
driver alone 6 crashes / 60 goals · brake only 0 / 59 · ADAS 0 / 70, 132 interventions, 87 episodes, 127 needless · ADAS + intent 0 / 70,
83 interventions (−37 %), 50 episodes (−43 %), 75 needless (−41 %), median time 11.6 vs 12.1 s · oracle 0 / 62, 131 interventions.
A 90 % reduction is not reachable in this scenario set (remaining interventions free cars the driver cannot free).
