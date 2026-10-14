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

## Takeover-needed model as decider (W5, tuning seeds 0-23, 72 drives)
`python -m sim.need_eval tune|confirm [tau...]` runs ADAS, v2 and `RC_NEED_PATH=pi/needless_v1.json` at each `RC_NEED_TAU` on the same
drives and prints paired one-sided Wilcoxon p-values vs v2 (`models/need_eval_tune*.json`). Tau 0.05-0.7: 0 crashes, takeovers 34-69
vs 30-31 for v2, goals 63-67 vs 65-66, no significant gain in needless interventions (best 92 vs 99, p = 0.43). Not adopted;
untouched-seed confirmation skipped because nothing beat v2 on the tuning seeds. Details in ml.md.

## Parking validation framework (`sim/parking_validation.py`, `sim/parking_dashboard.py`, `tests/test_parking_validation.py`)
Validates both Parking Simulator worlds (`park_perp`, `park_street`) without touching the GUI or the planners. The plan from
`adas/park_select.py` (`plan_perpendicular` from the ArUco marker seen by the simulated rear camera, `plan_parallel` from a click on the
strip) is injected into the relay's own click-to-go follower (`AutoNav`) and the whole relay stack drives the twin (`relay_scenarios.run`,
operator holding the throttle), so the brake gate, speed control and pure-pursuit tracking are the car's own. Nothing on the car consumes a
`ParkingPlan` yet, so the injection stands in for that integration.

| Item | Value |
|---|---|
| Commands | `python -m sim.parking_validation suite` (86 scenarios, nominal twin, seed 0) · `mc [runs] [workers]` (each scenario `runs` times, seeds 100+) · `python -m sim.parking_dashboard` |
| Outputs | `reports/parking_validation/`: `results.csv` (one row per run), `summary.csv` (per scenario), `dashboard.html`, `plots/*.png`, `results_{suite,mc}.csv`, `summary_*.json`, `traces_*.json` |
| Perpendicular (24) | perfect marker; marker rotated 5 / 10 / 20 deg; marker hidden in 25 / 50 / 75 % of frames; box on the line of sight (must refuse) and beside the path; marker 1.2-3.0 m away; LiDAR range noise 1 / 2 / 4 cm; localisation drift (2 cm/s + 0.5 deg/s, 5 cm/s + 1.5 deg/s) injected into the follower's pose; start offset +-25 cm / 8 deg; the rear camera as configured (pitch 32 deg); the follower's own goal re-planning left on |
| Parallel (62) | 24 random layouts (4-7 spaces: narrow 0.55-0.85 m, standard 0.95-1.3 m, wide 1.5-2.0 m, parked vehicles 0.30-0.46 m long, random occupancy; click on / between spaces / on an occupied one / beyond the row) each from the shipped start (70 cm from the road edge) and from 90 cm; named cases: narrow / standard / wide, too short and too shallow (must refuse), all occupied (must refuse), edge first / last (a wall right behind the row), click between spaces, click on an occupied one |
| Monte Carlo noise | `twin_intent_data.randomise` level 1 (steering gain x0.85-1.15, servo offset +-3 deg, motor lag, command delay, LiDAR noise 4-20 mm, dropout 0-12 %, yaw error +-2 deg, jitter), extra range noise 0-1.5 cm, camera pixel noise 0.3-1.2 px, start pose jitter (perp +-12 cm / +-6 deg, parallel +-15 cm / +-6 deg), neighbour vehicle lengths |
| Success | plan made and driven to a stop, 0 contacts, pose error <= 10 cm, heading error <= 10 deg, car inside the space (parallel); cases that must be refused succeed when the system refuses |
| Ground truth | from the world geometry, not the planner's goal: perpendicular = rear axle on the bay centre line, 6 cm off the wall, heading along the bay; parallel = body centred in the selected space, heading along the strip |
| Metrics per run | success, final position error (cm; lateral and along), heading error (deg), corrections (extra gear changes + re-plans + straightening strokes), collisions (0 / 1: the twin stops at the first contact), time from plan to stop, path length, min clearance, selected vs expected space |

Results (Monte Carlo, 860 runs = 86 scenarios x 10 seeds, 8 workers, 15 min; nominal suite of 86 alongside; no thresholds tuned):
- **0 contacts in 860 runs** (and in the 86 nominal ones).
- Perpendicular: 210 runs expected to park, 197 parked (13 not planned: marker not detected or no path), all inside the bay. Median error 8.9 cm: **lateral 1.3 cm
  (p95 4.3 cm)**, along the bay 8.7 cm - the follower stops within its 8 cm arrival tolerance of the goal, so the along-bay error sits at the 10 cm criterion;
  heading median 4.7 deg (p95 13). 116 of 197 within 10 cm and 10 deg, 164 within 15 cm. The perfect-marker case is 10/10 (6.2 cm, 1.8 deg).
  Marker rotation goes into the final heading (the plan aims along the marker), 20 deg rotation -> 12 deg error on the nominal run, 7.8 deg median in the Monte Carlo.
  LiDAR noise of 2-4 cm and a start offset lower the success (3-6 / 10), frame drop-outs to 75 % still park 4/10, localisation drift alone 7/10.
- Parallel: 590 runs expected to park, 532 parked (58 had no plan), 493 of them inside the space; median error 6.3 cm (lateral 2.1 cm), heading median 4.5 deg
  (p95 7.6); 342 of 532 within 10 cm and 10 deg, 409 within 15 cm; p95 position error 42 cm (a few layouts end outside the space: layouts 05, 10, 18, 22).
  Selection matched the nearest free fitting space in 374 of 620 runs. Standard space: 10/10 (3.4 cm, 4.7 deg, 12 s, 3.5 m).

Findings for the owners of the parking code (evidence in the CSV, nothing changed here):
1. The `park_perp` marker (z 9 cm, camera z 12 cm, pitch 32 deg) is above the rear camera's image at every distance: 0/10 detections
   (`perp_camera_as_configured`). The marker scenarios use the camera with pitch 8 deg (`level_rear_camera`); with it a 10 cm marker is found up to
   ~2.4 m (2.8 m and beyond are refused, 10/10, but at 2.4 m only 2/10 park within 10 cm).
2. `AutoNav`'s own re-planning fights an injected parking path: its clearance check (goal-based planner margin) calls the 6 cm stop gap "blocked"
   and stops the car at 0.4 m ("path blocked - stopped", 25 runs; `perp_follower_replan` 2/10 vs 6/10). The framework switches that check off for
   injected plans (`follower_replan=False`); a real integration has to do the same or use a follower that takes the parking plan as is.
3. Parallel planning depends on the lateral start position: from the shipped start (70 cm from the road edge) wide spaces and spaces at the end of a
   row often get no plan (`par_width_wide` 5/10, `par_edge_last*` 0/10 even 90 cm out, spaces 0.55-0.85 m with no plan at all - the multi-point
   search does not find a way in for the 0.7 m case). Contacts stay at 0 because the planner refuses rather than tries.
4. Follower arrival tolerance (8 cm, 8 deg straighten) leaves 4-9 cm along-bay / along-space error; tighter arrival would shift most "position error"
   failures to successes.
Limits: the ego car is the fixed 1:14 twin (vehicle length varies through the parked vehicles and space lengths, not the ego body); occupied spaces come from the
layout (ground truth), not from LiDAR occupancy detection; planning time is not added to the simulated latency; drift is injected into the follower's pose
estimate, which the scan matcher partly corrects.
