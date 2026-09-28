# Task list

Every task, asked for or noticed, is written here first and checked off when done. Unfinished work keeps its
current state in the note. Algorithm choices and paper citations are in `RESEARCH.md`.

## A. Blocking the next car session
- [ ] A1. **Deploy to the Pi.** The Pi still runs the version whose obstacle memory made a phantom wall and blocked
  forward driving. Deploy `adas/` and `pi/`, restart `rc-relay`, check a forward drive first.
  (28 Sep: user OK'd the upload, motors off for now. Steps: back up ~/rc_car, keep the car's tuning_real_car.json /
  cal_results.json / panel_params.json / car_model.json, upload, import + unit checks on the Pi, restart the
  relay, check LiDAR/ESP32 link/GUI stream with no motor commands; then the user turns the motors on.)
  Done: backup ~/rc_car_backup_20260928_1927.tgz, new adas/ + pi/ + car_model.json + intent_net.json copied in,
  the car's json files kept, imports OK; offline probe on the Pi (virtual car/LiDAR, ~/rc_bench): 12.7 ms median,
  20.3 ms 95th, 0 crashes, evasive went round the doorway box. Starting the relay from here was refused by the
  permission check - the user starts Drive from the panel.
- [ ] A2. Real-car tests requested by the user (28 Sep, motors on, obstacle in front):
  - [x] LiDAR centring test: the panel script failed (USB ports missing before the Pi restarts); measured from
    the live stream instead: object 0.48 m from the LiDAR spans -9.8..+11.6 deg, centre ~+1 deg -> the 63.4 deg
    offset is still right within ~1 deg (if the object was dead centre). Nothing changed.
  - Pi WiFi flaky (20% ping loss, 1 s spikes, SSH timeouts); USB LiDAR/ESP32 vanished once until a reboot
    (brown-out when the motors were switched on?). ESP32 did not answer a relay PING within 200 ms.
  - "Motor not responding": the relay passed the stick through (PWM out up to 183, servo 36-135), but the ESP32
    never answered PING. The user restarted the ESP32 -> its USB re-enumerated (19:40:24-34), leaving the relay on
    a dead handle; the relay must be restarted (panel Drive) to reopen it. Kernel also logs USB timeouts (-110) on
    both CP2102s -> power/cable. Idea: the relay should detect a lost ESP32 (PING/serial errors) and reopen it.
    After the relay restart: still no reply. RTS reset + 3-min serial listener while the user reset the ESP32:
    zero bytes (not even the ROM boot text); no ESP32 answers a UDP PING broadcast on the home WiFi either.
    Power LED on. -> ESP32 not running / serial path dead, not a software issue. Next: RC_CAR hotspot check,
    ESP32 on the laptop's USB, motor battery off vs on (brown-out).
    Remote RTS/DTR resets + the user's RST button: still zero bytes. ESP32 USB keeps dropping (19:40, 19:45,
    19:48, 19:49, then gone). The Pi's supply negotiated no USB-PD (usbpd objects all 0) -> the Pi 5 limits all
    USB ports to 600 mA total, shared by the LiDAR motor and the ESP32 (+ servos if fed from its 5 V pin).
    Likely cause: USB power budget / cable. User options: power the ESP32 + servos from the car battery via a
    5 V regulator (USB for data only, common ground), a 5 A USB-PD supply for the Pi (or
    usb_max_current_enable=1 only if the supply can really deliver), reseat/replace the ESP32 USB cable.
    ESP32 moved to the laptop (COM7): the USB-serial chip enumerates, but zero bytes on reset, no PONG, and
    esptool chip-id says "No serial data received" (the ROM bootloader is silent). -> the ESP32 chip / its 3.3 V
    supply looks dead, not the Pi. Next (user): BOOT+EN manual download mode and chip-id again; measure 3V3 pin;
    check what killed it before fitting a spare (servo/motor power on the ESP32 pins, VIN voltage, back-EMF).
    Real-car tests A2 wait for a working ESP32.
    BOOT+EN manual download mode: esptool still "No serial data received"; on the laptop's USB power the ESP32 is
    not on WiFi either (no PONG to a broadcast, no RC_CAR hotspot). -> ESP32 chip (or its 3.3 V regulator) dead.
    The user needs a replacement ESP32 (flash ESP32_RC/ESP32_RC.ino) and to fix what killed it first.
  - [x] Hybrid A* click-to-go into open space (28 Sep 21:3x, new ESP32, LiDAR 45.7 deg: a straight goal drove
    correctly; a goal 0.44 m ahead / 2.09 m left with the object 0.5 m in front started with a reverse-right leg -
    the planner's three-point turn, since a forward-left arc clips the object)
  - [ ] Evasive steer at the obstacle
  - [ ] Hybrid A* click-to-go to a goal behind/across the obstacle

## L. User requests (28 Sep, third round) - ADAS parked at "acceptable" (11/11 scenarios, 0 crashes), tune later
- [x] L1. (done: tried "back up then re-plan" (ROS recovery style) - 7x slower, reverted; instead weighted-A*
  inflation 3.0, 0.25 m steps, 5 primitives for searches with reversing. On the Pi, 31 recorded back-off jobs:
  median 463 -> 205 ms, 95th 1352 -> 294 ms, max 1807 -> 347 ms, same 26 found) (= K6) Faster back-off searches.
- [x] L2. (done, commit 515af39: "nudge" assist (opt-in, it tripled needless interventions in the MC when on by
  default) + MPPI fallback in the evasive WAIT phase; 12/12 scenarios, 64 tests pass) Steering correction instead of
  braking where a nearby steering is safe, plus a sampling-based fallback (MPPI) when Hybrid A* finds nothing.
- [x] L3. (done, commit 515af39: gui/dashboard.py, PySide6 + pyqtgraph, UDP stream ~20 Hz) Native Qt GUI replacing
  the web pages: modes, map, "what might happen", twin charts.
- [x] L4. Told the user how to run the 3D simulator and test everything on the GUI.
- [ ] L5. Live relay, doorway world, PWM 200 + evasive: the evasive phase flickers PLANNING <-> None every tick while
  the gate holds at pose ~[4.5, -0.18], then "no safe way around - braking only". Check whether the car had already
  passed the door (far wall) or failed to swerve at the box. LOW PRIORITY: the user says "no safe way around" when
  the wall is too close is fine for now.

## O. User request (28 Sep, evening): "work on the simulator, the TODO list and everything for the project"
(the ESP32 died, so no car until it is replaced). Order: M4 research -> M1 data + model -> M3 training tab ->
M2 Monte Carlo tab -> O1 -> N items that can be built and tested in the simulator.
- [x] O1. (done: pi/esp_link.py + tests/test_esp_link.py, 5 tests) Relay: detect a lost / silent ESP32 and reopen
  the port by itself. Background PING every 1 s (no serial read in the control loop any more - the UDP PING handler
  used to stall the loop 50-250 ms), "silent" after 3 s without PONG, boot banner "ROVER READY" counted as a reboot
  (brown-out sign), write errors -> "lost", reopened by /dev/serial/by-id name or any non-LiDAR ttyUSB; the relay now
  starts without an ESP32 (LiDAR + GUI only) instead of exiting, and exits 1 (systemd retries) if the LiDAR is missing.
  Dashboard top bar shows an ESP32 chip (OK / CONNECTING / SILENT / LOST, reboots). Not yet on the Pi.
- [x] O2. CUDA PyTorch: the user approved; `pip install torch --index-url .../cu128` running (was CPU-only 2.11).
- [x] O4. (done: restored with git checkout after the data job) MISTAKE TO UNDO: the new twin data generator was first written over the existing sim/intent_data.py (the
  older laptop-simulator intent trainer; sim/assist_eval, bypass_eval, demo_tests, export_slides, faults, lane_eval
  import MODEL_DIR from it). It now lives in sim/twin_intent_data.py; restore the original with
  `git checkout -- sim/intent_data.py` as soon as the running data job (started under the old name) has finished.
- [ ] O3. The laptop was on battery during this session (CPU capped at 2.4 GHz, workers throttled): data generation
  and GPU training are much slower unplugged - plug in for long jobs.

## M. User requests (28 Sep, fourth round) - started 28 Sep evening (see O)
- [ ] M1. **The intent model predicts badly in general** (user: full speed into a wall is only the example they
  saw - do NOT just patch that case). Audit where it fails across all situations (per scenario, speed, distance,
  driver style, turning vs straight, time before contact), then retrain / replace the intent model:
  - Train on the laptop's NVIDIA GPU (CUDA, e.g. PyTorch) for many epochs, aiming for high accuracy (report AUC,
    average precision, calibration/Brier, and the specific "full throttle at a wall" case as a test).
  - Must stay light enough for the Pi 5 at the relay's 20 Hz without processing delays: small MLP / 1D-CNN / GRU
    over the stick + scan history, exported to numpy weights or ONNX; benchmark on the Pi in `~/rc_bench` (K2 rule).
  - Broader, harder data: many worlds/scenarios, speeds, distances, turning and straight, frozen stick, late
    braking, near misses; hard-example mining from wherever the audit shows errors.
  - Evaluate per slice (not just one overall AUC) plus calibration, so a good average can't hide bad situations.
  - Consider a physics prior as an input/floor: TTC and free distance in stopping distances already exist; a
    learned model should never give low risk when time-to-contact is below the stopping time.
  - Regression tests over a scenario suite (full speed at a wall is one of many): P(crash) high well before the
    brake point on real threats, low on safe driving.
  Progress (28 Sep evening): v3 pipeline written - adas/intent_net.py tick_vector / flat_features / physics_floor
  + IntentNet kinds trees3 / mlp3 / gru3; sim/twin_intent_data.py (twin data with domain randomisation, 6 situation
  families, 7 room styles, every tick labelled); sim/train_intent_torch.py (GPU, trees vs MLP vs GRU, early
  stopping, temperature scaling, per-slice report vs v2, run-level warning lead time); pi/relay_assists.RelayIntent
  runs v3 when pi/intent_v3.json exists. Found causes of the bad v2 predictions: trained only within 1.2 m of an
  obstacle and on drivers who slow down, while the relay asks it every tick; features saturate at 1.5 m although a
  full-speed car covers ~1.7 m in the 2 s label horizon; forward only; one fixed car and clean sensor.
  Data: 3000 training drives (543k ticks) + 600 test drives (109k ticks), 15.4 % positive ticks, ~25 min on battery.
  Quick GPU check (few epochs), held-out test, v2 -> v3: AP 0.42 -> 0.85, AUC 0.73 -> 0.96, recall at 0.5
  0.07 -> 0.74 (v2 almost never fired), false alarms 0.7 % -> 3.6 % of safe ticks; fast > 0.5 m/s AP 0.34 -> 0.76,
  free way 1.2-2.5 m AP 0.21 -> 0.83, reversing 0.57 -> 0.89; the physics floor costs a little AP (0.85 -> 0.81)
  but cuts missed crash drives 7 % -> 3 %; crash drives missed 54 % -> 3 %,
  median warning 0 s -> 1.2 s before contact, but safe drives with a >= 0.25 s false alarm 4 % -> 16 % (watch in
  the Monte Carlo). Bug found + fixed: trees fitted on float32 features must see float32 on the car (14/400 ticks
  took the other branch).
  Full GPU run: trees3 chosen (val AP 0.862; MLPs 0.847-0.851, GRUs 0.845-0.846 - data-limited, early stopping at
  ~30 epochs), held-out AP 0.42 -> 0.86, AUC 0.73 -> 0.96, calibration error 0.12 -> 0.004, numpy exact,
  0.22 ms/tick laptop. Monte Carlo (models/mc_intent_v2_v3.json, 96 paired drives, lapsing + late): 0 crashes for
  all; needless takeovers ADAS 53 -> v2 25 / v3 26; overridden needlessly 128 -> 65 (v2) / 86 s (v3); trust sweep
  0.7 / 0.85 did not help v3 (84-91 s). -> DECISION: v2 keeps deciding takeovers (v2 was trained on exactly these
  Monte Carlo drivers), v3 (pi/intent_v3.json) supplies the P(crash) the driver sees and the warnings
  (RelayIntent.risk_net; GUI shows both). 83 tests + 12/12 scenarios pass.
  Next: retrain the takeover decision itself on a mixed set (v3 data + Monte Carlo drivers) or learn the decision
  (would the intervention be needless?) directly; weak slices: turning AP 0.65, false alarms at 0-0.5 m (19 %) and
  reversing (10 %); Pi benchmark in ~/rc_bench; randomisation ablation DONE (unseen 1.6x-wider cars: AP 0.63 nominal-trained vs 0.85 randomised, calibration error 0.12 vs 0.018, RESEARCH.md section 7).
- [x] M2. (done 28 Sep: dashboard tab "Monte Carlo lab", gui/lab_tabs.py + sim/relay_mc.py --live; checked end to end) **Monte Carlo visualiser in the simulator GUI** (`gui/dashboard.py`, new tab): pick Driver only / ADAS /
  ADAS + intent (and driver style, world, number of runs), run `sim/relay_mc.py` in a background process, show
  live results: trajectories on the map, crashes, needless interventions, burden, paired stats (Wilcoxon), and a
  replay of any single run.
- [x] M3. (done 28 Sep: dashboard tab "ML training lab" - identified twin, randomisation table, generate / train buttons, live loss + AP curves per model, calibration, risk over time on held-out drives, per-situation v2 vs v3; the real-log sim-to-real panel waits for labelled real drives, B15) **ML training visualisation in the simulator GUI** (new tab): start training from the GUI, live loss /
  AUC / precision-recall curves per epoch, confusion matrix, risk-vs-time on example drives (e.g. full throttle
  into a wall), compare model versions. **It must use the digital-twin training methods from M4** (user request):
  show the domain-randomisation settings (noise, latency, braking, vibration jitter), the sim data vs real-log
  data mix, sim-to-real gap metrics (accuracy on twin data vs on real car logs), and the real-to-sim calibration
  step, so the whole twin-based training loop is visible.
- [x] M4. (done 28 Sep: RESEARCH.md section 7 - methods table with references, how the project applies each, randomisation ablation result; the real-to-sim loop step waits for real drives, B15) **Research digital-twin-based ML training** (for the report/teachers; add to `RESEARCH.md` with
  citations): sim-to-real transfer, domain randomisation (Tobin et al. 2017), system identification + twin
  calibration from real logs, real-to-sim-to-real loops, synthetic data for driver-intent/risk models, and how the
  project's twin (hw_sim + measured braking + twin_report accuracy) fits. Then apply it: train on randomised twin
  data, validate on real car logs (ties into B15).

## N. Proposed software improvements (28 Sep, suggested to the user - not agreed yet; ask before starting)
Consistency on a vibrating car:
- [ ] N1. Closed-loop speed control (PI + feedforward from pi/car_model.json on the RF2O/EKF speed) instead of
  open-loop PWM, like an EV's torque/speed controller; jerk-limited commands.
- [ ] N2. Online adaptation of the car model: recursive least squares with a forgetting factor for speed gain,
  braking decel and steering curvature, so battery sag / floor / vibration changes are tracked during a drive.
- [ ] N3. Adaptive noise in the speed EKF (innovation-based adaptive estimation, Mehra 1970) + robust (Huber)
  weights in ICP/RF2O so vibration-induced outliers don't jerk the estimate; LiDAR motion deskew (LOAM, Zhang &
  Singh 2014) and a temporal scan filter / log-odds occupancy grid.
- [ ] N4. Uncertainty-aware safety: margins scaled by the measured spread (e.g. braking distance 95th percentile),
  calibrated intent probabilities (temperature scaling, Guo et al. 2017), conformal prediction bounds (Angelopoulos
  & Bates 2021) or a small deep ensemble (Lakshminarayanan et al. 2017).
- [ ] N5. Repeatability protocol like Euro NCAP AEB tests: every car test repeated N times, report mean +- std and
  pass rate, not single runs; twin noise randomised to the measured spread (ties into M4).
Real-EV style software:
- [~] N6. (dashboard part done 28 Sep: COLLISION WARNING chip + beep below 1.6 s to contact, then "PARTIAL BRAKE -
  SPEED LIMITED", then EMERGENCY BRAKE; still to do: buzzer/LED on the car) Staged AEB as in Euro NCAP / UN R152: forward-collision warning -> partial brake -> full brake, with
  warnings shown in the GUI (and a buzzer/LED if available).
- [ ] N7. Responsibility-Sensitive Safety (Shalev-Shwartz et al. 2017) as a formal safe-distance rule to cite
  alongside the gate/CBF.
- [x] N8. (done 28 Sep: pi/health.py + tests/test_health.py; the relay caps the throttle at 120 PWM in LIMP (LiDAR
  < 6 Hz, loop 95th > 60 ms, driver link < 12 packets/s while driving, Pi >= 80 C) and holds the motor in FAULT (no
  scans for 1 s, ESP32 silent/lost), 2 s hysteresis; dashboard chip HEALTH OK / LIMP MODE / FAULT. On the saturated
  laptop it correctly went LIMP "control loop slow (95th 152 ms)". Not yet on the Pi.) Functional-safety style supervision (ISO 26262 ideas): health monitor for LiDAR rate, loop latency, link,
  CPU temperature; degraded modes (limp mode = speed capped, ADAS off -> warn) with a state machine shown in the GUI.
- [ ] N9. Fault-injection tests in the twin (LiDAR dropout/freeze, latency spikes, packet loss, stuck throttle) as
  SOTIF (ISO 21448) scenario testing; ASAM OpenSCENARIO-like scenario files; CI running scenarios + tests.
- [ ] N10. Driver-facing EV features: adaptive cruise / follow distance setting, speed-limit zones on the map,
  park assist with distance bars, drive modes (Eco/Normal/Sport = throttle maps + margins), trip/energy log.
- [ ] N11. Architecture: ROS 2-style layering (perception / prediction / planning / control / HMI) with logged,
  replayable message streams (rosbag-like), A/B deploy with rollback on the Pi (OTA-style).

## K. User requests (28 Sep, second round)
- [x] K1. (done: the 28 Sep relay drive log confirms it - after the 0.12 s command delay a throttle cut rolls ~6 cm
  from 0.7 m/s (~4 m/s^2), an active brake pulse 1-3 cm (~8 m/s^2); the twin's 2 m/s^2 was an unfitted default.
  Twin, speed EKF, gate DECEL 1.2 -> 4.0 and the swerve timing updated; full speed at a wall stops 25 cm short)
  **Braking model = the real car**: the car stops almost instantly (user: 1-2 cm drift).
- [x] K2. (done, measured ON the Pi 5 in ~/rc_bench with `tools/relay_latency.py` (the unchanged relay against the
  virtual car, 20 Hz driver, evasive swerve at full throttle): input-to-motor 13 ms median / 20 ms 95th / 22 ms max
  (was 28 / 70 / 83), motor command every ~50 ms (worst gap 67 ms, was 125). Fixes: path searches in a worker
  process (`adas/plan_service.py`, started at launch, shut down on exit); k-d tree ICP in `pi/scanmatch.py` (the
  70 ms spikes); speed EKF + intent after the motor write; vectorised LiDAR thread; memory prune vectorised;
  RF2O 3 rounds; switch interval 1 ms; Hybrid A* compiled Dijkstra + early exit + vectorised primitives + 15 deg
  bins for back-off searches + time budgets; latency-compensated plan start; last point to steer from the turning
  geometry. Stage timing in the relay with RC_TIMING=1. Pi numbers: `python3 -m sim.profile_relay 2 --on-pi`,
  `tools/plan_bench.py`) **Everything must run on the Pi 5 without processing delays**.
- [x] K6. (done as L1: 0.21 s median / 0.35 s max on the Pi) Back-off (reversing) searches were 0.46 s median /
  1.8 s max on the Pi.
- [ ] K7. The Pi reached 74 C under sustained load (throttles at 80-85 C): fit the official active cooler.
  **Everything must run on the Pi 5 without processing delays**: profile every per-tick piece (gate
  sweeps, RF2O/EKF, intent MLP, evasive Hybrid A*, click-to-go), set a budget, move anything slow off the control
  loop (worker thread) or make it cheaper; benchmark on the Pi itself when it is reachable.
- [ ] K3. (in progress: gradient-boosted trees in numpy (AUC 0.86, avg precision 0.48 vs MLP 0.38), throttle
  history + time-to-contact + stopping-distance features, near-miss labels, 5 driver styles; trusted active
  drivers get a predicted-path soft cap; frozen-stick hold limited by the last point to steer. Latest 120 paired
  drives with the Pi's planning delay, 0 crashes: needless takeovers ADAS 66 -> ADAS+intent 31, overridden
  needlessly 168 -> 94 s (p = 0.0002). Open: the correct (earlier) last point to steer cost intent some of its
  takeover reduction (was 35 -> 5 with swerves that were too late to work on the Pi); aggressive drivers still
  ~3 margin interventions per drive (K5); retrain on real drives (B15))
  **Much smarter, more accurate intent-aware ADAS**: interventions down to a very small number, 0 crashes.
- [ ] K4. (in progress: 5 driver styles - lapsing, late, good, distracted, aggressive; drivers now slow down and
  stop like people (before, they only steered at constant throttle and crashed head-on even without lapses);
  ground truth counts near misses (< 2 cm) as needed. 120 paired drives: 0 crashes with any ADAS; ADAS+intent vs
  ADAS: needless takeovers 110 -> 19, override 185 -> 66 s (p < 0.0001). Still to add: corridors, clutter,
  moving obstacles) **More, different scenarios** in the Monte Carlo - keep trying.
- [ ] K5. Aggressive drivers (full throttle to within a few cm of walls) still get ~3 gate interventions per drive
  from the brake's margin (5 cm standoff, FOS 1.3); only shrink it with more braking data from the car.

## B. Safety and planning (the car must be impossible to crash, with minimal false interruptions)
- [x] B1. **Research** (done: RESEARCH.md sections 1-2) planning/safety algorithms and pick them (RESEARCH.md): Hybrid A* for the evasive path,
  a minimally-invasive safety filter (control barrier functions) for braking/steering corrections, fallback
  manoeuvres.
- [x] B2. (done, awaiting the user's sim drive: Hybrid A* `adas/hybrid_astar.py`, pure pursuit, speed cap 0.40 m/s, reverse legs, wait-and-retry keeping the original line; doorway at full throttle passes through the relay, no crash) **Evasive steer gets stuck in the doorway scenario, especially at full throttle** (user report). Replace the
  offset lattice with a Hybrid A* (or equivalent researched) planner over the LiDAR occupancy grid, aimed at the
  driver's desired path; re-plan quickly when execution gets stuck; speed limited to what the plan can do.
- [x] B3. (done: 40 cm box now swerves, 13 cm clearance) Wide obstacles trigger too late (40 cm box -> brakes instead of swerving). Plan earlier when the needed
  offset is large. (Likely solved by B2.)
- [x] B4. (no longer reproduces: `sim/relay_scenarios.py` doorway at full throttle on the relay code - through
  the door, 6 cm closest, 0 ticks of throttle cut while evading; kept as a regression scenario) Brief throttle cut
  mid-manoeuvre in the doorway run (t = 3.6 s).
- [ ] B5. (12 -> 9 needless of 44 by triggering evasive steer only on real contact courses; narrowing the gate's steering-slop band made it worse (13), reverted. Remaining: 5 gate speed-limits, 4 evasive)  Measure false-positive interruptions: count brakes/limits/swerves in normal driving (sim Monte Carlo +
  real logs) and tune down.
- [ ] B16. **Minimise speed cuts and needless interventions until only extreme, unpredictable driving gets one**
  (user request). Normal and reasonably sloppy driving must never be slowed, braked or steered. Target in the
  Monte Carlo (counterfactual ground truth): needless speed limits and needless takeovers close to 0 with crashes
  still 0; then confirm on real logs. Was: 53 needless speed limits and 7 needless takeovers in 48 drives.
  Step 1 done: the gate sweeps only the steering the car can be on before the next decision (the commands of the
  last 0.3 s + a model error of 0.08 1/m + 15 %, instead of a fixed +-0.35 1/m band) and uses a speed-dependent
  protective field (1.2 cm margin up to 0.15 m/s, 2 cm up to 0.4, 3 cm above, as AGV scanners do). 48 drives,
  crashes still 0: brake-only interventions 188 -> 88 (needless limits 117 -> 37, needless brakes 6 -> 0);
  ADAS 99 -> 54 (limits 43 -> 7, brakes 5 -> 0, takeovers 17 -> 12); ADAS+intent 93 -> 48 (limits 53 -> 14,
  brakes 6 -> 0, takeovers 7), 48/48 reach the goal. 32 cm gap now passes untouched.
  Step 2 done: composite sweep (earlier command only for the command-delay distance, then the current one);
  swerve trigger 1.6 -> 1.2 s (0.9 -> 0.7 s attentive); Monte Carlo now measures the burden (seconds overridden,
  wheel taken, throttle removed) and where each needless intervention happened (free / stopping distance),
  `python -m sim.mc_stats`. 96 drives, 0 crashes: ADAS needless takeovers 34 -> 12, overridden needlessly
  185 s -> 81 s (73 s with intent). Tried and rejected: later soft cap for trusted drivers (more brakes).
  Remaining: needless limits are all inside 1.3x the stopping distance (late swerves) - the next lever is a
  better speed estimate (C5) so FOS 1.3 can come down safely, or steering-based avoidance at the last point to
  steer (B8/B9).
- [x] B6. (done: `python -m sim.relay_scenarios` on the relay code incl. the relay's intent model, 11/11 pass -
  evasive box, doorway full throttle, driver already avoiding, full speed at a wall, centring, centring yields,
  limiter, narrow won't fit, 32 cm gap fits (passes untouched after B16), proximity, no slowing beside a wall)
  Corridor centring, limiter, narrow gap and side/rear alerts re-checked since the path gate replaced the cone gate.
- [x] B7. (done: `VP` = measured body 20 cm wide / front 0.28 / rear -0.05 from the tuning file, steering = the
  fitted 0.0656 rad/m per servo degree about the calibrated centre, the same as the path gate; the old turn-radius
  table is only a fallback) `pi/path_predict.py` `VP` still had the old body (14 cm wide).

- [x] B8. (done as L2: nudge steering correction; earlier partly: the gate is now a least-restrictive filter over the reachable steering with a speed-dependent
  protective field - B16; still to do: steering correction instead of braking when a nearby arc is safe, i.e. the
  QP over (steering, throttle)) CBF safety filter (RESEARCH.md 2) replacing the heuristic speed caps in `pi/path_gate.py`.
- [x] B9. (done as L2: adas/mppi.py in the evasive WAIT phase) MPPI local fallback when Hybrid A* has no path or execution deviates.
- [x] B10. (done inside B2: the planner adds reverse legs when no forward path exists) "Unstuck" mode with reversing (Reeds-Shepp) - the user's option 1.

- [x] B11. (checked on the relay code with the fitted speed model: full throttle at a wall now stops 23 cm short,
  was 0.75 m; regression scenario in `sim/relay_scenarios.py`) Brake hold stopped 0.75 m short of a wall at full speed.
- [x] B13. (fixed) Car froze next to a box: points already inside the body margin blocked every direction, even
  backing away. The path gate now ignores close points the motion moves away from (same margin box as the sweep).
- [x] B12. (done in J1) Unit tests for the Hybrid A* planner and the evasive state machine (doorway, boxed, reverse case).

## C. Localisation and car model
- [x] C1. (done: research RESEARCH.md section 3; RF2O + EKF implemented and measured on the twin and the real log,
  see C5) **Research accurate 2D LiDAR odometry** (linear and angular velocity from the LiDAR was poor).
- [x] C2. (done: held-out 3 s error physics 5.5 cm vs plain fit 5.7 cm vs physics+ML 5.8 cm - ML correction not useful on one 2-min drive, so the selection keeps whichever wins; found a steering asymmetry: left 0.0151 vs right 0.0113 rad per servo degree. More varied logs needed for the ML part) Physics-informed ML car model (`adas/car_model.py`, `sim/car_model_eval.py`, uncommitted). First version lost
  to the plain fit (15-19 cm vs 5.7 cm). Rewritten, NOT re-run. Commit only if it beats the plain fit.
- [ ] C3. (speed model done: `pi/car_model.json` from the logging drive, loaded by the relay and the Monte Carlo via `apply_car_model`; min clearance 4 -> 6 cm. Still to do: curvature/servo centre in the relay, the ML model when C2 passes) Use the model everywhere: relay speed estimate, gate speed caps, curvature for prediction/planning,
  the simulator's virtual car. (Fixes the relay's speed-model mismatch.)
- [ ] C4. Start-of-drive calibration run (panel): drive the legs, fit on the Pi, save `pi/car_model.json`, show the
  fit. Runs only when the user asks.

- [x] C5. (done: `adas/rf2o.py` (Jaimez 2016), `adas/speed_ekf.py`, in the relay as `RelaySpeed` - braking uses
  the more conservative of throttle model and EKF; `python -m sim.odometry_eval`: real drive EKF 3.4 cm/s RMSE vs
  throttle model 3.5 (moving), twin with the car 20 % off its model 3.1 vs 10.3; yaw rate 2.1 deg/s. Found: the
  throttle model thinks the car stops at once while braking. MC with it: 0 crashes, needless override 87 s ADAS /
  65 s ADAS+intent) RF2O range-flow odometry for speed and yaw rate; EKF fusing it with the car model.
- [x] C8. (done: `RelayAssists.speed = RelaySpeed` in the relay, MC and scenarios) The assists' own speed should
  also come from `RelaySpeed`.
- [x] B17. (done) Intent hold limited by the last point to steer for a driver not moving the stick; attentive swerve
  trigger only with an active stick. 11/11 scenarios; MC 96 drives: takeovers 14 -> 8 (p = 0.007), overridden
  needlessly 101 -> 80 s (p = 0.030), 0 crashes.
- [x] E3 (done as L3, PySide6 + pyqtgraph GL + UDP stream) next session: native Qt (PySide6 + Qt Quick 3D, the usual automotive HMI toolkit) GUI fed by a UDP/ZeroMQ
  push stream from the relay instead of HTTP polling.
- [ ] C9. Measure the braking deceleration on the car (only when the user asks - calibration) so DECEL/FOS can be
  tightened: full speed currently stops ~35 cm short of a wall.
- [ ] C7. (LOW PRIORITY - premise disproved: the -3.7 cm twin bias was NOT scan skew; it was the same with the car
  standing still (-3.7) and moving (-3.6), and worst on grazing beams. It was a map artefact, fixed in D3b.
  Skew is not measurable at the logged speeds; it only matters near full speed, ~8 cm per rotation at 0.8 m/s.)
  Scan de-skewing (KISS-ICP style) in the odometry and the simulated LiDAR.
- [x] D3b. Twin LiDAR map built properly: log-odds occupancy with free-space carving (Moravec & Elfes; Probabilistic
  Robotics ch. 9) instead of "cells hit 3 times". Median |sim - real| 3.7 -> 1.1 cm, bias -3.7 -> -0.6 cm, 86% of
  9928 beams within 5 cm, bias now uniform in every direction.
- [ ] D7. (dashboard part done: Reports tab; slides still to do) Show the twin report figures in the dashboard (Diagnostics mode) and as slides.
- [ ] C6. Room map (Cartographer-style submaps) for localisation, point-to-point navigation, return-to-start.

- [x] B14. (done: relay samples the stick on a 50 ms clock, features from live scans, P(crash) < 0.5 -> no evasive takeover; dashboard 'Driver intent' card with risk bar, trust state, learned reaction distance) Put the learned intent model + driver profile into the relay on the car (features from live scans,
  `pi/intent_net.json`), shown on the dashboard (risk gauge, "driver is avoiding it" message).
- [ ] B15. (sim retrain done: trained on the same noisy 720-beam sensing the relay sees, 150 rooms -> held-out AUC
  0.91 (was 0.86-0.88); real logs still to come) Retrain the intent model on real drive logs as they accumulate
  (the relay records stick + scans).

## D. Simulator / digital twin
- [ ] D1. (v1 done: the /dash 3D view shows the live relay state for the car or the simulator, incl. true walls, predicted path ribbon, contact X, manoeuvre, line. Still to do: world-fixed map frame, 3D obstacle models instead of LiDAR strokes) **3D view of the car simulator next to the car GUI**: one drive shown in 3D (world, car with the measured
  body and live steering, LiDAR rays, predicted path, collision X, manoeuvre + original line) and in the 2D GUI.
- [x] D2. (done v1: `python -m sim.twin_report` -> reports/twin_path.png: 48 x 3 s replays of the real logging drive, twin ends 4.5 cm from the car median, 9.6 cm 90th) **Twin accuracy evidence**: replay a real log's commands in the simulator, overlay simulated vs real
  (scan-matched) path with the error; the digital-twin testing cycle page.
- [x] D3. (done v1: reports/twin_lidar.png: static occupancy map from even scans, simulated beams at held-out scan poses, median |sim-real| 3.7 cm over 9953 beams; now 1.1 cm with the carved occupancy map, see D3b) **Real LiDAR vs simulated LiDAR demo**: take real scans from a log, rebuild the room, raycast the simulated
  LiDAR from the same poses, show both overlaid with the range error statistics.
- [x] D4. (done v1: `python -m sim.relay_mc 24` on the car's own decision code -> reports/monte_carlo_relay.png: no ADAS 17/24 crashes, ADAS 0/24 crashes and 24/24 reach the goal, 39 interventions of which 12 needless) **Visual Monte Carlo** of real-world scenarios (random rooms, obstacles, pedestrians, driver lapses) on the
  car's own relay code, side by side runs.
- [x] D5. (done via D8) (was: run, but NO difference yet: the steering-trend intent predictor in `pi/path_gate.py` never changed a decision - 12 needless interventions in both. Next: a learned intent model (GRU over stick/throttle/speed history, RESEARCH.md 4) predicting the driver's path distribution, and a driver model that telegraphs intent (gradual steering) so the comparison is fair) **Intent-aware vs not intent-aware** comparison on the same Monte Carlo runs (crashes, interruptions,
  warning lead time).
- [x] D8. (done: learned crash-risk intent model, AUC ~0.87 on held-out rooms; intent decides steering takeovers only. 48 paired drives: needless takeovers 17 -> 7 (-59%), 10 better / 0 worse, Wilcoxon p = 0.0008, crashes 0; brake-only baseline added. RESEARCH.md section 4) **Intent-aware must show a significant, genuine difference**
- [ ] D6. GUI "connection lost" flicker while the planner runs.

## E. GUI (EV-grade frontend)
- [x] E1. (native version done in L3: tabs Drive 3D / Map / Assists / Diagnostics / Reports / Events; v1 web: `pi/dash/index.html` at /dash on the car and in the simulator - 3D scene, speed/gear/throttle/steering cluster, time-to-contact ring, mode chip, alert banner, assist toggles, events, camera slot. Still to do: modes page (Drive/Assist/Autonomy/Diagnostics/Replay), intent bars, map - to be built in the native GUI, E3) Redesign: EV-style dashboard (speed, gear/direction, ADAS state, predicted path, alerts, camera slot),
  useful modes (Drive, Assist, Autonomy, Diagnostics, Replay), all relevant information visible.
- [x] E3. (done as L3: gui/dashboard.py) **Native (locally running) GUI instead of web GUIs** (user request), above all for the 3D simulator /
  digital twin: browser rendering and HTTP polling add lag. Follow the field's norm (Gazebo, CARLA, Webots and
  RViz are native apps): pick a native Python 3D toolkit, measure frame rate and input-to-screen latency against
  the web dashboard, and build the remaining E1 modes there. Keep the web page only for viewing from a phone.
- [x] E2. (done in gui/dashboard.py: path, TTC ring, intent card, 'what might happen', planned manoeuvre) Show "what might happen": predicted path, time to collision, intent probabilities, planned manoeuvre,
  alternatives considered.

## F. Autonomy (show everything the car can do)
- [x] F1a. Click-to-go (done: `adas/autonav.py` - Hybrid A* to a point with a single-arc analytic expansion and
  Reeds-Shepp-style reverse arcs, planner in a worker thread, pure pursuit + curvature/distance speed profile,
  re-plan when blocked. Relay: `GOTO x y` / `/api/goto/x/y`, operator holds the throttle as the dead-man switch,
  steer/brake hands back, holds after arrival until the throttle is released. 2D GUI: click the map. Twin
  (`python -m sim.autonav_eval` -> reports/autonav.png): 6/6 goals reached - doorway, side goal, furnished room,
  point beside the car, 52 cm gap, reverse in a corridor - 4-7 cm from the goal, 0 crashes, plans 1-440 ms.
  End-to-end through the relay in the simulator: doorway and corridor both arrive.)
- [ ] F1. (click-to-go done, F1a; the rest still to do) List and expose all autonomy modes in the GUI: obstacle avoidance run, follow-the-leader, return to start,
  explore/map the room, point-to-point navigation (click a goal on the map), auto-park (with camera later).

## G. Camera (one camera, front or rear) - plan first
- [x] G1. Write the camera plan (done: RESEARCH.md section 5) (RESEARCH.md): what it adds to every existing and future feature.

## H. Keep improving
- [ ] H1. (ongoing: RESEARCH.md section 6) Keep a running list of new feature ideas (autonomy, ML, visualisation) as the car is observed.

## I. Verification
- [ ] I1. User drives the laptop simulator and reports issues; fix from `logs/sim/`.
- [ ] I2. Same on the real car.

## J. Housekeeping
- [ ] J1. (partly done: tests/test_planning_safety.py - Hybrid A* doorway/reverse/boxed, click-to-go (6), gate beside-wall/head-on/not-frozen/phantom/latch, intent model, relay scenarios on the twin (2), Monte Carlo smoke; 58 tests pass. Still: dashboard) Tests for the new pieces (path gate, memory pruning, planner, hw simulator).
- [x] J2. (done: STATUS.md rewritten for the current state - what is/isn't on the car, calibration in use, results;
  README.md leads with the digital twin and its checks, the older simulator kept below) Update `STATUS.md` and `README.md`.
- [ ] J3. Results/slides from sim + car logs.
- [ ] J4. Calibrations only when the user asks (LiDAR yaw 45.7 deg since 28 Sep 21:21 - user-requested front calibration with the object centred: bearing -17.7 deg, sd 0.07, at 0.46 m, was 63.4; the mount probably turned during the ESP32 swap).
