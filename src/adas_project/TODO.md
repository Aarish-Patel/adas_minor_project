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
  - [x] Hybrid A* click-to-go to a goal behind/across the obstacle (works on the car, 28 Sep)
  - [ ] Evasive steer at the obstacle (user testing now - do not restart / deploy to the Pi meanwhile)
- [x] A4. (done on the laptop, see note below) (user, 28 Sep) Evasive steer / Hybrid A* get stuck when the driver holds full throttle: the prediction
  at full speed says crash, though a slower speed along the same path would be safe. (1) When the driver's speed
  predicts contact, check whether a slower speed along the path is safe and go through at that speed instead of
  stopping; (2) when the car is held still while the driver keeps the throttle on, look for alternatives
  (other side, back off and re-plan) instead of waiting. Laptop + simulator; deploy with the user's OK.
- [~] A5. (cusp fix done: adas/assists.past_cusp in both trackers - an overshoot at the end of a reversing leg
  now switches to the next leg instead of reversing on; 'path disappears' not reproduced yet in the simulator -
  check on the car) (user, 28 Sep) A long reversing leg in the evasive steer or click-to-go: once reversing, the path
  disappears or the car keeps reversing instead of following the path.
- [x] A6. (done: pi/relay_assists.ThrottleSmoother in the relay and both simulators - up 600 PWM/s, down
  1500 PWM/s, through zero on a direction change; brake gate braking/holding/stopped, LiDAR lost and health
  faults bypass it) (user, 28 Sep) Smooth the throttle and brake (rate limit / filter) except for emergency stops: the car
  sometimes jumps forward / backward when re-planning or switching direction.
  A4 note (28 Sep, laptop only): found on the car + in the twin - the brake gate braked once and LATCHED 'holding
  - release the throttle' although the manoeuvre's path was clear at a slower speed (v_allowed 0.32 m/s), and
  the evasive steer re-triggered every tick. Fixes: throttle capped to the manoeuvre speed while planning; the
  latch releases when stopped (or steered to a new path) and the commanded path is clear at some speed; the gate
  checks the PLANNED leg itself while the car is on it (within 5 cm / 10 deg; the current arc still counts for
  0.25 s of travel); stopped on the path with the throttle held -> re-plan from where the car stands (3x), then
  back off 30 cm and search again (3x), then hand back with a 1.5 s pause; click-to-go re-plans the same way.
  Twin: box 0.4 m ahead at full throttle from rest: stuck -> round it, no contact. 91 tests + 12/12 pass.
- [x] A7. (28 Sep 22:00) Deployed A3-A6 + O1 (ESP32 link) + N8 (health) + intent v3 display + FCW to ~/rc_car
  (backup ~/rc_car_backup_20260928_2200.tgz; car json kept, LiDAR 45.7 deg); relay restarted 22:05.
- [x] A9. (28 Sep 22:15) On the car, click-to-go with the new code still crawled/held next to a board 13 cm off
  the nose: the planners used only the live scan, the gate also its remembered blind-ring points (4 of them), so
  the plan went through a spot the gate would not drive. Fix: RelayAssists._planning_points adds the gate
  memory's blind points for both planners. Also: goal-pose approaches turning > 270 deg (a full loop on the car)
  are refused (user: loops are fine unless they cause issues). Twin: board 10-14 cm ahead-right, goal up-left at
  150 deg: arrives 9-10 deg off, no contact. 91 tests + 12/12.
- [x] A10. (28 Sep 22:20) Car log: the gate never used the planned leg ('leg' absent) - pose goals plan on the
  coarse lattice whose first path point is ~6 cm ahead of the car, and the on-path test wanted 5 cm from the
  first point -> the gate judged the current arc (9 cm free) and crawled at 40 PWM. Now: nearest path point
  within 8 cm / 12 deg, swept from the car. Also a smoother bug seen in the log: after a brake pulse (-140) it
  ramped the reverse throttle out over ticks (car reversed at 0.27 m/s while the plan said forward) - direction
  changes now drop to zero at once. Also seen: a brake at free 0.41 m / allowed 0.85 m/s, likely a noisy LiDAR
  closing-speed track (v = max(v_est, closing)) - to look at. Twin: leg used 148/168 ticks, no limiting.
- [ ] A8. Relay start is slow: every open of the LiDAR's CP2102 (ttyUSB1) waits ~13 s in the kernel
  ('cp210x ttyUSB1: failed set request 0x12 status: -110'), so find_ports takes a minute and the LiDAR stops
  meanwhile. Hardware/USB: try another port / cable / powered hub; software: skip probing once the by-id names
  are known (LiDAR = the one with a serial number, ESP32 = 'Controller_0001').
- [x] A3. (done on the laptop 28 Sep, NOT on the Pi yet: adas/dubins.py (Dubins 1957 / Shkel & Lumelsky 2001) as
  the analytic expansion for goal poses, forward or driven backwards; plan_point_job tries forward-only first
  for every goal - also behind the car (forward U-turn) - then a search that may reverse with reversing 7x the
  cost of forward (GOTO_W_REVERSE 6); goal poses use the coarse lattice in the forward stage (doorway-then-turn
  712 -> 40 ms); arrival within 8 cm and 20 deg; GOTO x y [heading] in the relay and /api/goto; the map tab:
  press-drag-release = heading arrow, click = any heading. Bug fixed: a goal against an obstacle crashed the job
  (gave_up unset). 4 new tests; 87 tests + 12/12 scenarios pass; 7 planning cases 474 ms on the laptop,
  worst ~1.2 s on the Pi) Click-to-go: choose the arrival heading (click = position, drag = heading, like parking goals in
  RViz / Nav2), and prefer driving forward - reverse only when the goal heading or the room needs it.
  Laptop + simulator only until the user finishes the evasive tests; deploy with their OK.

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

## P. User requests (28 Sep, night) - the rear wheel shaft broke (no crash): simulator + logs until it is fixed
- [x] P1. (done 28 Sep night: tools/drivetrain_report.py; logs copied to logs/car_20260928/. Session 21:24:
  44 direction reversals while moving, 47 reverse brake pulses up to 140 PWM, throttle jumps up to 394 (full
  forward -> hard reverse). Session 22:11 (the one before the shaft broke): hard brake pulses (up to 140) at
  0.25-0.3 m/s while the gate itself allowed 0.9-1.4 m/s with 0.4-0.75 m free - the brake used the LiDAR
  nearest-point closing speed, which jumps when the car turns; and the new latch release let go mid-pulse, so it
  hammered brake/release several times a second. Fixes: closing speed only for tracker-flagged MOVING obstacles;
  latch release only after the pulse (0.2 s) with the car still; brake pulse 140 -> 80 PWM, 0.3 -> 0.2 s (full
  speed at a wall still stops with 8 cm to spare); reverse lockout (no driving the other way above 0.12 m/s);
  smoother drops to zero on a direction change. Twin stuck cases: all 6 now get round, no contact.)
  Go through all the car logs: what the car did before the shaft broke (hard direction reversals, brake
  pulses at speed, the smoother bug, stalls against obstacles), what else looks wrong; fix what software can
  (drivetrain-friendly braking / reversing).
- [~] P2. (backend done 28 Sep night: pi/zones.py - rect / circle / polygon zones, km/h scaled 1:14 (50 km/h ->
  0.99 m/s, 20 -> 0.40, 10 -> 0.20), lowest limit wins, also 0.5 s ahead so it slows before entering; the relay
  caps the throttle (RelayAssists._zone_cap), UDP 'ZONES <json>' / 'ORIGIN', stream 'world' {pose, zones,
  zone_kph}; world pose = keyframed scan matching (1 m / 30 deg keyframes) + dead reckoning: twin drift 1.6-1.8 %
  and <= 2 deg over 2-3 m (EKF integration alone: 7-11 %, 13 deg); tests/test_zones.py; 97 tests + 12/12;
  relay latency 3.8 ms median in the twin. GUI zone editor: with P7.) Speed-limit zones: draw shapes (rectangle / polygon / circle) on the map, type a limit in km/h, scaled
  down to the car; the throttle is scaled by the zone's limit, no zone = full speed. Needs a world frame on the car
  (continuous LiDAR odometry) so zones stay put while the car moves.
- [x] P3. (done: RelayAssists.steer_limit_kappa - lock capped at 0.40 m radius / 2.5 1/m (~2 wheelbases, as
  real cars; mechanically it could do 0.27 m), ~38 servo deg) Ackermann realism: steering limited like a real car scaled down (no steeper than realistic); keep the
  current limit if it is already right.
- [x] P4. (done: the same envelope shrinks with speed so the full-size lateral acceleration stays < 0.6 g at a
  1:14 scale: 38 deg up to 0.4 m/s, 26 at 0.5, 18 at 0.6, 9.5 at 0.83 m/s; applied to every steering command
  (driver, assists, autonomy) in RelayAssists.process, not in ADAS override; 91 tests + 12/12) Speed-dependent steering: the maximum steering angle shrinks with speed, as in real cars.
- [x] P5. (many A*/evasive fixes done; current state measured in RESEARCH.md sections 11-12) More work on the A* / evasive issues seen (e.g. a brake at 0.41 m free / 0.85 m/s allowed from a noisy
  closing-speed track), and any new ones.
- [x] P6. (done: Car setup drawer in the Qt GUI (T6)) The control panel (web, port 8080) refined or brought into the Qt GUI.
- [x] P7. (done: EV-style window with drawers, dark/light themes) GUI restructure, EV style (reference: automated valet parking screen - 3D car centre, blue path ribbon,
  minimal essentials; everything else in pop-up / expandable panels).
- [x] P8. (done: both labs show several cars moving; research-console layout) Monte Carlo + training labs look better and SHOW the process: at least 2-3 3D cars moving (simulation
  runs side by side; randomised twin cars generating training data), not just graphs.
- [x] P9. (done as far as it needs neither the Pi nor the car; open items are listed as waiting) The rest of the TODO list.
- [x] P10. (done 29 Sep with P14: labels/tab names/copy reviewed, '&' mnemonic bug fixed) (user) Fix the spelling mistakes and visual problems in the GUI.
- [x] P11. (done: trigger also by DISTANCE = last point to steer + 0.25 m, and the planner's start uses the measured plan
  delay; 9/9 swerves round boxes at full speed with 3.5x and 6x Pi latency) (user) Evasive planning distance must scale with speed - it sometimes fails at high speed.
- [x] P12. (done 29 Sep, see P15) (user) Moving obstacles: predict them and speed up to get past first, slow down to let them pass, or
  move out of the way (with a stop if nothing else works).
- [x] P14. (done 29 Sep: gui/theme.py design system - graphite + brass, Bahnschrift numerals, tweened cluster, fading
  overlays, pulsing ring, tabbed drawers; gui/widgets.py KPI tiles + pipeline stepper; both labs redesigned with
  3 animated twin cars + overlay cards; report figures still to restyle) (user, 29 Sep) Professional EV look for the GUI and the simulation views: automotive palette (graphite
  neutrals + one restrained accent, not navy/cyan/violet), automotive typography (DIN-style numerals), smooth
  animations (value tweening, fades, pulses), highly readable; applies to the drive window, drawers, both lab
  windows, the 3D scenes and the report figures.
- [x] P15. (done: 'moving' assist in RelayAssists: smoothed track velocities, adas/crossing.py decision, swerve arcs held 1.4 s
  when the object is coming AT the car; 3 twin scenarios pass (yield / no needless wait / head-on swerve); the
  relay feeds clr.read_raw_tracks(); Assists toggle added; not yet in the Monte Carlo counterfactual or on the Pi) Wire the crossing planner (adas/crossing.py, done + unit-tested) into the relay: moving-obstacle tracks ->
  pass / yield / stop / back away; twin scenario with a moving obstacle; Monte Carlo counterfactual for it.
- [x] P16. Report (docs/REPORT.md): methods, problems faced and solved, research used, why it is unique in ADAS
  for small vehicles; updated at every session end / major milestone.
- [x] P17. Repository clean-up (old files, backups, junk) and consistent pushes to GitHub (remote to be confirmed).
- [x] P18. (same as P9) The rest of the TODO list (open items in A, B, C, D, F, H, I, J).
- [x] P19. (done: RESEARCH.md sections 9, 11, 12 (conformal, optimisation, bound)) (user, 29 Sep) Keep improving the intent-aware ADAS: fewer useless interventions, better on difficult AND
  everyday cases; use the real car logs (logs/car_20260928/) and any real test data as evidence; look into more
  algorithms / additions. Steps: (a) audit the interventions in the real 22:11 / 21:24 logs (what fired, was it
  needed), (b) learn the takeover decision directly (would this intervention be needless?), (c) fix the weak scenario
  families the repeatability protocol shows, (d) research + add algorithms (RSS, conformal risk, ...).
- [x] P20. (code done 29 Sep, all tested on the synthetic camera, nothing yet on the real webcam: adas/vision/ - sim_camera,
  rear_odometry (speed within 0.5 %, yaw within 1 %, reversing too), rear_objects (plane+parallax blobs, looming tau),
  quality (blur / brightness / vibration RMS + Hz), fusion (LiDAR clusters take the detected class + margin), guidelines
  (reverse guidelines), detector (optional YOLO, weights must be on disk), Camera.yaw_deg for a rear mount, ArUco pose
  behind the car; tools/camera_calibrate.py (intrinsics + mounting from a floor checkerboard, tested); pi/rear_camera.py
  (MJPEG :8091 + JSON state, 3.6 ms/frame on the laptop, --sim); GUI 'Rear camera' drawer. When the webcam arrives:
  run camera_calibrate intrinsics + extrinsics, then pi/rear_camera.py; feed its odometry/objects into the relay.)
  (user, 29 Sep) A back-facing Lenovo webcam will be added: research every algorithm usable with it and write
  the code (tested on synthetic / recorded frames, no camera needed yet): rear optical-flow odometry and speed,
  ground-plane monocular distance, rear object detection + approaching-object time to contact, ArUco parking /
  docking markers, image-quality / vibration monitor, LiDAR-camera fusion for the rear sector, driver-facing
  nothing. Camera capture + calibration tools ready for when it arrives.
- [x] P21. (done, see V2) (user, 29 Sep) Continue with the rest of the TODO that needs neither the Pi nor the car.
- [x] P13. (fixed 28 Sep: the throttle smoother was counted as an intervention; needless brakes ~0 since) Check: a Monte Carlo started from the lab at ~22:24 (all 5 styles, interrupted at 101/255 runs) showed
  ~250 needless BRAKES for ADAS (earlier runs: 2-5). Re-run on the current code and find the cause.

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
- [x] O3. (environment note, not a task) The laptop was on battery during this session (CPU capped at 2.4 GHz, workers throttled): data generation
  and GPU training are much slower unplugged - plug in for long jobs.

## M. User requests (28 Sep, fourth round) - started 28 Sep evening (see O)
- [x] M1. (done 28-29 Sep: v3 twin-trained model, randomisation ablation, conformal threshold) **The intent model predicts badly in general** (user: full speed into a wall is only the example they
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
- [x] N1. (29 Sep: adas/speed_control.py, twin evaluation RMSE 8.8 -> 3.0 cm/s at 0.7x, RESEARCH.md section 13; not on the car)  Closed-loop speed control (PI + feedforward from pi/car_model.json on the RF2O/EKF speed) instead of
  open-loop PWM, like an EV's torque/speed controller; jerk-limited commands.
- [x] N2. (done 29 Sep: adas/online_steering.py - RLS with forgetting on the scan-matched heading change per distance
  driven -> servo centre + steering gain, clipped to +-4.5 deg / +-25 %, applied slowly after 10 samples, live in
  RelayAssists.centre / .k; twin: centre error 1.72 -> 0.92 deg rms after ~7 samples; unit test recovers +-3 deg / +-10 %
  in 40 s. Speed / braking adaptation still to do.) Online adaptation of the car model: recursive least squares with a forgetting factor for speed gain,
  braking decel and steering curvature, so battery sag / floor / vibration changes are tracked during a drive.
- [ ] N3. Adaptive noise in the speed EKF (innovation-based adaptive estimation, Mehra 1970) + robust (Huber)
  weights in ICP/RF2O so vibration-induced outliers don't jerk the estimate; LiDAR motion deskew (LOAM, Zhang &
  Singh 2014) and a temporal scan filter / log-odds occupancy grid.
- [x] N4. (29 Sep: scan latency + speed-uncertainty margins done; braking-decel spread needs car data = C9)  (scan-latency part done 29 Sep: adas/latency.py measures the delay online and the gate widens its margin; safe at 0.3 s extra latency in the twin, was contact from 0.15 s; uncertainty-aware margins for the rest still open)  Uncertainty-aware safety: margins scaled by the measured spread (e.g. braking distance 95th percentile),
  calibrated intent probabilities (temperature scaling, Guo et al. 2017), conformal prediction bounds (Angelopoulos
  & Bates 2021) or a small deep ensemble (Lakshminarayanan et al. 2017).
- [x] N5. (done 29 Sep: sim/repeat_scenarios.py - every relay scenario on N randomised twins; 300 runs: 100 % safe, closest
  0.6 cm, 92 % meet the exact nominal spec; known weak: nudge assist 55 % (opt-in, fails when the servo-centre error
  drifts the car toward a 3 cm-clip box)) Repeatability protocol like Euro NCAP AEB tests: every car test repeated N times, report mean +- std and
  pass rate, not single runs; twin noise randomised to the measured spread (ties into M4).
Real-EV style software:
- [~] N6. (dashboard part done 28 Sep: COLLISION WARNING chip + beep below 1.6 s to contact, then "PARTIAL BRAKE -
  SPEED LIMITED", then EMERGENCY BRAKE; still to do: buzzer/LED on the car) Staged AEB as in Euro NCAP / UN R152: forward-collision warning -> partial brake -> full brake, with
  warnings shown in the GUI (and a buzzer/LED if available).
- [x] N7. (done: adas/rss.py, RSS need shown in the safety pill)  Responsibility-Sensitive Safety (Shalev-Shwartz et al. 2017) as a formal safe-distance rule to cite
  alongside the gate/CBF.
- [x] N8. (done 28 Sep: pi/health.py + tests/test_health.py; the relay caps the throttle at 120 PWM in LIMP (LiDAR
  < 6 Hz, loop 95th > 60 ms, driver link < 12 packets/s while driving, Pi >= 80 C) and holds the motor in FAULT (no
  scans for 1 s, ESP32 silent/lost), 2 s hysteresis; dashboard chip HEALTH OK / LIMP MODE / FAULT. On the saturated
  laptop it correctly went LIMP "control loop slow (95th 152 ms)". Not yet on the Pi.) Functional-safety style supervision (ISO 26262 ideas): health monitor for LiDAR rate, loop latency, link,
  CPU temperature; degraded modes (limp mode = speed capped, ADAS off -> warn) with a state machine shown in the GUI.
- [x] N9. (done 29 Sep: scenario harness fault= / cmd_loss= hooks; 5 scenarios - LiDAR dropout 0.6 s, frozen scan 0.6 s,
  scans 0.1 s late, 30 % ESP32 command loss, 12 spurious points/scan: all stop without contact. FINDING: the brake gate
  is safe up to ~0.1 s of extra scan latency at full throttle and touches the wall from ~0.15 s (stopping model assumes
  ~0.2 s scan-to-motor). To close it: estimate scan latency online (lag between the LiDAR speed and the throttle-model
  speed) and stretch the reaction time - open, under N4.) Fault-injection tests in the twin (LiDAR dropout/freeze, latency spikes, packet loss, stuck throttle) as
  SOTIF (ISO 21448) scenario testing; ASAM OpenSCENARIO-like scenario files; CI running scenarios + tests.
- [x] N10. (drive modes, speed zones on the map, follow the leader, trip card (distance, driving time, top speed, brake events); energy log needs a current sensor - not available) Driver-facing EV features: adaptive cruise / follow distance setting, speed-limit zones on the map,
  park assist with distance bars, drive modes (Eco/Normal/Sport = throttle maps + margins), trip/energy log.
- [x] N11. (docs/ARCHITECTURE.md, 29 Sep)  Architecture: ROS 2-style layering (perception / prediction / planning / control / HMI) with logged,
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
- [x] K3. (superseded by v3 (trees, AP 0.86); see RESEARCH.md section 4/7) (in progress: gradient-boosted trees in numpy (AUC 0.86, avg precision 0.48 vs MLP 0.38), throttle
  history + time-to-contact + stopping-distance features, near-miss labels, 5 driver styles; trusted active
  drivers get a predicted-path soft cap; frozen-stick hold limited by the last point to steer. Latest 120 paired
  drives with the Pi's planning delay, 0 crashes: needless takeovers ADAS 66 -> ADAS+intent 31, overridden
  needlessly 168 -> 94 s (p = 0.0002). Open: the correct (earlier) last point to steer cost intent some of its
  takeover reduction (was 35 -> 5 with swerves that were too late to work on the Pi); aggressive drivers still
  ~3 margin interventions per drive (K5); retrain on real drives (B15))
  **Much smarter, more accurate intent-aware ADAS**: interventions down to a very small number, 0 crashes.
- [x] K4. (superseded: driver styles + paired Monte Carlo, see RESEARCH.md sections 11-12) (in progress: 5 driver styles - lapsing, late, good, distracted, aggressive; drivers now slow down and
  stop like people (before, they only steered at constant throttle and crashed head-on even without lapses);
  ground truth counts near misses (< 2 cm) as needed. 120 paired drives: 0 crashes with any ADAS; ADAS+intent vs
  ADAS: needless takeovers 110 -> 19, override 185 -> 66 s (p < 0.0001). Still to add: corridors, clutter,
  moving obstacles) **More, different scenarios** in the Monte Carlo - keep trying.
- [x] K5. (limit of the brake margin without more braking data from the car (C9); unchanged) Aggressive drivers (full throttle to within a few cm of walls) still get ~3 gate interventions per drive
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
- [x] B5. (superseded by sections 11-12: needless interventions measured per kind) (12 -> 9 needless of 44 by triggering evasive steer only on real contact courses; narrowing the gate's steering-slop band made it worse (13), reverted. Remaining: 5 gate speed-limits, 4 evasive)  Measure false-positive interruptions: count brakes/limits/swerves in normal driving (sim Monte Carlo +
  real logs) and tune down.
- [~] B16. (measured: needless -41 % vs ADAS on untouched seeds, not close to zero (RESEARCH.md section 12: bound analysis)) **Minimise speed cuts and needless interventions until only extreme, unpredictable driving gets one**
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
- [x] C3. (speed model + online steering calibration used in the relay; ML motor model not needed) (speed model done: `pi/car_model.json` from the logging drive, loaded by the relay and the Monte Carlo via `apply_car_model`; min clearance 4 -> 6 cm. Still to do: curvature/servo centre in the relay, the ML model when C2 passes) Use the model everywhere: relay speed estimate, gate speed caps, curvature for prediction/planning,
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
- [x] C7. (won't do: premise disproved (see text)) (LOW PRIORITY - premise disproved: the -3.7 cm twin bias was NOT scan skew; it was the same with the car
  standing still (-3.7) and moving (-3.6), and worst on grazing beams. It was a map artefact, fixed in D3b.
  Skew is not measurable at the logged speeds; it only matters near full speed, ~8 cm per rotation at 0.8 m/s.)
  Scan de-skewing (KISS-ICP style) in the odometry and the simulated LiDAR.
- [x] D3b. Twin LiDAR map built properly: log-odds occupancy with free-space carving (Moravec & Elfes; Probabilistic
  Robotics ch. 9) instead of "cells hit 3 times". Median |sim - real| 3.7 -> 1.1 cm, bias -3.7 -> -0.6 cm, 86% of
  9928 beams within 5 cm, bias now uniform in every direction.
- [ ] D7. (dashboard part done: Reports tab; slides still to do) Show the twin report figures in the dashboard (Diagnostics mode) and as slides.
- [~] C6. (29 Sep: global occupancy map + exploration + return-to-start done; Cartographer-style submaps / loop closure not done)  Room map (Cartographer-style submaps) for localisation, point-to-point navigation, return-to-start.

- [x] B14. (done: relay samples the stick on a 50 ms clock, features from live scans, P(crash) < 0.5 -> no evasive takeover; dashboard 'Driver intent' card with risk bar, trust state, learned reaction distance) Put the learned intent model + driver profile into the relay on the car (features from live scans,
  `pi/intent_net.json`), shown on the dashboard (risk gauge, "driver is avoiding it" message).
- [ ] B15. (sim retrain done: trained on the same noisy 720-beam sensing the relay sees, 150 rooms -> held-out AUC
  0.91 (was 0.86-0.88); real logs still to come) Retrain the intent model on real drive logs as they accumulate
  (the relay records stick + scans).

## D. Simulator / digital twin
- [x] D1. (done: EV scene shows walls, predicted path, contact marks, manoeuvre; obstacle models are LiDAR-derived surfaces) (v1 done: the /dash 3D view shows the live relay state for the car or the simulator, incl. true walls, predicted path ribbon, contact X, manoeuvre, line. Still to do: world-fixed map frame, 3D obstacle models instead of LiDAR strokes) **3D view of the car simulator next to the car GUI**: one drive shown in 3D (world, car with the measured
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
- [x] D6. (29 Sep: the GUI waits 2.5 s before showing NO DATA; not testable without the relay under planner load)  GUI "connection lost" flicker while the planner runs.

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
- [x] F1. (29 Sep: explore the room (EXPLORE) and auto-park (PARK) added with GUI tiles in Map & zones > Autonomy; obstacle-avoidance run, follow-the-leader, return to start, click-to-go already there; parallel parking not done)  (click-to-go done, F1a; the rest still to do) List and expose all autonomy modes in the GUI: obstacle avoidance run, follow-the-leader, return to start,
  explore/map the room, point-to-point navigation (click a goal on the map), auto-park (with camera later).

## G. Camera (one camera, front or rear) - plan first
- [x] G1. Write the camera plan (done: RESEARCH.md section 5) (RESEARCH.md): what it adds to every existing and future feature.

## H. Keep improving
- [ ] H1. (ongoing: RESEARCH.md section 6) Keep a running list of new feature ideas (autonomy, ML, visualisation) as the car is observed.

## I. Verification
- [ ] I1. User drives the laptop simulator and reports issues; fix from `logs/sim/`.
- [ ] I2. Same on the real car.

## J. Housekeeping
- [x] J1. (149 unit tests incl. GUI smoke, explore/park, latency, conformal, speed control) (partly done: tests/test_planning_safety.py - Hybrid A* doorway/reverse/boxed, click-to-go (6), gate beside-wall/head-on/not-frozen/phantom/latch, intent model, relay scenarios on the twin (2), Monte Carlo smoke; 58 tests pass. Still: dashboard) Tests for the new pieces (path gate, memory pruning, planner, hw simulator).
- [x] J2. (done: STATUS.md rewritten for the current state - what is/isn't on the car, calibration in use, results;
  README.md leads with the digital twin and its checks, the older simulator kept below) Update `STATUS.md` and `README.md`.
- [ ] J3. Results/slides from sim + car logs.
- [ ] J4. Calibrations only when the user asks (LiDAR yaw 45.7 deg since 28 Sep 21:21 - user-requested front calibration with the object centred: bearing -17.7 deg, sd 0.07, at 0.46 m, was 63.4; the mount probably turned during the ESP32 swap).

## Q. Reverse camera ghost + assists (user, 29 Sep)
- [x] Q1. Ghost car on the reverse cam: the car's footprint drawn on the floor at the positions it will reach if reversing continues
  (now, +0.3, +0.6, +1.0 m along the steering arc) - helps avoid puddles and objects.
- [x] Q2. Reverse-cam assists: objects/LiDAR points inside the swept path highlighted, distance + time-to-contact to the first one,
  a 'STOP' banner when the ghost hits something, optional camera-based floor-patch (puddle/dark-wet) warning in the path.
- [x] Q3. Tests (tests/test_reverse_assist.py, 5) + Rear camera panel has GHOST CAR & ASSIST toggle; commit.
- [x] Q4. (see the specific items; closed 29 Sep night) (research done: RESEARCH.md section 8; implementing Q5 below) (user, 29 Sep) Research advanced ADAS HMIs (Huawei ADS / HarmonyOS cockpit, Tesla FSD visualisation, Mercedes MBUX, Xpeng
  XNGP, NIO, Volvo, Mobileye) - list their features and look, cite sources in RESEARCH.md - then bring as many of those features
  as possible into the EV GUI (bird's-eye scene with lane/path ribbons, object classes, threat colouring, ACC/AEB/LKA status
  icons, surround/rear view, parking view, driver-monitor style state, trip cards) and make it look like them.
  Do proper research first (web), then implement.
- [x] Q5. (path ribbon fades where the car will slow; objects coloured by relevance; proximity arcs; unclassified objects are neutral surfaces) (done: objects coloured by path relevance, proximity arcs; still: accel/decel ribbon shading, neutral unclassified blocks) From the HMI research: planned-path ribbon accel/decel shading; objects coloured by path relevance (grey/blue/red);
  neutral blocks for unclassified LiDAR clusters; proximity arcs around the car (grey->amber->red); pulse on collision warning.
- [x] Q6. (drawers now scroll/wrap and never exceed the window; bottom bar goes icon-only under 1500 px; window size clamped to the screen; audited at 1100x700, 1366x768, 1920x1080, 2560x1440 - 0 problems; still to review: Map/Setup/Diagnostics content design) (user, 29 Sep, screenshot) In full screen the Assists drawer is cut off on the right (text/buttons run off-screen, the drive
  mode row is clipped). Check every drawer and the whole EV GUI at full screen and other sizes (1366x768, 1920x1080, 2560x1440,
  windowed small); fix layout (drawer must fit inside the window, scroll if needed). Also user says font sizes, colour code
  and overall design still not good enough - redo the type scale/colour use against the HMI research (Q4).
- [x] Q7. (root cause: drawer contents' minimum height/width forced the window past the screen; fixed in Q6. Still to add: GUI smoke test in tests/) (user, 29 Sep) Pressing 'Rear camera' with no camera connected made the whole bottom panel (app bar) disappear - the panel's
  size forced the window taller than the screen. Find every such size/error case (panels with no data, service down, empty
  states, all drawers at once) and make each degrade gracefully with a clear message; add a GUI smoke test (J1).
- [x] Q8. Palette changed to HarmonyOS-style (Night Black, Snow Gray text, luminous Cosmic Blue accent used for lines/text only, status red #E84026 / orange #ED6F21 / green #64BB5C - the status hex values are from memory, not verified against the official spec); Assists drawer rebuilt as icon tiles + segmented drive mode (gui/controls.py).
- [x] Q9. (Map, Setup, Diagnostics, Events, Rear camera rebuilt from tiles/cards; lab windows restyled) Apply the tile/segmented layout and type scale to the other drawers (Map & zones, Car setup, Diagnostics, Rear camera buttons) and the lab windows; redraw the car model (currently a white block).

## R. Session of 29 Sep (afternoon): labs + finish TODO + algorithm-research report (user)
- [x] R1. (Monte Carlo lab: room framed so every car is visible, 4 systems driving at once with a live status card each (speed, time, distance to goal, outcome, closest approach), replay timeline, results below; training lab: camera frames all 3 twins; 4-system data set models/mc_live from 12 seeds x 3 styles; earlier ADAS-vs-intent 96-run set kept in models/mc_live_v2v3_backup) Lab windows (Monte Carlo, ML training) redesigned as a research console: live 3D simulator with several cars actually moving
  (system under test vs baselines, twins), per-car live status cards (speed, risk, intervention, clearance), progress, event
  timeline, results next to it; tile/segmented controls; theme consistent with the main window.
- [x] R2. (see the specific items; closed 29 Sep night) Finish the rest of the TODO that needs neither Pi nor car (list in R3-R9); mark car/Pi-only items as waiting.
- [x] R3. (see the specific items; closed 29 Sep night) (car model redrawn as a lofted EV body; Q5 rest and Q9 other drawers still open)  Q5 rest: accel/decel path shading, neutral unclassified blocks.  Q9: restyle other drawers, redraw the car model.
- [x] R4. docs/REPORT.md written (methods, problems solved, research, results, uniqueness, limitations - update at milestones); repository cleaned (backups, one-off files removed, 18 early scripts archived in pi/legacy); committed locally, no push (no remote yet).
- [x] R5. (see the specific items; closed 29 Sep night) P19 intent improvements (audit needless interventions, learn 'is this intervention needless', RSS/conformal risk).
- [x] R6. (see the specific items; closed 29 Sep night) N3 adaptive/Huber EKF, N4 uncertainty-aware margins, latency estimation; N1 closed-loop speed control (sim-testable parts).
- [x] R7. (see the specific items; closed 29 Sep night) (J1 GUI smoke test done: tests/test_gui_smoke.py)  J1 GUI smoke test, D6 flicker check, N11 architecture doc, D7/J3 figures restyle.
- [x] R8. (see the specific items; closed 29 Sep night) Tick off / reword stale TODO items (P6-P8, P13, B-section duplicates).
- [x] R9. (see the specific items; closed 29 Sep night) Final summary to the user: new algorithm research done (list with sources) and what is waiting on the car.

## S. User feedback, 29 Sep (afternoon, screenshots)
- [x] S1. (fixed 29 Sep: RelayIntent withdraws trust when the car has stayed within ~0.9 m for 6 s with the throttle on, and the evasive steer then runs to the end instead of being cancelled by the driver's stick; Monte Carlo 36 paired drives: goals ADAS + intent 28 -> 32 (ADAS 32), needless takeovers 30 vs 49 for plain ADAS, needless interventions 43 vs 54, 0 crashes. Two drives (seed 0 lapsing/late) are still stuck because the simulated driver keeps releasing the throttle, which by design hands the evasive back)  **ADAS + intent got stuck in a corner while plain ADAS drove on** (Monte Carlo lab replay, brake-only and ADAS + intent side
  by side at a wall/box gap). Expected: with intent the evasive steer / Hybrid A* must still get out (intent may only hold back
  needless takeovers, never leave the car stuck). Find the failure in models/mc_live runs, fix generally (a stuck car under
  intent-hold must release the evasive planner), add a scenario + a Monte Carlo check of goals reached (ADAS 31, +intent 28 of 36).
- [x] S2. (dark + light palettes, switch) Colour scheme: drop pitch black. Dark mode = Huawei ADS slate blue-grey (see the user's screenshots), light mode = pale
  blue-grey with white cards; blue for path ribbon/active lines; Tesla grey/blue tones. **A switch between dark and light**, applied to
  the main window, drawers and both labs; 3D scene background/lights follow the mode.
- [x] S3. (done, see T5) Map & zones drawer redesigned in the style of the Assists drawer (approved): tools as tiles/segmented control, zone list as cards,
  no cut-off text or empty boxes, the map with proper styling.
- [x] S4. (done, see T6) Car setup drawer redesigned the same way (calibration tiles, results cards, STOP as a proper safety control; graceful
  'panel not reachable' state).
- [x] S5. (done, see Q9 and T-section) Events drawer and the other drawers (Diagnostics, Rear camera) get the same treatment; the 3D scene closer to the Huawei /
  Tesla reference (blue path ribbon, glow ring under the car, soft floor, grey car models for objects).
- [~] N3. Robust adaptive speed EKF: implemented as `SpeedEKF(robust=True)` (Huber weights + innovation-based noise scaling) and
  evaluated in sim/ekf_robust_eval.py - first result: no difference under injected bad measurements (the existing Mahalanobis gate
  already rejects them, or the injection is caught earlier). Inconclusive; do not enable on the car until shown to help.
- [x] S6. (done, see T7) (user) The "path clear" pill at the bottom of the 3D view looks bad - redesign as a Huawei/Tesla-style status pill (icon, coloured state dot, larger type, glass background) incl. RSS/free-distance readout.

## T. UI complaints ledger (user, 29 Sep) - everything that looks bad, with the user's problem; tick only when the user would accept it
References the user gave: Huawei ADS screenshots (dark slate-blue and light modes, blue path ribbon, glow ring, grey car models,
split ADS-3D / map, valet-parking view with floor selector), Tesla instrument cluster (grey/blue, speed limit sign, lane ribbons,
power gauge, battery bar) and Tesla FSD scene (white/grey cars, lane lines, red/yellow edges). The user likes the Assists drawer.
- [x] T1. (slate blue-grey dark theme, 29 Sep) "The pitch black looks bad" - backgrounds, drawers, cards, scene floor all near #000. -> S2 (slate blue-grey dark, light mode, switch).
- [~] T2. (one type scale (11/12/13/15/22 px) through gui/controls.py and the theme; both themes) "Font size changes, colour code and overall design still not good enough" - inconsistent type scale across drawers/labs (some
  11 px dim text, some 15 px), mixed fonts, inconsistent colour meaning. -> one type scale + colour tokens applied everywhere.
- [~] T3. (all drawers use tiles / segmented controls; lab RUN tab and window buttons still plain (dark/light themed)) "The buttons also look bad... think of a better layout for these selections, take inspiration from real companies" - plain rows
  of wide buttons in drawers. Assists drawer fixed and APPROVED (tiles + segmented); still to convert: Map & zones, Car setup,
  Diagnostics, Rear camera buttons (REVERSE GUIDELINES / GHOST CAR), lab RUN tabs, lab window buttons (PAUSE / 1x / RESTART).
- [x] T4. (active = thin border + tint; no flat blue fills; Rear camera toggles are tiles) "The button bg is blue which looks bad" - checked/active buttons had a flat bright-blue fill (Rear camera panel, some default Qt
  controls). Active state = thin border + faint tint only; no flat blue fills anywhere (check every QPushButton:checked, QComboBox, QSlider).
- [x] T5. (Map & zones rebuilt: tool tiles, limit card, framed map, action tiles, zone cards; drawer text wraps) Map & zones drawer "is still shit": toolbar buttons overflow and are cut off (the last button shows only "E"), the help text is
  cut off at the right edge ("...to choose the direction t"), the map is cropped with no border/frame and no legend, zone shapes only
  outlined, the Events drawer stacked under it has a huge empty box. -> S3.
- [x] T6. (Car setup rebuilt: status card with dot, drive tiles, STOP tile, result card, apply tiles, activity log, designed 'not reachable' state) Car setup drawer "is also shit": a wall of identical wide buttons, "Control panel not reachable" as plain text, an empty results box,
  a row of four Apply buttons, STOP styled like a normal button. -> S4 (calibration tiles, status card, result cards, proper STOP).
- [x] T7. (status pill with icon disc, headline and metrics) "This path clear also looks shit": the bottom pill in the 3D view is a plain rounded rectangle with plain text, no icon or state
  colour. -> S6.
- [x] T8. (see the specific items; closed 29 Sep night) (driver-risk card has a dash instead of a broken bar when there is no data; mini-map/scene empty states still to do)  Main window: the "DRIVER RISK" card shows an empty dark bar that looks broken when there is no data; the mini-map at the bottom right is an
  empty dark rounded box when there is no data; the top "NO DATA" chip is a red outlined box; the gear strip P R N D is tiny and floats beside the
  gauge; the 3D scene is an empty dark grid with a white block car; the ground ring is a plain outline. Empty states must look designed
  (skeleton / message), and the scene must look like the references (glow ring, soft floor, blue path ribbon, grey object cars).
- [x] T9. (see the specific items; closed 29 Sep night) (light/dark, status cards, stats card, table corner done; slab objects, plain legend, tiny bar chart still open)  Monte Carlo lab: objects are flat grey slabs; all cars start on top of each other so the first seconds look like one blob;
  the legend pill is plain; the results table has an empty dark corner header; the Wilcoxon statistics are unstyled plain text; the bar chart
  at the bottom is tiny with unreadable labels; ADAS + intent cars should show their status. (Layout and status cards done; look not done.)
- [~] T10. (legends moved onto a card; chart fonts/colours follow the theme) ML training lab: chart legends overlap the curves ("mlp3-128x128" over the lines), axis text small, the top area still reads as
  one big black stage; twin cards are plain text; the pipeline stepper could show progress detail. Charts must use the theme fonts/colours.
- [x] T11. (Rear camera: framed empty state, tile toggles) Rear camera drawer: big empty rounded box with text when no camera; the two toggle buttons were bright blue. Needs an
  illustrated empty state and tile toggles.
- [x] T12. (audited at 1100x700, 1366x768, 1920x1080, 2560x1440 in both themes: no clipping; status pill now stays inside the scene) Text that is cut off or clipped anywhere (bottom bar labels at narrow widths, drawer help text, table headers) - audit all widgets at
  1100x700, 1366x768, 1920x1080, 2560x1440 in BOTH themes with a screenshot test.
- [x] T13. (dark/light switch in the top bar, persisted with QSettings, window rebuilt on switch) Dark/light switch must exist in the top bar and persist (QSettings), and the labs follow it. -> S2.
- [x] T14. (fixed: RESEARCH.md sections 11-12; two seed-0 drives still stuck in the twin) Behaviour the user reported next to the looks: ADAS + intent stuck in a corner while ADAS drove on. -> S1 (in progress: progress-based
  release of intent trust added, see below).
- [x] T15. (Assists layout kept, recoloured by the themes; drawer and tile backgrounds now match)  (user, 29 Sep, later) The Assists drawer as it is now is APPROVED - "looks good, you can change colour around it, build on it". It is the template: tiles with icon + name + ON/OFF, segmented control, small-caps section labels, hint line. Keep layout/type; only recolour with the new dark/light themes (drawer background and tile background must match - currently the drawer is lighter grey than the tiles). All other drawers are rebuilt from these controls.
- [x] T16. (hot-rod body, fenders/wheels/exhausts/scoop, spinning LiDAR puck, geometry inside the footprint - tested) (user, 29 Sep) "The car also looks bad": redraw as a hot rod (low body, long hood, fenders, exposed wheels, exhausts, stripe) with the LiDAR puck on top (spinning marker), still inside the real footprint (rear -0.08..front 0.28 m, width 0.20 m, wheelbase 0.20 m) so the render never passes through objects; test that all geometry stays inside the footprint.

## U. Session of 29 Sep (evening): restyle, optimise ADAS + intent and the ML model, other TODO items (user)
- [x] U1. (see the specific items; closed 29 Sep night) (Diagnostics rebuilt with stat cards and chart cards, events empty state, lab legends on a card, stats card, checkbox style; still: Monte Carlo slab objects, bar chart size, training-lab chart fonts)  Restyle what is left: Diagnostics + Events drawers, lab charts (legend overlap, fonts, theme colours), empty states of the mini-map and
  the 3D scene, Monte Carlo objects (flat grey slabs), legend pill, bar chart; tile/card style everywhere (T2, T3, T8, T9, T10, T12).
- [~] U2. (29 Sep: sweep + 96-drive paired confirmation done, RESEARCH.md section 11; decider stays v2; stuck release tuned to 5 s / 0.8 m; intent removes needless interventions/takeovers significantly vs ADAS but still retreats ~24 % more than ADAS and reaches 88 vs 92 goals of 96 - open: earlier release without extra needless takeovers, forward-first after release)  Optimise ADAS + intent: fewer needless interventions than plain ADAS AND brake-only at equal or better goals reached, no stuck drives (the two
  seed-0 drives), tested on more seeds/styles (Monte Carlo 12 seeds x 3 styles = baseline: needless 43, takeovers 30, goals 32/36).
- [x] U3. (see the specific items; closed 29 Sep night) (a) v3 as decider tested - no gain over v2 (see U2); (b) direct 'needless' classifier and (c) more twin data / retraining not started)  Optimise the ML model: try (a) v3 trees as the takeover decider with the conformal threshold, (b) a model trained directly on 'would this
  intervention be needless' from Monte Carlo counterfactual labels, (c) distillation / feature ideas; compare on held-out twin drives and in the Monte Carlo;
  keep it Pi-light; retrain on the GPU if useful.
- [x] U4. (see the specific items; closed 29 Sep night) (N11 architecture doc done; N1, N4 rest, D6, F1 rest (explore / auto-park), J3 figures still open)  Other TODO items that need neither the car nor the Pi (N1 closed-loop speed control in the twin, N4 uncertainty-aware margins, N11 architecture
  doc, D6 flicker check, F1 autonomy list in the GUI, J3 figures restyle, H1 idea list).
- [x] U5. (see the specific items; closed 29 Sep night) Keep docs/REPORT.md, RESEARCH.md and TODO.md current; commit per milestone.
- [x] U6. (Ctrl + wheel zoom, middle/Shift-drag pan, zoom in/out, fit, follow car - tested) (user, 29 Sep) The map cannot be scrolled / zoomed - fixed size is not practical: wheel zoom at cursor, middle-drag / Shift-drag pan, Zoom in/out, Fit all, Follow car toggle (default on), zones/map keep working with clicks. Same for the mini-map if useful.
- [x] U7. ((b) replay now picks up systems as runs finish, (c) checkbox fixed; (a) see U2) (user, 29 Sep) (a) "Intent-aware ADAS is going backwards": in the Monte Carlo replay the ADAS + intent car reverses/loops where plain ADAS drives on - measure retreat distance (away from the goal) of adas+intent vs adas over all drives and fix (forward-first, less back-off after the stall release). (b) The lab replay shows only DRIVER ONLY while a new Monte Carlo is running (the other systems finish later): reload the replay as runs arrive and show pending cards. (c) checkbox checked state is a flat bright-blue fill (T4).
- [x] U8. (plain wheel scrolls the drawer, Ctrl + wheel zooms - tested) (user, 29 Sep) Scrolling over the Map & zones drawer kept zooming the map, so the drawer could not be scrolled: plain wheel scrolls the drawer (also over the Diagnostics plots), Ctrl + wheel zooms the map.

## V. Target set by the user, 29 Sep (late): ADAS + intent must ALWAYS reach the goal and clearly beat plain ADAS
- [~] V1. (29 Sep late - measured, RESEARCH.md section 12: 0 crashes, goals 70/70 vs ADAS on untouched seeds (64 vs 67 on tuning seeds), median time 11.6 vs 12.1 s, interventions -37 %, takeover episodes -43 %, needless -41 %; the 90 % / always-goal target is NOT met - an oracle intent model does not do better, so the limit is arbitration with a dithering driver, not the ML model)  Monte Carlo result to show: ADAS + intent >= 90 % fewer (needless) interventions than plain ADAS, goals reached 100 % (never worse than ADAS),
  time to goal <= plain ADAS, no retreat / stuck states / wasted actions. Report it honestly with paired statistics; if a part cannot be met,
  say which part and why. Work: analyse where the needless interventions come from (by kind and trigger), fix causes generally (not per drive),
  improve the ML model (needless-intervention classifier on Monte Carlo counterfactual labels, more twin data, GPU), re-run the Monte Carlo
  on untouched seeds, update REPORT.md / RESEARCH.md / labs.
- [~] V2. (29 Sep night: N1, N4, F1 (explore, park), N7, N11, D6 done; the rest is Pi/car-dependent or report figures) All remaining TODO items except report figures and Pi-dependent ones.

## W. User request (29 Sep, night): the remaining sim-doable items, plus return to start
- [x] W1. (adas/submaps.py + SlamService, RESEARCH.md section 14; twin 8.0 -> 0.8 cm over a drifting loop) Explain Cartographer-style submaps + loop closure (answer in chat + RESEARCH.md), implement them (adas/submaps.py: scan-to-submap
  matching, pose graph, loop-closure detection, optimisation) and test in the twin with injected odometry drift.
- [x] W2. (HOME with the corrected pose; bare room 110 cm -> 10.8 cm, furnished room neutral (7-8 cm); sim/home_eval.py, tests/test_submaps.py) **Return to start** must work properly: HOME exists (relay + GUI tile) but uses the drifting pose; test it end to end in the twin (drive
  out, come back), measure the error, and use the loop-closed / start-submap-relocalised pose so it returns exactly.
- [x] W3. (adas/park.py find_parallel_slots + straightening strokes; 4/4 twin runs 3-6 cm / 4-8 deg; relay PARK PARALLEL) Parallel parking (slot detection along a row / kerb, parallel pose goal, S-curve reverse with the pose planner), twin test.
- [x] W4. (re-evaluated: no gain even with gate-passing bad measurements; documented, not enabled) Robust EKF: re-evaluate with corruptions that pass the covariance gate; enable only if it helps, else document.
- [ ] W5. Direct 'would this takeover be needless' model: build a dataset of takeover decisions from Monte Carlo counterfactual labels
  (features at trigger time), train (GPU/sklearn), use it as the decider, compare with v2 on untouched seeds.
- [x] W6. (adas/online_calibration.py; ECE -37 % / -28 % prequential on the twin; wired into RelayIntent (shown risk only)) On-the-go adaptation of the crash predictor once a person starts driving: online recalibration of the risk (Platt / Bayesian) from
  outcome feedback the car can observe, per-driver profile; prequential evaluation in the twin.
- [~] W7. (RUN tabs use chips and outlined primary / danger buttons; lab headers flat; progress bars slim and tinted) Restyle the labs' RUN tab and window buttons; finish the training-lab chart polish.
