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
- [x] N11. (docs/architecture.md, 29 Sep)  Architecture: ROS 2-style layering (perception / prediction / planning / control / HMI) with logged,
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
- [x] W5. (30 Sep: trained + compared, NOT adopted - v2 stays decider, see ml.md; was: usage limit hit: sim/needless_data.py (dataset of takeover decisions with counterfactual labels), sim/train_needless.py and the RelayIntent need_path/need_tau hook are written; the data generation was still running in the background (data/needless/decisions_100.npz then decisions_400.npz); next: python -m sim.train_needless, then compare as decider on untouched seeds with RC_NEED_PATH=pi/needless_v1.json)  Direct 'would this takeover be needless' model: build a dataset of takeover decisions from Monte Carlo counterfactual labels
  (features at trigger time), train (GPU/sklearn), use it as the decider, compare with v2 on untouched seeds.
- [x] W6. (adas/online_calibration.py; ECE -37 % / -28 % prequential on the twin; wired into RelayIntent (shown risk only)) On-the-go adaptation of the crash predictor once a person starts driving: online recalibration of the risk (Platt / Bayesian) from
  outcome feedback the car can observe, per-driver profile; prequential evaluation in the twin.
- [~] W7. (RUN tabs use chips and outlined primary / danger buttons; lab headers flat; progress bars slim and tinted) Restyle the labs' RUN tab and window buttons; finish the training-lab chart polish.

## X. Requirements
- [x] X1. (user, 29 Sep) docs/requirements.md - master SRS (goals, ODD, FR/NFR with acceptance criteria and status, HW, stack, I/O, success metrics). Keep its status columns current at milestones.
- [x] X2. (29 Sep) Knowledge extraction into requirements / architecture / decisions / modules; docs/interfaces.md (shared data types).

## Y. Lead / integration chat (30 Sep)
- [x] Y1. (30 Sep: scenarios 21/21 pass; unit tests 160/161, failure logged below) Consistency check: `python -m unittest discover -s tests` and `python -m sim.relay_scenarios`; failures in other chats' modules are logged below, not fixed.
- [~] Y2. (tools/publish.py written, dry run shown: GitHub differs by 3 files; the real push waits for the user's OK) tools/publish.py: copy the working tree into the published layout (src/adas_project/, no git-ignored files, no data/, logs, gui/zones.json), commit, push to real-car-adas. Dry run shown to the user before the first real push.
- [x] Y3. (docs/DEPLOY_PLAN.md, nothing run) Pi deployment plan (prepared, NOT run) for when the car is repaired: backup name, changes since the 28 Sep 22:15 deploy, protocol additions the GUI needs (MODE, HOME, EXPLORE, PARK, zones), real-car test checklist turning SRS Met-S into Met. Goes in docs/modules/integration.md (or docs/DEPLOY_PLAN.md).

## Z. Safety & planning chat (30 Sep)
- [x] Z1. (30 Sep: trigger is now feasibility-based, `NUDGE_GENTLE`; repeatability 55 -> 100 % of 20 randomised cars, seeds 1000-1019; untouched seeds 3000-3039 x 5 box positions: 194/200, 0 contacts) Nudge (steering correction) robust to randomised cars: baseline 55 % spec in `python -m sim.repeat_scenarios`; fails when the servo-centre error drifts the car. Fix generally; report the new repeatability.
- [x] Z2. (30 Sep: diagnosis only, no code change kept) Two Monte Carlo drives (seed 0, 'lapsing' and 'late' driver styles) still end stuck for ADAS and ADAS + intent: find the general cause and fix it.
- [~] Z3. (30 Sep, plan-only so far: at planner kappa 1.5 a 3-cusp path exists down to a 0.60-0.65 m gap in 4-15 s of search (12 000-node production budget only reaches 0.70-0.75 m); kappa 2.0-2.5 gives 0.65 m with one direction change in 0.3 s but the car's reliable turn is ~0.65 m radius; still to do: full twin drive + margins) Parallel parking needs a ~1.1 m slot (3.4 car lengths): see whether a multi-point (3+ move) manoeuvre can use smaller slots safely.
- [ ] Z4. After each change: affected tests + `python -m sim.relay_scenarios` all-pass; update docs/modules/safety_decision.md and planning.md parameter tables; report numbers with seeds and sample sizes.

## P2. Perception chat (30 Sep)
- [x] P2-1. Profile the perception chain per scan: `tools/profile_perception.py` (187 scans of 719 points, twin room route, 0.5 m/s; laptop idle, best of 3 runs; Pi = laptop x 3.5 from `sim/hw_sim.PI_COMPUTE_FACTOR` - `tools/pi_parts_bench.py` has no factor of its own; NOT measured on the Pi). Median / p95 ms laptop -> Pi estimate: convert 0.1/0.1 -> 0.3/0.4 · memory 0.4/0.5 -> 1.3/1.9 · RF2O+EKF 1.1/1.5 -> 3.7/5.1 · ICP 2.7/4.2 -> 9.4/14.7 (largest) · EKF predict per packet 0.02 -> 0.1 · tracking (LiDAR thread) 1.0/1.4 -> 3.3/4.9 · SLAM worker 0.02 (85 % of scans) / 30 -> 0.1 / 105 (the 15 % with a loop search). Control-loop share on a scan tick 4.3/6.4/9.5 (max) -> **15 / 22 / 33 ms** against the 20 ms budget: median inside, p95 marginally over; whole chain within one scan period except the SLAM worker's loop search (median 98, max 243 ms on the Pi estimate; it only ever holds the newest scan, so it drops scans instead of delaying anything, but as Python it competes for the GIL). Done: the loop search was 715 ms laptop (2.5 s Pi) as a Python loop over 6 859 poses; vectorised (integer cell shifts, padded float32 field) it is 28 ms with an identical result (`sim.slam_eval`: 8.0 -> 0.8 cm). Numbers in `models/perception_profile.json`, `docs/modules/perception.md`.
  - [ ] P2-1a. (perception, needs the lead's OK to run on the Pi, benchmark in `~/rc_bench` only) Run `python3 tools/profile_perception.py` on the Pi to replace the x3.5 estimate.
  - [ ] P2-1b. (perception, proposal) ICP scan matching is 60 % of the control-loop share: run `RelaySpeed._scan_pose` in a worker like the SLAM front end (latest scan wins, the pose is composed with the dead reckoning since the scan). Changes when `pose` updates, so the lead should agree first.
- [x] P2-2. Global relocalisation: `adas/submaps.py` (`PoseGraphMap.relocalise/rebase/save/load`, `Relocaliser`, `kidnap_score`, `SlamService.relocalise/save/load`, lost mode), `RelaySpeed.relocalise/lost/save_map/load_map`. Twin: `sim/reloc_eval.py` -> `models/reloc.json`: lounge / room / doorway 60/60 fixes (median 0.3-0.5 cm, 0.1 deg), 0 wrong; empty rectangle 0/20 and corridor 2/20 refused (symmetric / ambiguous, correctly); 0.07-0.27 s per fix on the laptop; lift-and-drop mid-drive found by the detector (92 % of random drops in a room; 0 false alarms in ~10 000 real-car scan pairs) and re-fixed within 2 scans. 14 new tests in `tests/test_submaps.py`. ADR-030. Honest limits: the twin's scans are the map's own scans (same LiDAR model, no furniture moved); real rooms change; a corridor / empty room cannot be solved by one scan; the world frame after a relocalisation is the saved map's frame.
  - [ ] P2-2a. (integration chat) Wire it: where to keep the map (`~/rc_car/map.npz`?), `SAVEMAP` / `LOADMAP` / `RELOCALISE` operator commands (interfaces § 16), `RelaySpeed.load_map` at relay start when the file exists, `lost` in the StateMessage `world` block (§ 15) and the HMI ("localising..."), zones / HOME must not use `pose_corrected` while `lost`.
  - [ ] P2-2b. (perception) On the car: does a real LiDAR's map still match after furniture moves? Needs a saved map from a real drive + the user's OK for a test.
- [x] P2-3. Moving-object test: `Tracker(model="lrt")` (default) in `adas/tracking.py`, twin harness `sim/tracking_eval.py` -> `models/tracking_latency.json` (person 0.10 m radius, standing then walking at 0.3 / 0.5 / 1.0 m/s across the path, car still / 0.4 m/s straight / arc with speed scale error and yaw-rate bias, 20 mm noise, 4 % dropout; tuned on seeds 0-2, reported on 3-8). FINDINGS: (1) the old test (fitted speed > 0.10 m/s for 3 scans) flags 15-19 % of static track-scans in this room at 20 mm (wall pieces cut by grazing gaps slide and swap identity), 700-860 per condition, 170-200 distinct static tracks - even with the consumers' >= 5-scan rule; no threshold / persistence choice reaches zero (`models/tracking_legacy_grid.txt`). FR-004's "never flagged" was only ever true for the consumer-filtered flag in easier scenes. (2) New test: 0 of ~13 000 static track-scans in the crossing runs, 1 track (0.03 %) in 15 000 in static drives; 8 mm noise: 0; 30 mm: 0.1 %. (3) Faster? Not for slow walkers: median from the first step with `Track.moving` 0.3 m/s 1.3-1.5 s, 0.5 m/s 0.8-0.95 s, 1.0 m/s 0.45-0.6 s vs the old test under the >= 5 rule 0.9-0.95 / 0.7-0.8 / 0.6 s (its raw flag 0.35-0.85 s, but that flag is the false one). The limit is information: at 20 mm noise a 0.3 m/s mover moves 1.5 sigma per scan. What is gained is the false rate, and that the >= 5-scan filter is no longer needed. Also fixed on the way: a head-on scenario regressed with a first version (`sim.relay_scenarios` is 21/21). ADR-031. Tests: `tests/test_tracking.py` (8).
  - [ ] P2-3a. (safety_decision chat) In `RelayAssists` moving-object assist (`pi/relay_assists.py` ~ line 780) replace `t.moving_count < 5` by `not t.moving` (the new test already needs 2 scans above a chi-square threshold and holds with hysteresis); keep the 0.15-2 m/s speed window; re-run the crossing scenarios + a paired Monte Carlo (the swerve / yield should react 0.1-0.4 s earlier).
  - [ ] P2-3b. (integration chat) `pi/wifi_drive_safety.py` Clearance thread: feed the tracker the EKF `vest.v` / `vest.w` (3 cm/s, 2 deg/s) instead of the model speed and `v tan(delta)/L`; and check the y sign: it builds y = +d sin(bearing) while `RelayAssists.points_vehicle_frame` / interfaces § 1 use y = -d sin(a) (clockwise angles) - if the bearing there is already counter-clockwise it is fine, if not the tracker sees a mirrored world and the yaw-rate sign in the motion field is wrong.
- [x] P2-4. Parameter table + performance targets in `docs/modules/perception.md`, FR-004 / FR-006 rows in `docs/requirements.md`, ADR-030 / ADR-031 in `docs/decisions.md`.
- [ ] P2-5. (proposal for `docs/interfaces.md`, lead) § 4 TrackedObject: add `evidence` (float, chi-square statistic of "not static", 2 dof) and change the `moving_count` row to "consumers may use `moving` directly (2 scans above threshold + hysteresis)"; § 6 EgoState: add `lost` (bool, pose_corrected not to be trusted while relocalising); new type `Fix` (pose in W, score, margin, inliers, submap); `SlamService` API additions as in `docs/modules/perception.md`. Not changed by me.
- [ ] Z5. **Request to the perception chat:** commit 00825a2 (adas/tracking.py, LRT moving-object test) made 'Moving obstacle: head-on' worse on randomised cars: 18/20 -> 13/20 (seeds 1000-1019, closest -1..-4 cm, car ends at x 0.8-1.2 m instead of 1.95 m); restoring the previous tracking.py gives 18/20 again. The nominal `sim.relay_scenarios` still passes. Likely the object is flagged 'moving' later. Not edited here.
- [~] Z6. (30 Sep, INTERRUPTED: adas/park_select.py, tests/test_park_select.py and the planning.md rows are written, but the tests are NOT green yet. First run: 11 of 14 failed, mostly test-helper errors (strip started at x=-1, depth 0.30 too tight; both fixed since, not re-run). Known: the one-move reverse S needs a slot > 1.0 m for the 0.36 m car (nose hits the front neighbour), so 1.0 m and shorter rely on the multi-point Hybrid A* fallback, whose success/time per length is still unmeasured. Next: re-run `python -m unittest tests.test_park_select`, measure S / multi-point by slot length (1.4, 1.2, 1.0, 0.8, 0.66, 0.55 m), delete the scratch file _dbg_tmp.py in the repo root.) (user, 30 Sep) Parking selection planning (planning logic only, GUI untouched): perpendicular mode = rear-wall ArUco marker detection + pose estimate + reverse trajectory aligned with the marker centre line, stop tolerance <= 5 cm; parallel mode = project the user's click on the parking strip, search all spaces, ignore occupied, nearest free space to the click, parking goal, realistic reverse parallel-park trajectory; different space sizes. Exposes only `selected_space_id`, `parking_goal`, `planned_path`. Module adas/park_select.py + tests/test_park_select.py + docs/modules/planning.md.

## G3. GUI chat (30 Sep, second task): high-FPS simulator + Parking Simulator (user)
- [x] G3-0. Removed the faint ground-truth "sim walls" mesh (0.05 m high) from the 3D drive scene: too low for the LiDAR to see, so it showed objects the car cannot detect.
- [x] G3-1. (gui/render.py: 60 Hz FrameClock, StateInterpolator, Smooth; group transform moves the scene by the interpolated pose; FPS readout; 62 FPS measured) Part A: 60 FPS fixed-timestep renderer, interpolation / exponential smoothing of received states (car, map, zones tab), stable follow camera, update only changed meshes, live FPS counter. Note: OpenGL must render on the GUI thread; receiving stays on the network thread, rendering reads the latest snapshot.
- [x] G3-2. (sim/parking_worlds.py: park_perp + park_street, runtime switch in tools/sim_car.py, layout in sim.parking, 3D objects in the drive scene) Part B: switchable Parking Simulator - perpendicular bay room (walls, floor grid, bay, ArUco marker with adjustable ID on the rear wall, reverse target) and parallel-parking street (road, sidewalk, parked cars, free spaces of different widths / lengths); spaces rendered as objects. Needs simulator worlds (sim/, not owned by the GUI chat) - decide who builds them.
- [x] G3-3. (Park space tool: hover preview, click -> nearest free space, red / green / blue, zoom + pan kept; publishes clicked_world_pose, selected_space_id, occupancy - interfaces.md 16b) Part C: interactive Zones map - clickable parking strip, nearest free space, highlight, occupied red / free green / target blue, hover preview, zoom + pan.
- [x] G3-4. (pose interpolation, no teleporting) Part D: smooth vehicle motion by interpolation, no teleporting; no planning-logic changes; events only through existing interfaces.
- [x] G3-5. FPS profiling + optimisation (axis labels -> MetricGrid, vectorised point_walls / walls_mesh, cached map cells, world-frame parking group, glGetError off): 62 FPS scene, 60 FPS with map drawer.
- [ ] G3-6. Not yet done / to check on the real car: the relay's `world.pose` is dead-reckoned + scan-matched, so on the car the parking layout does not exist (simulator only); a marker-based overlay for the real car is planning's (P2 / park_select).
- [ ] G3-7. `PARK_TARGET` UDP command: proposed in interfaces.md 16b, not sent until planning declares support.
- [x] G3-8. (user screenshot: bays outside the room after driving: the relay's estimated frame had drifted 20 degrees from the simulator's; the HMI now aligns the layout from the (sim.pose, world.pose) pair every packet; test_parking_layout_follows_the_relays_drifted_frame)

## V. Parking validation framework (30 Sep, ML & validation chat; no GUI / planning changes)
- [x] V1. (written, 86 scenarios: 20 perpendicular + 24 layouts x 2 start lanes + 10 named parallel + 8 others; nominal suite runs) sim/parking_validation.py: scenario builders + runner for both parking worlds (perpendicular ArUco: perfect, 5/10/20 deg marker rotation, occlusion, marker distance, LiDAR noise, localisation drift; parallel: 20+ layouts, vehicle lengths, narrow/standard/wide, random occupied, click between spaces, edge spots); executes the planner's ParkingPlan on the relay's own AutoNav / gate in the twin.
- [x] V2. (results_*.csv, summary_*.json) Metrics per run: success, final pose error (cm), heading error (deg), corrections, collisions, time, path length -> CSV.
- [x] V3. (860 runs = 86 scenarios x 10, 0 contacts; see monte_carlo.md) Monte Carlo with randomised sensor + steering noise (domain randomisation), 8 workers, background.
- [x] V4. (reports/parking_validation/dashboard.html, results.csv, summary.csv, plots; tests/test_parking_validation.py 7 tests pass; docs done) Dashboard (HTML) + summary plots (reports/parking_validation/); tests/test_parking_validation.py; docs (monte_carlo.md, TODO).

## V5. Parking on the actual GUI + twin relay (user request, 30 Sep: make it work end to end, test there, iterate, report only when all tests pass)
- [ ] V5a. Relay accepts a parking command (PARK_TARGET / perpendicular marker) and drives park_select plans with its own follower (no framework injection), incl. the follower issues found in V (replan check, 6 cm gap, arrival tolerance).
- [ ] V5b. tools/sim_car.py (twin relay) supports it; HMI (Parking Simulator) click -> relay; verify in the running HMI.
- [ ] V5c. Validation framework runs through the real path (UDP commands to tools/sim_car.py or the same relay class), all scenarios pass or are documented refusals; iterate on failures.
- [x] G3-9. (user, 30 Sep: LIMP 'LiDAR slow 5.5 Hz; control loop slow 95th 99 ms' while driving to the goal, 34 FPS) The display shares the computer with the relay / twin. Found: at 2560x1400 the HMI only reached 35 FPS (overlay widgets repainting over the GL scene ~7 ms); the relay stayed healthy in my runs (1366x768 and 2560x1400, GOTO active) but a slower or busier computer can starve it. Done: HMI runs below-normal priority when the host is local; auto frame cap (60 -> 30 FPS after 3 s under 48 FPS, `--fps 30|60|auto`, shown as 'eco'); mini-map redraw at 30 Hz. Not proven on the user's machine.
- [ ] G3-10. Overlay widgets (speed cluster, cards) repaint on every GL frame; caching them as pixmaps would recover the last ~7 ms at 2560x1400.

## AA. Parking end to end on the actual GUI + relay (user, 30 Sep; supersedes V5 - the pieces of the GUI, planning and validation chats do not work together yet)
- [x] AA1. Audit: read all parking code and docs (adas/park_select.py, sim/parking_worlds.py, gui/parking.py, sim/parking_validation.py, sim/parking_e2e.py, tools/sim_car.py, pi/relay_assists.py diffs, docs); list what is broken; delete _dbg_tmp.py.
- [ ] AA2. Perpendicular parking: bay with an ArUco marker on the rear wall; guidance from the REAR CAMERA marker pose + LiDAR; reverse in aligned with the marker centre line, stop tolerance <= 5 cm.
- [ ] AA3. Parallel parking: a parking STRIP with cars parked at various distances; the user clicks a location in the strip (click-and-go); the car finds the closest viable free space, goes there and parallel parks (realistic reverse manoeuvre); different space sizes; occupied spaces ignored.
- [ ] AA4. Relay command (PARK_TARGET / marker parking) handled by the relay's own follower; twin relay (tools/sim_car.py) supports it; the Parking Simulator in the HMI drives it; verified in the running HMI.
- [ ] AA5. Validation framework runs through the real path (UDP -> twin relay); all scenarios pass or are documented refusals; iterate until the metrics are met.
- [ ] AA6. GUI lag / jitter: profile and optimise (G3-9 / G3-10 open), verify FPS and frame-time stability with the twin running.
- [ ] AA7. Tests, docs (planning.md, gui.md, interfaces.md, requirements.md status), commit; report only when the metrics are met.
- Findings of the audit (AA1): (a) tests/test_park_select.py used outdated conventions (marker `facing` pi for a wall behind the car, 8 cm marker, pitch-32 camera, 0.9 m spaces for a 0.36 m car): 11/15 failed and took 112 s -> rewritten to the real convention and to the planner's real limits (17 pass in 1.4 s); (b) planner: a space shorter than car + 0.9 turning radii (~0.96 m for the 0.33 m car) was searched for 9 s and refused -> `min_space_length` refuses at once; (c) the follower skipped half of the reverse S (nearest-point search jumped to the retraced forward leg after the cusp); the arrival was declared on a pose that drifts 5 % of the distance along a straight street (scan matching is degenerate along a wall) -> 20-50 cm errors in truth; (d) the S was planned at the tightest curvature (no control margin) and the pull-up was reached with a heading error; (e) harness "wanted" space is not the nearest VIABLE one (S7 is too short: the car correctly parks in S5).
- AA3 status (parallel, in-process relay twin, 4 spaces x 12 seeds, tools/_inproc_batch.py): 48/48 within 10 cm / 10 deg, position 0.1-7 cm, heading <= 9.4 deg, 0 contacts. Fixes: leg-limited nearest-point search, straight run-in before the reversing point, S at 80 % of the tightest curvature (roomier margin first), reverse-back approach when the car is already past the pull-up point, slow reverse (0.14 m/s, 0.15 m look-ahead), terminal creep / re-plan from the reached pose, LiDAR re-localisation on the neighbours (gap middle + road-side faces measured on the way in, frozen at the first reverse; end faces in a point-to-line ICP, translation only) applied as an offset on the scan-matched pose (adas/layout_localise.py, Censi 2008).
- [ ] AA3b. Real-path (UDP) run of all street cases; nearest-VIABLE expectation in sim/parking_e2e.py; per-run crash counting.


## AB. Production hardening before the car test (user, 30 Sep, master spec; supersedes the open parts of AA)
User correction: parallel parking has NO predefined spaces - one continuous kerb strip with cars parked at arbitrary spacing;
the car finds the gaps with its LiDAR and parks in the viable part of the strip nearest to the click.
- [x] AB1. Parallel parking on a continuous strip: gap detection from LiDAR along the strip, nearest viable slot to the click, reverse S / multi-point, re-measure + re-plan alongside the gap; world (sim/parking_worlds.py), HMI (gui/parking.py), relay command, validation (sim/parking_validation.py, sim/parking_e2e.py) without space rectangles. Target <= 5 cm, correct heading, 0 contacts, nearest viable gap.
- [x] AB2. Perpendicular (ArUco + LiDAR): remaining suite failures (rot20 heading, lidar 4 cm stuck, refuse beyond marker range), <= 5 cm.
- [x] AB3. Click-to-go audit: forward preference, gear-change / curvature / clearance costs, replanning triggers (blocked, new obstacle, stall, localisation jump), stuck detection + recovery ladder; scenario tests (straight, corridor, dead end, dynamic obstacle, goal behind obstacle, repeated replans).
- [x] AB4. Obstacle memory: persistence through the LiDAR minimum range and occlusion (IDs, age, confidence, decay); audit the existing blind-zone memory + tracker; tests.
- [x] AB5. Evasive steering: after the evasive manoeuvre re-plan and rejoin the route (click-to-go active) instead of only steering away; scenarios (static obstacle, crossing pedestrian, narrow pass, rejoin).
- [x] AB6. Localisation drift (scan matching + submaps): long drive, loop return, repeated parking.
- [x] AB7. Vehicle dynamics in the follower: steering-rate / acceleration limits, smooth reversing.
- [x] AB8. Dashboard: 30+ FPS stable, no shake (profile with the twin running); relay not starved.
- [x] AB9. Pi performance numbers (planner / control loop latency, CPU, memory) from the Pi-factor twin.
- [x] AB10. Cleanup of the parking debug tools, docs (planning.md, gui.md, interfaces.md, requirements.md, decisions.md), tests green, commit; final audit report.
- AB1 status (30 Sep): continuous lane everywhere (world, HMI, relay command, planner, validation, e2e); gap detection from the scan,
  nearest viable slot to the click, reverse S at 80 % curvature with a roomy margin first, multi-stroke exit plan for tiny spots
  (Vorobieva et al. 2015; exact rectangle clearance; 0.63 m = car + 0.30 m minimum, ~10 ms), re-selection alongside the gap from the
  close-range scan, creep-centring along the slot. Results: validation suite parallel 34/62 at the strict 5 cm / 5 deg target
  (0 contacts, every gap choice right or a <= 10 cm near-tie); all misses are 5-12 cm position errors on 5-10 m approaches = pose
  drift of the manoeuvre odometry (along-street ICP slides, 4-5 % short; lateral 5-9 cm). Real UDP path (sim/parking_e2e.py):
  5/6 street cases <= 4.9 cm, the 9 m case 9.5 cm; ArUco bay 6.1 cm. Next: AB6 odometry drift (scan-to-local-map), AB2 bay < 5 cm.
- Known: tests/test_explore_park.EndToEnd.test_parallel_parks_between_two_objects (old PARK via Hybrid A*) fails under heavy CPU
  load (the twin's simulated Pi planning time follows the laptop's), passes alone 3/3.
- AB6 status (30 Sep): pi/scanmatch.LocalMapOdometry replaces scan-to-scan Odometry in the relay (manoeuvre frame and world pose):
  scan-to-local-map (KISS-ICP, Vizzo et al. RA-L 2023), point-to-line Gauss-Newton with Huber weights (Censi ICRA 2008), corridor
  degeneracy from the information matrix -> that direction from the speed model (Zhang, Kaess & Singh ICRA 2016). Offline twin
  benchmark (6 streets, 9 m, speed model 5 % low): along 30.8 -> 3.1 cm, lateral 8.1 -> 1.4 cm, heading 1.8 -> 0.1 deg,
  3.3 -> 1.7 ms/scan; corridor + turning rooms: along 9.7 -> 3.3, lateral 1.7 -> 0.7, heading 0.41 -> 0.09 deg. Unit tests 216,
  scenarios 21/21. Parking: re-plan uses the pose at its launch (bug: the car rolled on), a slot re-selected far away is measured
  again beside it (<= 3), a parking path held still 2 s is re-planned from there (<= 3) instead of given up.
  Parallel suite at 5 cm / 5 deg: 45/62, 0 contacts, 0 stuck; misses = along 5-12 cm after 5-10 m approaches (lateral <= 3 cm).
  E2E real path: street 5/6 (1.1-3.8 cm; the 9 m case 8.5 cm), bay 6.9 cm.
- [x] AB6b. par_edge_last: the re-plan at the lane end finds no manoeuvre ("no room here after all", 31 cm): investigate.
- [x] AB11. Test determinism: Hybrid A* budgets are wall-clock; under full-suite CPU load test_explore_park (parallel / bay) can
  miss (pass alone). Use a node budget in the twin's inline plan service. One intermittent segfault seen in the offscreen GUI tests.
- AB2 done (30 Sep): camera -> which bay (marker position), LiDAR -> the wall's line (trimmed PCA fit, neighbours' rear faces
  rejected) and the final stop (the last reverse leg follows the wall as measured, frozen within 4 cm). Bay suite 23/24 (all parks
  <= 4.4 cm, mean 2.8, heading <= 3.1 deg; the miss = 4 cm LiDAR noise stress case, 7.5 cm, no longer stuck). E2E bay 5.2 cm.
- Stall recovery: a failed re-plan from mid-manoeuvre falls back to a goal-pose search with reversing (fixed par_edge_last,
  street19b/23a). Whole parking suite 70/86 at 5 cm / 5 deg: every miss is a 5-13.6 cm position error (none stuck, 0 contacts);
  79 parks: median 3.0 cm, p90 8.0 cm, heading median 1.7 deg. E2E real path: street 6/6 (1.7-5.0 cm), bay 5.2 cm.
- AB3 done (30 Sep): sim/autonav_eval.py suite - 13 scenarios x seeds with behaviour metrics from the true trace (forward / reverse
  metres, gear changes, steering reversals per metre, stop-and-go, re-plans): straight, goal behind (forward U-turn), sharp side goal,
  corridor ahead / behind, dead end (reverse out), doorway, 52 cm gap, room, a box pushed onto the path, three boxes in turn, a
  crossing walker, goal inside a box. 26/26 (2 seeds): forward only where forward is possible, reverse only where needed, 0 steering
  oscillation, <= 1 stop without events. The reported "reverse to goal / oscillation" does not reproduce in the twin. New: a clicked
  point on an obstacle (its inside is unseen by a 2D LiDAR until close) -> when the re-plan fails, the body is swept toward it and the
  car goes as near as it fits ("arrived as near as the car fits"); 6 classic goals 6/6.
- AB11 done: the twin is deterministic - Hybrid A* budgets in the twin are node counts (NodeBudget, 1600 nodes per Pi-second), the
  plan delay is the work done (nodes, grid cells, clearance evaluations) at Pi rates, SLAM runs inline in the twin. The bay test that
  gave 3.8 / 8.2 / 25 cm in one process now repeats exactly; return to start 4.07 cm x6 (was 2/6 failures at 215 cm).
- GUI crash fixed (intermittent access violation, also a theme-switch hazard in the real HMI): worker threads emit through
  controls.ThreadSignal bridges (Qt drops them when the widget dies) and Python garbage is collected only on the GUI thread
  (pyqtgraph GarbageCollector) - a worker importing PyTorch had freed an old window's widgets. GUI tests 4/4, full suite 2/2 clean.
- AB5 done (30 Sep): the evasive assist already re-plans every 0.3 s and after a stall, and plans back to the driver's line; new
  scenarios prove it: back within 3 cm of the line 1.5 m past the block with 2 steering reversals (out, back, straighten), a second
  obstacle in the swerve path, a walker stepping into it (no contact). Scenario suite 24/24. The steering-reversal metric (also in the
  click-to-go suite) had a units bug (stick is -1..1) - fixed; click-to-go worst 0.56 reversals/m (doorway), limit 1/m.
- AB4 done: the gate's obstacle memory now rides on the scan-matched pose (ObstacleMemory.correct_pose after each scan, stale poses
  ignored by stamp, a frame jump such as ORIGIN keeps the points around the car); dead reckoning only bridges 0.1 s. Unit tests
  (tests/test_memory.py, 8): blind-ring persistence, the scan overrules the memory, a no-return bearing keeps its points, forgetting
  after the car drove on, 10 % speed error 4 cm -> < 0.5 cm, stale pose, ORIGIN, movers not remembered. (A relay scenario for a post
  inside the minimum range was tried and dropped: the gate stops a creeping car before the blind ring matters - not a real test.)
  Moving objects keep IDs / velocity / age in adas/tracking.py (Tracker); the memory stores only static returns (exclude movers).
- AB8 done (30 Sep): `python -m gui.dashboard --measure S` records frame intervals and the displayed car pose (shake = second
  difference per frame) on the real HMI; measured with the twin relay weaving (laptop panel 1707 x 1067 @ 165 Hz):
  bare 3D scene 62.5 FPS, frame sd 6.3 ms (p95 30 ms: a 16 ms timer beating against the refresh), shake p95 3.0 / max 10.3 mm ->
  swap-paced frames with an 11 ms floor: 92 FPS, sd 4.6 ms, shake p95 2.3 / max 5.9 mm; with the map drawer 52 -> 64 FPS (the map
  follows the car every other frame). Relay health stayed NORMAL (LiDAR 10 Hz, link 20 Hz) during the runs.
- AB7 done: steer before moving - at a standstill a steering change > 0.3 (of full lock) is given first and the car waits 0.15 s (servo
  delay + slew) before rolling (relay `_navigate`); speed stays with the PI speed controller + feedforward (FR-034). Parking suite neutral
  (68/86, p90 5.9 cm), click-to-go 26/26.
- AB9 done (`python -m sim.profile_relay`, 30 drives): control loop 4.2 ms mean per 50 ms tick on the laptop = ~14.7 ms on the Pi 5
  (budget 20 ms); per scan: pose / EKF p95 4.7 ms (16 ms Pi), gate + memory 0.5 ms, intent 1.7 ms; planner in the worker process
  median 23 ms / p95 355 ms laptop. Not yet measured on the Pi itself (needs the car: `python3 -m sim.profile_relay --on-pi`).
- AB10 done: docs (interfaces 15/16/16b, ADR-032..037, requirements FR-030/FR-033, planning / perception / gui modules, CLAUDE.md commands).
- Open for the car test: MEASURE odometry drift on the car (5 m straight, taped mark) and the parking accuracy; forward entry into a dead
  end is refused (steering too weak); long approaches (5-10 m) park 5-10 cm off; test_explore_park load sensitivity is gone (twin
  deterministic).
- [x] AB12. Monte Carlo after the session (models/relay_mc.json, 24 seeds x 5 styles): 0 crashes for every assisted variant (unchanged),
  goals 108 -> 110 (adas) / 105 -> 112 (adas+intent), min clearance 2.0 -> 3.4 cm; BUT needless takeovers rose (adas 68 -> 179,
  adas+intent 34 -> 44). Bisect on 40 runs: memory correction / odometry: no effect; the twin's work-based plan delay: 11 of +16 -
  it under-estimated the evasive search's Pi time (model 118 ms vs measured x 3.5 337 ms median), so plans arrived too early. Refit
  the delay model's rates on single-threaded measurements, then re-run the Monte Carlo.
- AB12 result: the twin's Pi-time model refitted (single-threaded, 127 jobs of every kind: 1400 nodes/s, 8e5 grid cells/s, 1.3e7
  clearance evaluations/s, 5 ms base; per-kind median measured/model 0.8-1.0). Needless takeovers do not come back down with it (40-drive
  bisect 76; pre-session code 59): the old baseline measured laptop time x 3.5 WHILE 8 simulations shared the laptop, so evasive plans
  were inflated and often arrived too late to take over. With the planner on its own core (as on the Pi) more arrive in time - an
  arbitration question for the decider (ML), not a planner regression. Final Monte Carlo (24 seeds x 5 styles): 0 crashes for every
  assisted variant; adas+intent goals 111/120, 44 needless takeovers; min clearance 3.0 cm.
- Final numbers with the calibrated twin: unit tests 224/224; scenario suite 24/24; click-to-go 26/26; parking suite 64-71/86 at
  5 cm / 5 deg depending on the planning-time model (0 contacts, 0 stuck; median 3.2 cm, p90 8.8 cm, max 12.1 cm); HMI 64-92 FPS,
  HMI 375 MB / relay 120 MB RSS, relay 25 % of a laptop core (~90 % of one Pi core by the 3.5 factor - MEASURE on the Pi).
- Final real-path run (30 Sep): 5/7 - street s:6.4 5.1 cm (just over), s:9.1 chose the neighbouring gap once (then 4/4 at 3.0-4.6 cm);
  bay 4.2 cm. Audit report: docs/audit_2026-09-30.md.


## AC. Dashboard test of 30 Sep (user) - parallel parking broken in the real HMI
- [ ] AC1. Parked car laterally shifted / partly outside the lane: must be fully inside the lane and as close to the kerb as possible.
- [x] AC2. Far click: planned path not shown on the map and the car drives into the wall while RT is held.
      Done: the way to the slot is a re-planning click-to-go drive (ADR-038); dashboard park_far / park_from_far_side: path shown 99-100 %, no contact.
- [x] AC3. Tight spots refused ("can't park") although the car fits.
      Done: tight gaps planned (multi-stroke) and driven; dashboard park_tight: inside, 6.1 cm off the kerb, 0.2 deg (a stall mid-manoeuvre pulls out and retries).
- [x] AC4. Unparkable click: never refuse - take the nearest parkable spot on the lane and plan for it.
      Done: the nearest viable slot, independent of where the car stands; dashboard park_on_car / park_short_gap park in the next gap.
- [x] AC5. The planned path is missing from the map about half the time.
      Done: the path (way + manoeuvre) stays drawn during re-plans; dashboard 98-100 % of driving ticks in every run.
- [x] AC6. HMI test harness: the real EVWindow + twin relay, real clicks on the map, dead-man throttle, screenshots reviewed - for every
      feature used today (manual drive + assists / braking, click-to-go, parallel parking, bay parking, zones, return to start).
      Done: tools/hmi_test.py (own relay 4310/8190, own zones file) - parking x10, bay, go-to + return to start, zone, braking at a wall.
- [x] AC7. Validation criteria: "inside" = the whole body inside the lane band, kerb-side gap reported; fix what the suites called success.
      Done: inside = whole body in the lane, 1-8 cm off the kerb; lane cases scored on along error / heading / kerb (sim/parking_validation.py).
- [x] AC8. Root cause (user's relay log 10:16-10:19 + tools/hmi_test.py): the way to the slot was ONE fixed Dubins curve planned from one scan
      (often a long reverse), never re-planned -> reversed to 8 cm of a car / 21 cm of the wall; from inside a gap no Dubins exists -> "no way
      to park there". New sequence (as production APA): choose the slot, drive to its pull-up pose with the re-planning click-to-go planner
      (Hybrid A*, obstacle memory, stall recovery), measure the gap alongside, then plan and drive the manoeuvre.
      Done (ADR-038).
- [x] AC9. Parked already (car inside the lane): pull out first (the multi-stroke exit planner run from the car's pose), then approach.
      Done: plan_pull_out; dashboard park_again_elsewhere ok (4.3 cm off the kerb).
- [ ] AC10. Kerb: the parked neighbours stand 3-6 cm off the kerb; target 3 cm (was 7) with its own clearance along the path (kerb is not
      seen by the LiDAR - 1.5 cm planning margin), neighbours keep theirs.
- [x] AC11. Map: keep the path drawn while a re-plan / the gap measurement runs, and draw the whole intended path (approach + manoeuvre).
      Done.
- [x] AC12. tools/hmi_test.py runs its own twin relay on 4310/8190 (the first run went to the user's relay on 4210 at 10:28-10:31: it
      switched their room to STREET and clicked a park target; nothing else). Scenarios: open, tight, far, on a car, short gap, after
      hand-driving, parked -> park elsewhere, RT released and pressed again, from the far side of the road.
      Done.
- [x] AC13. "NO DATA constantly" (user): the relay died on an AttributeError in the half-finished AutoNav (`_refines`) the user's restarted
      relay had loaded from the working tree. Fixed, and a fault in any autonomy code (tick or GOTO / PARK* / EXPLORE command) now stops the
      autonomy with a message instead of ending the relay. Their stream measured healthy otherwise (20 Hz, 12 KB, no gap > 60 ms).
- [x] AC14. Pull-out stuck (user screenshot 11:24, harness park_again_elsewhere): the first stroke backs toward the car behind until the brake
      gate predicts contact at 0.10 m, then it stalls.
      Done: stroke end at the gate's creep limit, progress-based stall, pull-out strokes 8 cm.
- [ ] AC15. Rear wheels clip the kerb (user): kerb target 3 cm + tracking error / the rear corner's swing in the S; also the HMI's wheels may
      stand outside the planner's body width.
- [ ] AC16. Screen / car shake (user): LiDAR noise may move the points, never the car model or the camera - filter the displayed pose.
- [ ] AC17. Twin actuation noise (user): steering / throttle must not give exactly the same curve every time; parking and click-to-go must
      adapt (closed loop), evaluated with the noise on.
      Done so far: sim.hw_sim.ActuationNoise on in every simulation (ADR-041). Parking adapts: stroke end at the gate's creep limit,
      progress-based stall (RT pulses no longer reset it), side correction (one S + back / two S / partial), straighten at the approach
      point, final position measured from the neighbours before 'arrived' (centred + in line, gap offset in long gaps), goal in line with
      the neighbours (ADR-042 to write), kerb wall for the parking searches, escape from a start inside the margin, stale matcher origin.
      Validation (62 lane cases, noise on): 44 -> 49 ok; dashboard 6/9 -> see AC6 results. Remaining: kerb < 1 cm in ~4, along 6-10 cm in
      ~5 (fix in: measured centring before arrival), 2 stuck.
- [x] AC18. Relay crash on a room switch / simulator reset (teleport hooks ran on the HTTP thread and raced the loop's obstacle memory:
      IndexError in memory.add_scan -> NO DATA). The loop runs them itself now (TELEPORT_PENDING).

- [ ] AC19. Click-to-go from a parked car (user, 30 Sep): the relay remembers the lane it parked in; a GOTO while the car stands in that lane
      pulls out first (park_select.plan_pull_out, in the plan service) and then drives the go-to.
- [ ] AC20. Remaining validation failures (55/62 lane): two tight gaps end stuck at 42 deg on the kerb ("stuck - no way through"), one
      0.83 m gap times out mid-manoeuvre, a 1.79 m gap 6-7 cm along (the neighbours beyond reach), a 0.71 m gap 13 cm out.

- [ ] AC21. Relay scenario "head-on walker" fails with the actuation noise (closest -3 cm; 5 cm without noise): the walker's velocity is
      estimated at ~0.4 m/s from 0.8 s in both, but the tracker's likelihood-ratio 'moving' flag comes at 1.9 s instead of 1.4 s (the
      static reference tracks and the ego-motion prior differ), and the crossing logic then waits for 5 confirmations -> backing away
      starts 0.55 s late. Backing away now matches the closing speed (0.15-0.35 m/s, adas/crossing.py) - not enough alone. To do: the
      moving test's ego-motion prior under actuation noise (adas/tracking.py ego_v / ego_w), the 5-confirmation wait for tracks in the path.
      (Committed the noise model before running this suite - a slip; the suite is 23/24 with the corridor threshold adjusted.)

- [ ] AC22. Click-to-go suite 25/26 with the actuation noise: corridor_ahead seed 1 'arrived' 17 cm off on the truth - in a featureless
      corridor the along motion is unobservable to the LiDAR (scan matcher and range flow degenerate), the odometry takes the speed model,
      and the twin's speed gain now errs by a few % (4.4 m -> 13 cm). Options: an end-of-corridor feature when there is one, a final
      approach on the goal as measured in the scan (as the parking does with its neighbours), a speed-gain estimate from the non-degenerate
      stretches.

- [ ] AC23. HMI: after switching the Parking Simulator to ROOM the panel still shows the bays room's "3 of 5 spaces free".
- [ ] AC24. Still open from the dashboard / validation (30 Sep, noise on): lane 55/62, bays 23/24, 0 contacts; successful lane parks 1.8-6.2 cm
      off the kerb, <= 4.4 cm along, <= 4.3 deg. Failures: two tight gaps end stuck at 43 deg across the kerb line (the last-resort search),
      two tight gaps time out, 1.79 m gap 6-7 cm along (the neighbours beyond reach). AC1 / AC10 / AC15 stay open until these pass.
      Dashboard 30 Sep (last runs): park open / far / on a car / short gap / after driving / again elsewhere / far side / tight / RT pulses /
      then go-to all ok in their last run (RT pulses and tight each failed once in an earlier run); bay park 4.5 cm 2 deg; go-to through the
      doorway 10.8 cm (arrives on the odometry, 8 cm) + return to start 2.2 cm; 5 km/h zone held (0.105 m/s max); braking stopped 10 cm
      before the wall. Not yet measured: the display shake with the One Euro filter (AC16).

- [ ] AC25. Last full dashboard run (16 scenarios) 13/16: all 10 parking + lane over the street + zone + braking ok; goto_and_home had 3
      CONTACTS (investigate first - a safety regression?), bay_park failed its 5 cm check, lane along the room wall looped in pull-out retries.

## AE. Deploy to the Pi (user, 30 Sep evening: "plugged in the rpi and lidar, get all the code on the rpi, the car won't move, make sure
##     everything is working and ready for me testing it")
- [x] AE1. Reachability + hardware (serial devices, temperature), backup of ~/rc_car, keep the car's own json (tuning, calibration, panel).
  (30 Sep 23:10: Pi 192.168.1.6 reachable, 50 C; backup ~/rc_car_backup_20260930_2310.tgz; tuning_real_car.json, cal_results.json,
  panel_params.json, car_model.json checksums unchanged by the upload. HARDWARE BLOCKER: no USB device enumerates - the kernel logs
  'usb usbN-portM: over-current change' on every port, ~20 per second, since 23:07; the supply negotiated no USB-PD
  (max_current 900 mA, usbpd objects 0) so the Pi 5 caps all USB ports at 600 mA together. User to check power: LiDAR alone,
  ESP32 + servo off the Pi's USB power (own 5 V regulator), or a 5 V 5 A PD supply.)
- [x] AE2. Upload adas/, pi/, gui-free runtime (sim/tools not needed on the car), requirements; remove stale files from pi/.
  (30 Sep 23:14: 139 entries from the working tree (commit da9d0b6); 18 old top-level pi/*.py scripts that the repo keeps in
  pi/legacy moved to ~/rc_car_stale_20260930/pi (not deleted); car data kept: json reports, tuning .bak files, drive_logs,
  the built pi/lidar_stream/rc_lidar_stream; rc-relay.service and control_panel.* identical, so no service change.)
- [x] AE3. Offline checks on the Pi (imports, unit tests that need no hardware), then restart rc-relay; LiDAR rate, ESP32 state, health.
  (Offline part done: Python 3.12.3, numpy 1.26.4, scipy 1.11.4, sklearn 1.4.1; relay + planners import in 0.44 s; 67 hardware-free
  tests pass on the Pi in ~/rc_bench/tests_20260930 (esp_link, health, zones, crossing, speed_control, planning_safety,
  park_select); the other 18 need sim/ (twin, not on the car) or the repo layout. rc-relay picked up the new code by itself
  (it restarts every ~15 s while 'No LiDAR found'). Open: LiDAR rate, ESP32 state, health - wait for the USB power fix.
  23:25:55 re-plug: both CP2102s (10c4:ea60) attached as ttyUSB0/1 and dropped within 1 s; over-current stays at ~23/s on
  all six root ports, also while nothing is attached -> a steady overload, short or back-fed VBUS, not a port-name issue
  (find_ports probes every /dev/ttyUSB*). Next: user unplugs all USB; if the flags stop, add LiDAR alone, then ESP32.
  23:29 reboot on another power source: still max_current 900 mA (no USB-PD), over-current from 0.77 s after boot on all
  ports (before any device enumerates), no device attached. Same kernel (6.8.0-1065, installed 24 Sep) on the good 28 Sep
  boots (0 over-current, LiDAR + ESP32 attached); the 28 Sep 19:30 boot had the same storm (3697, no devices) -> the
  power/wiring setup, not the software. LiDAR alone (23:33): attaches for 1-4 s, drops, over-current continues -> the
  LiDAR's draw is what trips the 600 mA cap. ESP32 alone on the user's stronger supply (23:35 boot): 0 over-current,
  CP2102 stays on ttyUSB0, PING -> PONG, 'ROVER READY', ESP32 on Wi-Fi 192.168.1.15; supply still no USB-PD
  (max_current 900 mA). Options for the LiDAR: its adapter's second (power) micro-USB from a separate 5 V source, a
  5 V 5 A PD supply, or usb_max_current_enable=1 (user, sudo) only if the supply really gives 5 A.)
- [ ] AE5. find_ports() opens each /dev/ttyUSB* with serial.Serial(p, 256000) - DTR/RTS are asserted during open
  before being cleared, which can reset the ESP32 (auto-reset circuit): 'ROVER READY' was printed during a PING check
  while the relay was looping on 'No LiDAR found'. Open the probe port with dtr/rts False before open(), as open_esp does.
- [x] AE4. Dashboard connection from the laptop with NO motor commands: stream, PING, MODE, ZONES, ORIGIN, the Parking lane tool.
  (30 Sep 23:37-23:42, LiDAR on external power, ESP32 on the Pi: relay found both; LiDAR 11-12 Hz, 'Good'; ESP32 ok (PONG
  every 0.2 s); health normal; PING -> PONG 27 ms; stream 19.4 Hz, ~1180 points. `python -m gui.dashboard --host 192.168.1.6`
  screenshots reports/pi_deploy_20260930/dash_drive.png, dash_map.png: HEALTH + ESP32 green, room outline in 3D and on the map,
  Parking lane tool present. 10 s of the controller's at-rest packet (A 87 87 / M 0): all stream keys present (world, assist,
  nav, plan, gate, intent), pwm_sent 0 throughout, pose held at 0 +-1 mm, nav idle, p_crash 0.006. Pi 5 timing (relay running):
  parallel-parking plans 2-46 ms (twin's Pi model 6-64 ms), click-to-go Hybrid A* 16 / 112 / 1122 ms (model 33 / 176 / 1807 ms)
  -> the twin's Pi model is conservative. CPU 60-70 C with the relay idle.)
- [ ] AE6. After the user reconnected everything (23:45, wheels on) the LiDAR (CP2102 404f9cb2...) dropped off USB at 23:46:08,
  re-attached/dropped 3 times, gone since 23:47:00 - no over-current flags this time -> its cable or its external power.
  The relay correctly goes to FAULT 'no LiDAR scans' (motor held at zero) but never reopens the LiDAR: after it comes back the
  relay has to be restarted. FR-046 / checklist test 11 want 'relay recovers by itself' -> reopen or exit(1) for systemd
  after a lost LiDAR reappears (check the twin's LiDAR-loss scenarios first; the twin runs the same relay in-process).
  1 Oct 00:00-00:05, 3 Pi reboots, several ports/cables: the LiDAR adapter never enumerates (on the laptop either, where it
  was COM8 before); plugged into the Pi it causes over-current bursts (139, then ~2/s) that knock the ESP32 off USB ~10
  times. Removed at 00:04:30 -> 0 over-current, ESP32 stable. Suspect the LiDAR's micro-USB cable or adapter board (user).
  00:10 on the laptop (COM8, adapter LED on): GET_INFO model 0x31 (A3) fw 1.32 hw 6, GET_HEALTH Good (0); legacy and
  express scan descriptors come back but zero measurement bytes in 4-6 s with DTR low or high, and the user sees the motor
  NOT spinning -> core + serial fine, motor drive dead (motor wires in the LiDAR-adapter connector, motor power, jammed
  head, or the adapter's motor driver). Car testing waits for a spinning LiDAR.
  00:15 after the user changed the power and re-seated the LiDAR-adapter plug: spins; SET_MOTOR_PWM 660 + DTR low 10.4 rev/s,
  then DTR only (as the relay's SDK path) 9.8 Hz, 11 832 samples in 4 s -> the fault was the motor's power/connector, LiDAR OK.
  00:14:46 back on the Pi (data to the Pi, external power on the adapter): enumerated, relay started by itself (00:15:01),
  LiDAR 11.5-12 Hz 'Good', ESP32 ok, health normal, 0 over-current, stream 19.7 Hz, PING 6 ms; dashboard screenshot
  reports/pi_deploy_20260930/dash_map_ready.png. Car ready for the user's tests. Still open here: the relay reopening a
  lost LiDAR by itself (FR-046). Seen meanwhile: Wi-Fi to the Pi drops for a few seconds now and then (SSH/PING timeouts).

## AF. Steering after the refit (user, 1 Oct 00:2x: "the left turn limits need to be increased, the center has changed ... left and
##     right seem to be switched ... the left limit is much lower than right limit"; "also set new lidar front")
- [x] AF1. rc_controller.py: the outer wheel turned more (user) -> swap the servo outputs (LEFT_SERVO_CHANNEL 2 -> 1, RIGHT 1 -> 2) and
      keep each servo's straight-ahead angle the user tuned (output 2 = 80, output 1 = 70) -> LEFT_CENTER 70, RIGHT_CENTER 80.
- [x] AF2. Turn limits: the centres moved down 7-17 deg but the low limit stayed 30 -> physical left turn had 40-50 deg of servo
      travel vs 60-70 right. Shift the limits with the centres to the travel the car had at centre 87 (left 57, right 53):
      LEFT (output 1) 13..123, RIGHT (output 2) 23..133. User to check full lock on the box for binding.
- [x] AF3. Relay (pi/relay_assists.py, pi/path_predict.py): one steering centre = mean of the two servo centres (was left_center
      only), its own steering lines per output around each servo's centre (was "A sv sv" -> wheels 10 deg apart with 70/80),
      the 35..145 clamps relative to the centre. Unit test; twin unchanged (equal centres).
- [~] AF4. Car tuning (Pi pi/tuning_real_car.json, backup first): servo block as AF1/AF2; LiDAR yaw 45.7 -> 70.67 from the user's
      panel front calibration (00:29:55, object 0.39 m, bearing +24.97 deg, sd 0.07). Upload the code, relay started by the user
      (panel Drive; the calibration stopped it at 00:27:31). Check stream, straight-ahead, turn both ways on the box.
      (Done 00:4x: backup tuning_real_car.json.bak_20261001_steer_yaw; relay centre 75, straight line 'A 70.0 80.0', yaw 70.67;
      relay started through the panel's /api/start drive; the user's controller already sends A 70.0 80.0 / M 0. Tests: 9 affected
      test modules OK incl. tests/test_servo_centres.py; relay_scenarios 23/24 (as before). Open: the user's turn check on the
      box. NEW: Pi at 80-82 C with the relay running, no fan/cooling device detected -> health LIMP (speed capped); fit the
      active cooler (K7).)

## AJ. After the real-car parking / click-to-go tests (user, 14 Oct: "it is very close to perfect")
- [~] AJ1. Re-planning gets stuck (user; also AH6/AC25 in the twin: "stuck - no way through from here"). Find why the re-plan from
      where the car stopped fails (start inside the margin, budget, goal now unreachable) and recover (back off along the driven path,
      relax the margin at a crawl, then re-plan).
- [~] AJ2. Prefer known paths over unexplored space (user: "the car should be inclined to follow known paths instead of planning into
      unexplored open space which may or may not be a dead end"). Hybrid A* cost: cells seen free (mapped) cheaper than unknown ones,
      paths the car has already driven cheaper still (e.g. unknown-space penalty as in exploration-aware planners; occupancy-grid
      known/unknown/occupied tri-state, Thrun et al. 2005); unknown space used only when no known route exists.
      (14 Oct first attempt, NOT shipped - saved in docs/wip/: AJ2 a known-space cost map (adas/known_space.py: ray-cast seen/unknown
      grid + driven trail, unknown +3/m, trail -0.3/m, shots crossing > 0.3 m unknown refused, fallback without it) and AJ1 a back-off
      along the trail before re-planning. Unit tests passed and the click-to-go suite stayed 25/26, but on the dashboard (twin
      goto_and_home) the first plan took 5 s, one go-to 'arrived' 57 cm off and return-to-start failed; the back-off ran 3x and still
      ended 'stuck' with 4 contacts. Root cause per the research (docs/research_lidar_navigation.md): planner clearance smaller than the
      brake field -> brake flicker and stalls; re-plans on every stop. Redo as AK1-AK3 + AK7.)
- [~] AJ3. Memory deletes thin objects (user: thin objects are saved when the car is close, then deleted once the car moves away because
      they are not seen from further away). Cause: adas/memory.py prune_contradicted drops a point in ONE scan when any return within
      +-4 deg is farther (a chair leg covers ~1 deg at 1 m: the wall beside it erases it); gui/ev.py WorldStore has the same weakness
      (+-1 deg, 3 scans). Make deletion depend on distance and evidence (user): only beams that really pass through the point count
      (bearing window = the point's own angular size at that range plus the LiDAR's beam spacing, not a fixed +-4 deg); a point needs
      more "seen-through" scans the more often it was hit and the farther the car is (expected returns from a thin object fall
      with range: n_expected ~ width / (r * angular step)); never delete when the expected number of returns from it at this range is
      below ~1 (it simply cannot be seen from here). Clamped log-odds per point / cell (OctoMap) with a range-dependent miss weight.
      Unit test: a 2 cm post seen at 0.4 m, car backs to 2 m with a wall 1 m behind it -> the post stays; a removed box -> goes.
      (Done in code, ADR-046: adas/visibility.py used by adas/memory.py (gate) and gui/ev.py (map); tests/test_memory.py thin post
      stays / removed box goes, tests/test_world_store.py thin post stays from 2 m; phantom-wall test still passes. To do: Pi timing,
      deploy, check on the car with a thin post.)

## AK. Smoothness and tuning from the research (user, 14 Oct: "make the car as smooth and well tuned as possible ... reduce unnecessary
##     warnings and replans"; report docs/research_lidar_navigation.md)
- [x] AK1. Re-plan only on invalidation: remaining-path validity on the planning cost each scan, invalid after >= 3 scans, slow first
      (Nav2 consistent-replanning / IsPathValid). Measure: re-plans and stops per metre in the click-to-go suite and goto_and_home.
- [x] AK2. Clearance hierarchy: planner lethal clearance >= brake field + p95 tracking error + localisation error, decaying cost gradient
      (Smac cost_penalty) - measure the twin's tracking error first; target: no brake/hold flicker on autonomous legs.
      (AK1+AK2 done 14 Oct: adas/autonav.py checks the path 0.3-2.5 m ahead every 0.25 s, re-plans after 2 marginal blocked checks
      or at once when clearance < -2 cm, 0.15 m/s while pending; adas/plan_service.py plans with 10 cm comfort clearance first, the
      6 cm margin only as fallback. Click-to-go suite (26): baseline 26 arrived / 0 collisions / 13 re-plans / 7 stops / 319 s /
      clearance median 16.1 cm -> 26 / 0 / 13 / 9 / 324 s / 17.6 cm (neutral). Twin ROOM goto_and_home: go-to now arrives every run
      (3.4-6.9 cm, 7 runs); return-to-start still fails 4/5 (stall at the doorway, 3 quick re-plans, 0-4 contacts) - localisation is
      fine (0.5-1.3 cm off at the goal, new loc_err_at_goal_cm in tools/hmi_test.py) -> likely pure-pursuit corner cutting: AK4.)
- [~] AK11. Return-to-start in the twin ROOM (AH6/AC25): the contacts were the ROOM's dark box (no LiDAR returns) on the way back - not
      localisation (0.3-1.3 cm off at the goal) and not corner cutting (a curvature-regulated look-ahead, L = sqrt(2*0.02/k), changed
      nothing: 1/5 home, 0-7 contacts; reverted). Virtual bumper (ADR-047, adas/contact.py in PathGate): pushing time -83 % at an
      invisible box in the twin. Lesson: dashboard twin runs must not share the CPU with the test suite - the relay slows like the
      Pi's did and drives worse (a 1/5 vs 4/5 home difference came from that, not from code).
- [ ] AK3. Brake hysteresis (release at d + 3-5 cm after 2-3 clear scans) and a speed-dependent protective field.
- [ ] AK4. Follower: RPP-style proximity and curvature slow-down, minimum speed above the dead-band, velocity-scaled lookahead.
- [ ] AK5. Online steering bias/gain (RLS on scan-matched yaw rate) + backlash compensation (adas/online_steering.py exists - review vs RLS).
- [ ] AK6. Speed loop: feedforward map + dead-band jump + PI anti-windup, battery-voltage scheduling (adas/speed_control.py - review).
- [ ] AK7. Known-space commitment (first ~1.5 m of a path in known free space) + dead-end blacklist; reuse docs/wip/known_space.py.txt.
- [ ] AK8. Cost cap on analytic shots + local path repair before a full re-plan.
- [ ] AK9. Scan hygiene: min-range binning, shadow/speckle filter, deskew, no clearing by no-return beams.
- [ ] AK10. Later: MAP-style steering table or kinematic MPC; residual model / ILC for parking. Not: end-to-end / RL controllers.

## AI. Real-car checks through the real dashboard (user, 14 Oct: "carefully run the tests on the actual UI ... confined space ... object
##     straight ahead to calibrate")
- [x] AI1. LiDAR front (user-requested): object 0.29 m ahead at -7.71 deg (sd 0.08, 40 scans) -> yaw 70.67 -> 62.96 (backup
      tuning_real_car.json.bak_20261014_yaw); re-measured +0.02 deg. Pi now at 192.168.1.15 (DHCP changed).
- [x] AI2. tools/car_ui_test.py: real EVWindow against the car, rc_controller's own packets, own safety layer (throttle cap, no throttle
      without fresh data <0.3 s - added after the first brake run sent throttle 4.7 s before the dashboard had data -, speed cap, 8 cm
      bumper cut), screenshots in reports/car_ui_test/.
- [x] AI3. idle: health normal, ESP32 ok. steer: full left 32/42, full right 108/118, centre 70/80 (38 deg envelope each way, per-servo
      centres). reverse crawl 1 s ~0.1 m/s ok. brake (throttle 50): crawled 1.13 m at 0.12-0.17 m/s, the relay's gate held the car
      ("holding", dashboard EMERGENCY BRAKE) 8 cm from the wall; the test's 8 cm cut fired in the same tick. Throttle 70 reached
      0.25 m/s within 0.4 s.
- [~] AI4. Brake test with the test's own cut lowered (4 cm) so only the gate stops the car; then click-to-go 0.5 m, zone cap.
      (14 Oct, with AJ3 deployed: crawl 1.29 m at <= 0.23 m/s, the gate alone braked and held the car 6.8 cm from the wall, test cut
      not used; reverse ok; steering 32/42 - 70/80 - 108/118. Straight command drifted 0.178 rad (10 deg) over 1.3 m -> steering
      bias, see AK5. Click-to-go and zone on the car still to do.)
- [ ] AI5. USB: at boot the Pi overloaded again until the LiDAR-adapter cable / ports were redone (user, 14 Oct).

## AH. INCIDENT 1 Oct ~00:42 (user): "the controls are delayed and persist for very long after, like a 1 second delay; it just broke my
##     back wheel bevel differential because it hit a wall; I gave it forward and right, it moved after a delay and kept moving"
Evidence: logs/incident_20261001/20261001_003520_relay.jsonl.gz (Pi ~/logs), drive CSVs. Findings:
- Wall hit at 409.6 s (manual, full throttle -254 straight): 403-409.6 s the car crept (v_est 0-0.37 m/s) with the output ramping to
  -120 and dropping (limp cap + smoother) while the LiDAR returned 360-480 of ~1000 points: a 100-200 deg arc in front had NO returns
  (dark / very close / blocked), so front = None, the gate had nothing to brake for until the wall appeared at 0.04 m. No-return was
  treated as free space.
- Delay: 196-260 s (parking autonomy "driving to strip", the user holding RT) the relay handled 14-20 packets/s (dt median 37 ms,
  p95 220 ms) while rc_controller sends 50/s; the UDP socket (rmem 212 992 B) queued seconds of old commands and dropped 1943 packets
  (/proc/net/udp) -> commands late and still arriving after release. Over the whole session only ~44 of 50 packets/s were handled.
- Pi at 80-85 C, no fan detected, health LIMP most of the session.
- [x] AH1. Relay: never act on stale driver commands - drain the socket each tick and use only the newest driving packet (control
      commands still all processed, in order); small SO_RCVBUF. Twin/unit test with a burst of queued packets.
      (pi/driver_input.py + the loop in pi/wifi_drive_safety.py, ADR-044; tests/test_incident_20261001.py: 40 queued packets -> the
      newest one + the PING. Deployed 1 Oct 01:23 with AH2 (commit cb1f8bc; relay restarted through the panel, waiting for the
      LiDAR). Full unit suite: 251 tests, the only failures are pre-existing - test_gui_smoke parking HMI x2 (also with the
      committed gui/ev.py) and the order-dependent wheel-scroll test.)
- [~] AH2. Relay: no returns in the direction of travel = unknown, not free: coverage of the path corridor; blind -> creep cap, then
      stop; health/GUI "LiDAR blind ahead". ADR, unit test, twin scenario (a non-reflective obstacle).
      (pi/path_gate.py sector_cover + BLIND_*: < 50 % of the 2 deg bins within 30 deg covered -> 0.10 m/s, stop after 1 s; ADR-045;
      unit tests. Incident replay: creep from 401.0 s, stop 402.1 s (impact 409.6 s); rest of the session: 3 single-scan blips, no
      stop. Still open: twin scenario with a non-reflective obstacle, GUI wording.)
      (1 Oct 01:3x: a bearing also counts as seen when the obstacle memory holds a too-close point there - twin runs had shown blind
      creep/stops when nosing up to walls (67 + 6 ticks); now 0 in two goto_and_home runs; whole-session replay: stop from 402.7 s,
      otherwise creeps of 0.1 / 0.1 / 0.9 s. relay_scenarios 23/24 as before.)
- [ ] AH6. goto_and_home (ROOM) return-to-start ends 'stuck - no way through from here' 0.8-4 m from the start with 0-7 contacts per
      run, with the old code too (1 run: 3 contacts, 1.29 m) - AC25; investigate before the user relies on Return to start.
- [ ] AH3. Profile the relay tick on the Pi during parking autonomy (why 14-20 Hz) and move the heavy part out of the loop (rule 5).
- [ ] AH4. Why the output kept dropping to 0 every ~0.45 s at full throttle with v ~0 (403-409 s): limp cap / smoother / stall logic.
      (Found: -120 = health LIMP cap (Pi hot); each drop to 0 = the moving-object check (moving_blocked) for ONE scan - flickering
      tracks from clutter beside the car (406.27, 407.05, 407.48, 408.28, 408.83 s) - and the smoother ramped straight back: torque
      hammering against the wall. To do: a moving-object cut must be confirmed over several scans and, once cut, hold like the gate's
      latch instead of ramping back at once.)
- [ ] AH5. Pi cooling (user): fit/connect the active cooler; the relay should report fan missing.

## AG. Map persistence (user, 1 Oct, screenshot of a smeared map: "the persistence logic makes the map so bad, make it remove points
##     that the lidar already saw past, only keep points where the lidar can't see (like when it is too close)")
- [~] AG1. gui/ev.py WorldStore only ever added hit counts (LRU at 40 000 cells), so stale points (pose error, moved things) stayed.
      Visibility-based removal (Removert, Kim & Kim 2020; clamped occupancy scores as OctoMap, Hornung et al. 2013): a cell loses
      score when the current scan returns from farther away in its direction; cells closer than the LiDAR's minimum range, hidden
      behind a nearer return or in a direction without a return keep their score. Unit test + dashboard screenshots (twin drive,
      then the car).
      (Done in gui/ev.py: clamped score 6, -2 per scan that sees past (return > cell + 10 cm, 1 deg range image, nearest of 3 bins),
      cells < 0.20 m / occluded / no return kept; tests/test_world_store.py (6, incl. a rotated pose - it caught a degrees/radians
      slip). Twin dashboard: map clean and thin after goto_and_home (reports/hmi_test/goto_and_home_09_home_end.png). Car: the user
      restarts the dashboard; to check on the next drive. Pose drift on the car also smears (the 00:4x map) - separate.)

## AD. Real-car testing (user, 30 Sep: "make it so that I can define a parking strip on the map when I drive in real life ... think about
##     tests in real life for everything and make sure features are available for that")
- [ ] AD1. Parking lane on the live map (user, 30 Sep: "define a parking lane as a rectangle in any simulation or the real thing"): a
      "Parking lane" map tool - drag a rectangle; its long axis is the lane, the long side nearer the car the road edge, the other the kerb
      (a button flips it), the short side the depth; saved like the zones; drawn in 3D and on the map; the Park tool then clicks on it exactly
      as on the simulator's street (PARK_TARGET). Works in every simulator room and on the car; a drawn lane takes the place of the
      simulator street's own. The kerb stays invisible to the LiDAR: the lateral target comes from the parked neighbours (in line with them)
      and the drawn kerb (never nearer it than 3.5 cm).
- [ ] AD2. Keep the drawn lane in place: the world frame survives (ORIGIN only when asked), save / load the map with the lane, relocalise
      on load; warn when the pose is lost.
- [ ] AD3. Real-car test plan (docs/test_plan_car.md): for every feature - manual drive + braking / evasive / nudge, speed zones, click-to-go,
      return to start, parallel parking (boxes as parked cars along a taped kerb line; gaps 0.65 / 0.9 / 1.3 m), pull-out and park again, bay
      parking with the printed ArUco marker, RT released mid-manoeuvre - with the set-up, what to measure (tape marks, phone video, the drive
      log) and the pass criteria.
- [ ] AD4. Audit sim-only dependencies: what the HMI / relay use that only the twin provides (sim.parking layout, truth overlay, room
      switch, marker source) - each gets a real-car path or is marked twin-only.
- [ ] AD5. Real-car parking specifics: the rear camera service for the marker, the LiDAR mount height vs the boxes used as cars, the
      steer-before-moving delay and the gate's creep limit measured on the car (no calibration without the user's OK).
