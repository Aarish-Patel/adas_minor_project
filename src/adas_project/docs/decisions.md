# Engineering Decision Log (ADR)

| | |
|---|---|
| Document | ADR-LOG-ADAS-001 |
| Version | 1.0 |
| Date | 29 September 2026 |
| Related | `docs/requirements.md` (what), `docs/architecture.md` (structure), `docs/RESEARCH.md` (methods and references) |

Each record keeps the reasoning behind one design decision: the situation, the choice, and what it costs. Records are never deleted;
a changed decision gets a new record and the old one is marked **Superseded**. Rejected alternatives that were tried and measured
are recorded too, so they are not tried again without new evidence.

| ID | Title | Status |
|---|---|---|
| ADR-001 | One 2D LiDAR as the primary sensor | Accepted |
| ADR-002 | Minimal sensor architecture on a Pi + ESP32 split | Accepted |
| ADR-003 | The driver keeps control; the ADAS intervenes only when needed | Accepted |
| ADR-004 | Physics has the last word: the path gate is never overridden | Accepted |
| ADR-005 | Learned models may only withhold interventions | Accepted |
| ADR-006 | Simulation-first development on a digital twin running the car's own code | Accepted |
| ADR-007 | Monte Carlo with counterfactual ground truth instead of deterministic validation alone | Accepted |
| ADR-008 | Twin data with domain randomisation and counterfactual labels as the training set | Accepted |
| ADR-009 | Gradient-boosted trees in NumPy on the car; PyTorch only for training | Accepted |
| ADR-010 | v2 intent model decides takeovers; v3 supplies the displayed risk | Accepted |
| ADR-011 | Vehicle frame at the rear axle, x ahead, y left, SI units | Accepted |
| ADR-012 | Ego-motion from LiDAR scan matching fused with the throttle model | Accepted |
| ADR-013 | Object tracking: cluster association with a strict "moving" test | Accepted |
| ADR-014 | Hybrid A* for evasive steering and autonomy, forward-first | Accepted |
| ADR-015 | Least-intrusive escalation ladder | Accepted |
| ADR-016 | Qt (PySide6) native HMI instead of a web dashboard | Accepted (supersedes the web viewer) |
| ADR-017 | EV-cockpit HMI design system with dark and light themes | Accepted |
| ADR-018 | No ROS; a single relay process with ROS-mappable interfaces | Accepted |
| ADR-019 | Modularity rules: hardware-free core, heavy work off the control loop | Accepted |
| ADR-020 | Calibrations only on explicit request | Accepted |
| ADR-021 | Loop-closure SLAM (Cartographer structure) for the world pose | Accepted |
| ADR-022 | Online recalibration of the displayed risk, not of the decision | Accepted |
| ADR-023 | Rejected: robust adaptive EKF | Rejected |
| ADR-024 | Rejected: v3 model or a stricter defer rule as the takeover decider | Rejected |
| ADR-025 | Working agreements for development sessions | Accepted |
| ADR-026 | Rear camera: comfort functions only, warnings never brake | Accepted |
| ADR-027 | Fix failures generally, never for the one reported case | Accepted |
| ADR-028 | Repository publishing: branch `real-car-adas`, project under `src/adas_project/` | Accepted |
| ADR-029 | Scale conventions: 1:14, full-size km/h in the HMI | Accepted |

---

# ADR-001 One 2D LiDAR as the primary sensor

**Status:** Accepted

**Context.** Production ADAS fuses cameras, radars, ultrasonics and sometimes LiDAR, with large compute. The project platform is a
1:14 RC car with a Raspberry Pi 5 and a small budget. The project claim is that driver-intent awareness, not sensor count, is what
removes needless interventions.

**Decision.** A single 360° 2D LiDAR (RPLIDAR A3) is the only sensor the safety functions depend on. Speed, yaw rate and pose are
derived from it (fused with the throttle model). A rear camera is optional and adds comfort functions only (reversing aids).

**Consequences.**
- One sensor to calibrate, time-stamp and model in the twin; a clear, testable claim.
- 360° coverage in one plane, direct range, independent of lighting.
- Blind to objects below or above the scan plane, glass, mirrors and very dark surfaces (documented as outside the ODD).
- No classification of object type; behaviour is geometric.
- A LiDAR failure is a total perception failure, so the health monitor treats it as a fault (safe stop).

# ADR-002 Minimal sensor architecture on a Pi + ESP32 split

**Status:** Accepted

**Context.** The ADAS needs a general-purpose computer for planning and ML, and deterministic PWM generation that keeps working if that
computer stalls.

**Decision.** Raspberry Pi 5 runs perception, decision and logging; an ESP32 generates servo and ESC PWM, receives commands over serial
and stops the motor by itself if no valid command arrives for 500 ms. The laptop is the operator station (HMI, driver input, twin,
training) and is never in the safety path.

**Consequences.**
- An independent last line of defence that does not depend on Linux timing.
- The serial link needs supervision (PING, reboot detection, re-open) - added after a dead ESP32 on 28 Sep.
- Two firmware targets to maintain.

# ADR-003 The driver keeps control; the ADAS intervenes only when needed

**Status:** Accepted

**Context.** An assist that stops or steers for everything is switched off by drivers. The user asked repeatedly that normal and even
sloppy driving must not be slowed or steered.

**Decision.** The system is a driver assistant, not an autopilot. It intervenes only when the driver, left alone, would come into (near)
contact. "Needed" is defined by a counterfactual: the same driver, continuing without the ADAS for 2 s, comes within 2 cm of an
obstacle. All autonomous modes require the operator to hold the throttle (dead-man switch).

**Consequences.**
- A measurable definition of a needless intervention, used in validation (ADR-007) and training (ADR-008).
- The definition depends on a driver model in simulation; on the real car it can only be approximated.
- Some interventions are always needed (a driver who cannot free the car), which bounds how far needless interventions can fall.

# ADR-004 Physics has the last word: the path gate is never overridden

**Status:** Accepted

**Context.** Assists, planners and learned models all propose commands. One of them being wrong must not cause a collision.

**Decision.** Every command passes last through the path gate, which brakes, holds or caps the throttle when the free distance along the
predicted swept path is shorter than the stopping distance (reaction + braking at the measured deceleration, with a safety factor).
Margins grow with measured scan latency and speed uncertainty. Only the explicit, visible ADAS override disables it.

**Consequences.**
- A single, testable safety argument, independent of the ML.
- The gate's margin causes some needless speed caps for aggressive drivers; it can only be tightened with more braking data from the car.
- Comfort filters (throttle smoothing) must never delay a gate cut - enforced and tested.

# ADR-005 Learned models may only withhold interventions

**Status:** Accepted

**Context.** A learned driver model could be wrong in situations it has not seen.

**Decision.** The intent model may hold back a steering takeover (trust an attentive driver) and changes what is displayed; it never
commands the actuators and never suppresses the brake. A physics floor puts a lower bound on the learned risk (driver inactive while
the free way is shorter than the stopping distance). Trust is withdrawn when the car makes no progress.

**Consequences.**
- The worst case of a wrong model is an unnecessary or a late *takeover*, never a missed brake.
- The model cannot produce savings the gate would not allow; the gain is limited to takeovers.

# ADR-006 Simulation-first development on a digital twin running the car's own code

**Status:** Accepted

**Context.** The car is fragile (a dead ESP32, a broken rear shaft), tests are slow and not repeatable, and dangerous cases cannot be
driven on purpose.

**Decision.** Every feature is developed and verified first on a digital twin: a vehicle model fitted to real drive logs, a simulated
LiDAR with the real noise, dropout and latency, and **the exact relay, gate, assist and intent classes the car runs**. Only sensor and
actuator adapters differ. Real logs can be replayed through the current code. When the car is available, features go on the car.

**Consequences.**
- Fast iteration and thousands of repeatable drives; car time is spent on confirmation.
- Results are "verified in the twin" until repeated on the car (status Met-S in the SRS).
- The twin must be kept honest: identified from logs, compared against logs, and randomised (ADR-008).

# ADR-007 Monte Carlo with counterfactual ground truth instead of deterministic validation alone

**Status:** Accepted

**Context.** A fixed scenario suite checks known cases but cannot say how often a system intervenes needlessly, nor compare two systems
fairly. Single runs differ by chance (planner timing enters the simulated Pi delay).

**Decision.** Validation has three layers: (1) a deterministic scenario suite with pass/fail specifications; (2) repeatability on
randomised twin cars; (3) a Monte Carlo in which driver-only, brake-only, ADAS and ADAS + intent drive **identical** randomised rooms with
identical drivers and attention lapses, every intervention judged by the counterfactual, compared with **paired** statistics (Wilcoxon),
tuned on some seeds and confirmed on untouched ones. An oracle intent model gives the upper bound.

**Consequences.**
- Claims come with sample sizes and p-values; tuning and confirmation data are separated.
- Runs are expensive (tens of minutes); differences of a few drives are treated as noise.
- The result depends on the driver models; they are part of the documented assumptions.

# ADR-008 Twin data with domain randomisation and counterfactual labels as the training set

**Status:** Accepted

**Context.** The first intent model predicted poorly in general, not only in the "full speed into a wall" case the user reported. Real
data is scarce and has no labels for "would have crashed".

**Decision.** Training data is generated on the twin: many rooms and scripted situations, seven driver styles, reactions at random
distances; every drive on a newly randomised car (speed, braking, delays, steering gain, LiDAR noise, jitter, mount error - 15
parameters). Labels come from letting the drive continue without ADAS (contact or near miss within 2 s). Evaluation is on held-out
drives, sliced by situation, plus an ablation on cars outside the training ranges.

**Consequences.**
- Held-out AP rose from 0.42 to 0.86; on unseen wider cars the randomised model kept AP 0.85 where the nominal one dropped to 0.63.
- The model knows only what the twin's drivers do; retraining on real logs is planned.
- Dataset size and balance are controlled explicitly (dangerous moments over-sampled).

# ADR-009 Gradient-boosted trees in NumPy on the car; PyTorch only for training

**Status:** Accepted

**Context.** The model must run in a few milliseconds on the Pi with no GPU and without heavy runtime dependencies. Several model
types were trained on the laptop GPU (MLPs, GRU, trees).

**Decision.** Training uses scikit-learn and PyTorch (CUDA) on the laptop; the deployed model is exported to JSON and evaluated in
plain NumPy on the car. Gradient-boosted trees were chosen for v3 (best held-out AP, calibrated by temperature scaling). Features are
cast to float32 exactly as in training.

**Consequences.**
- No PyTorch or scikit-learn on the car; inference well below 2 ms.
- An exporter per model type must be kept in sync; a float32/float64 mismatch once flipped 14/400 decisions - now tested.

# ADR-010 v2 intent model decides takeovers; v3 supplies the displayed risk

**Status:** Accepted

**Context.** v3 (twin-trained) predicts crash risk much better than v2. Better prediction was expected to reduce needless takeovers.

**Decision.** Keep v2 as the takeover decider and use v3 for the risk shown to the driver and the warnings. Decided on a 96-drive paired
confirmation on untouched seeds: no v3 configuration beat v2 with significance.

**Consequences.**
- Two models on the car; each used where it was measured to be best.
- Documented finding: predicting the crash is not the bottleneck; arbitration with the driver is (see ADR-024, oracle result).

# ADR-011 Vehicle frame at the rear axle, x ahead, y left, SI units

**Status:** Accepted

**Context.** Kinematic (bicycle / Ackermann) models are simplest about the rear axle; the LiDAR reports clockwise angles; the servo has its
own convention. Mixed conventions caused sign errors early in the project ("moving backwards and to the right instead of forward and left").

**Decision.** All algorithms use the vehicle frame with the origin at the rear-axle centre, x ahead, y left, angles and curvature positive
counter-clockwise (left), SI units (m, s, rad). Conversions happen only at the edges: LiDAR (clockwise degrees from ahead, 0.12 m ahead of
the axle), servo degrees, PWM, and the HMI (degrees, full-size km/h = m/s × 14 × 3.6). World frame = pose at start or last origin reset.

**Consequences.**
- One convention inside the core; sign bugs are caught at a small number of adapters, each unit-tested.
- Frames map directly onto ROS REP-103 (base_link, odom, map) if ported.

# ADR-012 Ego-motion from LiDAR scan matching fused with the throttle model

**Status:** Accepted

**Context.** There are no wheel encoders or IMU. A throttle-only speed estimate is blind to a sagging battery or carpet; range-flow
odometry had a yaw-rate bias; integrating the EKF drifted 7-11 %.

**Decision.** Speed and yaw rate from an EKF fusing LiDAR range flow with the fitted throttle model; world pose from keyframed ICP scan
matching; drift corrected by submaps and loop closure (ADR-021). For braking, the more conservative of the model and EKF speeds is used.
Scan latency and speed uncertainty are estimated online and widen the gate's margins.

**Consequences.**
- Pose drift 1.6-1.8 %; speed RMSE 3-4 cm/s on real logs.
- Bare corridors and rooms are ambiguous for scan matching; handled by predicted-distance fallbacks and loop closure.
- Adding encoders / IMU later is an EKF extension, not a redesign.

# ADR-013 Object tracking: cluster association with a strict "moving" test

**Status:** Accepted

**Context.** With a 2D LiDAR, objects are clusters of points. Range noise (~20 mm) made static objects look like they were moving,
which caused needless brake pulses and swerve flips in real logs.

**Decision.** Nearest-neighbour association of clusters across scans with smoothed velocities; an object is "moving" only after at least
5 consistent scans at 0.15-2 m/s. Closing speed is used by the gate only for tracks flagged moving. Static geometry goes through the
obstacle memory, not the tracker. A chosen action for a moving object (e.g. a swerve) is held for 1.4 s instead of re-decided each tick.

**Consequences.**
- Real-log replay: reverse brake pulses 53 -> 3; no flip-flopping.
- A genuinely moving object is recognised about 0.5 s late; the gate still protects in the meantime.
- No object classes (see ADR-001).

# ADR-014 Hybrid A* for evasive steering and autonomy, forward-first

**Status:** Accepted

**Context.** A hand-shaped offset-lattice planner failed in the doorway scenario; paths must be drivable by an Ackermann car with a
speed-dependent steering limit.

**Decision.** Hybrid A* (Dolgov et al. 2010) with Dubins analytic expansion for goal poses, a forward-only search first and reversing
at 7× cost, loops over 270° refused, running asynchronously with a time budget; MPPI as local fallback; parking searches with reversing
from the start. Paths are tracked with pure pursuit and closed-loop speed.

**Consequences.**
- One planner for evasive steering, click-to-go, return to start, exploration and parking.
- Planning takes 1-440 ms, so it must never run in the control loop (ADR-019).
- Tight slots need 3+ car lengths with the current margins.

# ADR-015 Least-intrusive escalation ladder

**Status:** Accepted

**Context.** Braking is the most noticeable and least helpful intervention when a small steering change would do.

**Decision.** Responses escalate: warning -> speed limit (zones, drive mode, steering envelope) -> steering correction -> evasive
manoeuvre -> brake/hold. A manoeuvre is committed (not dropped for a lifted pedal or a twitch) but the driver can always take over by
steering against it or releasing the throttle. After a real override, the system stays back for 6 s unless the brake's envelope is reached.

**Consequences.**
- Fewer brakes and fewer repeated takeover attempts; needless takeover episodes -43 % against plain ADAS.
- More states and rules to test; each has its own scenario.

# ADR-016 Qt (PySide6) native HMI instead of a web dashboard

**Status:** Accepted - supersedes the browser viewer (three.js), which is kept for the first simulator only

**Context.** The first HMI was a web page. The user asked for tools "the way the field normally builds them"; the HMI needs smooth 3D,
drawers and many live charts at 30 FPS on a laptop.

**Decision.** A native desktop HMI in PySide6 (Qt 6) with pyqtgraph + OpenGL for the 3D scene and charts. The car serves state over UDP
(and HTTP for other tools), so any client can subscribe.

**Consequences.**
- Native performance and widgets; one Python code base with the twin and labs.
- Desktop-only; a phone view would need the HTTP interface.
- OpenGL contexts must be shared across windows (a crash that was fixed).

# ADR-017 EV-cockpit HMI design system with dark and light themes

**Status:** Accepted

**Context.** The user judged the HMI against production cockpits (Huawei ADS, Tesla) and rejected near-black themes and rows of plain
buttons; teachers will judge the project partly on it. NHTSA guidance asks for short glances and fixed colour meaning.

**Decision.** One design system (`gui/theme.py`, `gui/controls.py`): slate blue-grey dark theme and pale light theme with a switch;
blue for lines, paths and active state, never as a flat fill; green / amber / red reserved for ok / caution / act; settings as icon
tiles and segmented controls (the approved Assists drawer is the template); drawers scroll and never exceed the window.

**Consequences.**
- A consistent, legible HMI tested at four screen sizes in both themes.
- Every new panel must be built from the design-system controls.

# ADR-018 No ROS; a single relay process with ROS-mappable interfaces

**Status:** Accepted

**Context.** ROS 2 is the field's standard middleware, and an earlier version of the project used ROS 2 + Gazebo. On the Pi 5 the
safety loop needs a small, predictable latency, and the twin must run the same code thousands of times per hour on a laptop.

**Decision.** The car runs one relay process with in-process calls between modules; the laptop talks to it over UDP / HTTP. Interfaces
are defined as data objects that map 1:1 onto ROS 2 messages (documented in the architecture), so a later port is mechanical. The
earlier ROS 2 / Gazebo version is archived in the repository.

**Consequences.**
- Low latency and simple deployment; the Monte Carlo runs the car code directly without middleware.
- No rosbag / RViz / Nav2 tooling; logging, replay and visualisation were built in.

# ADR-019 Modularity rules: hardware-free core, heavy work off the control loop

**Status:** Accepted

**Context.** The same algorithms must run on the car, in the twin, in replays and in tests; planning and SLAM take far longer than a
control cycle.

**Decision.**
1. `adas/` imports no drivers, simulator or GUI; `pi/` wires it to hardware; `sim/` and `tools/` wire it to the twin.
2. Planning, loop closure and vision run in worker threads / processes with budgets; the control loop only polls results.
3. Every decision is logged and streamed; every published number has a script that regenerates it.
4. Comfort filters sit after safety cuts and are bypassed by them.

**Consequences.**
- Same code everywhere; 161 automated tests run without hardware.
- Asynchronous results arrive late by design; the car holds still or brakes while waiting.

# ADR-020 Calibrations only on explicit request

**Status:** Accepted

**Context.** Repeated calibration runs on the vibrating car gave inconsistent results and consumed session time.

**Decision.** Calibrations (LiDAR offset, servo centre, speed / stopping, turning) run only when the operator asks. Results are stored and
applied only after confirmation, with a backup. Online estimates (steering gain and centre, latency) run continuously but only adjust
predictions, not the stored calibration.

**Consequences.**
- Development time goes to the ADAS itself; drifting calibrations are compensated online where possible.
- Some limits (braking deceleration spread) remain estimates until a calibration is requested.

# ADR-021 Loop-closure SLAM (Cartographer structure) for the world pose

**Status:** Accepted

**Context.** Return to start, speed zones and exploration need a world pose that does not drift. In a bare room, scan matching alone
ended 110 cm from the true start.

**Decision.** Submaps + correlative scan matching for loop closure + sparse pose-graph optimisation (Hess et al. 2016), in a worker thread;
the relay applies the resulting correction to its world pose.

**Consequences.**
- Return to start in a bare room: 110 cm -> 11 cm; drifting loops closed to ~1 cm.
- No global relocalisation after the car is picked up; maps are session-only.

# ADR-022 Online recalibration of the displayed risk, not of the decision

**Status:** Accepted

**Context.** A twin-trained model meets a real driver and car whose base rates differ; its probabilities drift out of calibration.

**Decision.** A per-session Platt recalibration, tracked by a Kalman-filter logistic regression with forgetting, learns from safety events
the car observes (brake latch / hold within 2 s of a prediction). It changes only the displayed risk and warnings; the takeover decision
and the gate are unaffected.

**Consequences.**
- Calibration error -37 % (nominal) / -28 % (unseen cars) prequentially in the twin.
- Needs ~30 resolved outcomes before it acts; cannot make a bad model discriminate better.

# ADR-023 Rejected: robust adaptive EKF

**Status:** Rejected (implemented, measured, not enabled)

**Context.** Huber-weighted, innovation-adaptive EKFs are standard against outlier measurements; scan matches can be wrong.

**Decision.** Keep the plain EKF with its Mahalanobis gate. The robust variant gave identical speed and yaw errors in every test,
including bad measurements that pass the gate, because the filter already weights each LiDAR measurement low.

**Consequences.** Less complexity. The remaining error is measurement bias, to be addressed with a redundant sensor (encoders, IMU or
visual odometry), not a different filter.

# ADR-024 Rejected: v3 model or a stricter defer rule as the takeover decider

**Status:** Rejected

**Context.** The target was a ≥ 90 % reduction of interventions. Options tried: v3 as decider at several thresholds; deferring every
swerve until the brake envelope; an oracle that knows the counterfactual.

**Decision.** None adopted. v3 did not beat v2; deferring to the brake envelope increased needless interventions; the oracle did not
reduce interventions at all. Instead, the arbitration changes of ADR-015 (commitment, respect of overrides, progress assists) were adopted.

**Consequences.** Recorded result: interventions -37 %, episodes -43 %, needless -41 % against plain ADAS with equal goals and 0
crashes; a 90 % reduction is not reachable in this scenario set because the remaining interventions free cars the driver cannot free.
A direct "is this takeover needed" classifier is the next candidate and must beat these numbers on untouched seeds to be adopted.

# ADR-025 Working agreements for development sessions

**Status:** Accepted

**Context.** Sessions are often interrupted (usage limits, hardware breaking); work was lost or repeated when it lived only in chat.

**Decision.**
1. Every task (asked or noticed) is logged in `docs/TODO.md` before work starts and ticked when done; partial work stays listed with its state.
2. When the car is available, features go on the real car for the user to drive and try to break; otherwise the twin.
3. Algorithms are chosen from published / established methods and cited; tools follow what the field normally uses.
4. ML is trained on the laptop GPU for many epochs but must stay light enough for the Pi (NumPy inference).
5. Never deploy to the Pi, restart its services or move the car without the user's OK; Pi benchmarks go in `~/rc_bench`.
6. No package or weight downloads without permission.
7. Documentation (`docs/requirements.md`, `architecture.md`, `decisions.md`, `modules/*`, `interfaces.md`) is updated with every
   milestone; `docs/REPORT.md` at session ends.

**Consequences.** Any new session can continue from the documents alone; progress is slower per step but never lost.

# ADR-026 Rear camera: comfort functions only, warnings never brake

**Status:** Accepted

**Context.** A rear Lenovo webcam will be added. Camera algorithms (visual odometry, looming, floor patches) are tested only on a synthetic
camera; the floor-patch detector is a brightness / texture heuristic.

**Decision.** The camera adds reversing guidelines, a ghost footprint (car position after 0.3 / 0.6 / 1.0 m of reversing), hazards in the
swept path with a STOP strip below 35 cm, floor-patch warnings, image-quality / vibration measurement and late fusion labels. It runs as a
separate service (MJPEG + JSON), never in the control loop, and none of its outputs brakes the car - the LiDAR gate does.

**Consequences.** The safety case does not depend on the camera; a missing or degraded camera is a designed empty state, not a fault.

# ADR-027 Fix failures generally, never for the one reported case

**Status:** Accepted

**Context.** The user reports failures as examples (e.g. a poor crash probability when driving at a wall, a car stuck in a corner).
Patching the example leaves the weakness elsewhere.

**Decision.** Every reported failure is analysed for its general cause (sliced evaluation, Monte Carlo breakdowns, log replay), fixed
generally, and verified on a broad set (all slices, untouched seeds), not just the reported scenario.

**Consequences.** Slower fixes; evaluation tooling (slices, bounds, oracle) is part of the project.

# ADR-028 Repository publishing: branch `real-car-adas`, project under `src/adas_project/`

**Status:** Accepted

**Context.** The GitHub repository `Aarish-Patel/adas_minor_project` already held an earlier ROS 2 / Gazebo version on `main` at
`src/adas_project/`. The working copy (`RC_Car/`) is a separate local repository without a remote.

**Decision.** The project is published on branch `real-car-adas` with its history rewritten into `src/adas_project/` (large drive logs
removed from history); the ROS 2 version is kept unchanged in `archive/ros2_gazebo/`. `main` is untouched until the user decides to merge.
Changed files are copied into the published layout and committed there. No README for now (user request).

**Consequences.** Two repositories to keep in step; drive logs, `data/` and user state (`gui/zones.json`) are never published.

# ADR-029 Scale conventions: 1:14, full-size km/h in the HMI

**Status:** Accepted

**Context.** The ADAS is meant to behave like a full-size EV; the car is 1:14.

**Decision.** Internally SI in car units. The HMI shows full-size-equivalent speed (m/s × 14 × 3.6); speed-limit zones are entered in
full-size km/h and converted; the steering envelope limits lateral acceleration to a full-size-equivalent ~0.6 g. Time is not scaled.

**Consequences.** Numbers on screen read like a real car; test thresholds in code stay in car units.

# ADR-030 Relocalisation: coarse-to-fine correlative matching against the submaps, refuse rather than guess

**Status:** Accepted

**Context.** Loop closure (ADR on submaps) needs a pose guess within ~0.45 m / 14°. A car lifted and put down, or switched on inside a
room that was mapped before, has none, and the pose it reports is wrong for the rest of the session. The map was also lost at every
restart.

**Decision.** `PoseGraphMap.relocalise` searches every submap over its whole extent and all headings (0.4 m / 6° max-pooled likelihood
field, then 0.2 m / 3° around the survivors, then 5 cm / 1° and ICP: multi-resolution correlative matching after Olson 2009 with the
pooled-field pyramid of Hess et al. 2016). A fix needs a score ≥ 0.60, ≥ 55 % of the scan within 10 cm of the map and a margin ≥ 0.06
over the best genuinely different pose; `Relocaliser` also needs a second scan to give the same place after the front end's motion.
Otherwise there is no fix (an empty rectangular room is symmetric; a corridor is ambiguous along its length). `rebase` joins the new
stretch to the map by an edge to the fixed submap and no odometry edge across the break. `kidnap_score` (range-image agreement of two
scans) triggers the search after a lift-and-drop; `SlamService.relocalise()` / `load()` trigger it on request or after loading a saved
map (`PoseGraphMap.save/load`). With no fix after 8 s the service carries on as a new session.

**Alternatives.** Feature or place-recognition descriptors (Scan Context, M2DP): more code and data for a 2D scan in a small room, and
they still need the geometric verification above. AMCL particle filter: needs a fixed occupancy map and a motion prior; the correlative
search is exact over the window and reuses the loop-closure fields.

**Consequences.** ~0.07-0.3 s per attempt on the laptop (0.25-1 s on the Pi estimate) in the worker thread only; the detector misses
about 8 % of drops in a room (poses that happen to look alike) and all in a corridor, so the operator trigger stays. The world frame
after a relocalisation is the saved map's frame, not the pose at power-up.

# ADR-031 Moving-object test: likelihood ratio on the track history with aperture-aware covariances

**Status:** Accepted

**Context.** The tracker called an object moving when its fitted speed exceeded 0.10 m/s for 3 scans (the relay then wanted ≥ 5). At 20 mm
range noise a fifth of the static wall fragments were flagged (a wall cut into pieces by grazing gaps has a centroid that slides along it,
and neighbouring pieces swap identities): the "never flagged" acceptance of FR-004 was only met through the ≥ 5-scan filter, and no
threshold / persistence choice of that test reached zero false flags (`models/tracking_legacy_grid.txt`).

**Decision.** `Tracker(model="lrt")` (default): every position sample has a covariance (range noise / sqrt(points) + sampling jitter;
along the cluster axis a large term per cut end: aperture problem); weighted least-squares lines over the newest 3..N samples (scan
statistic); statistic = squared Mahalanobis distance of the fitted velocity from "fixed in the world", with the car's motion error
(dv, dw) estimated from the other tracks (Huber-robust consensus, prior from the speed model) and its covariance included; χ² threshold
18.4 (flag) / 9.2 (keep), 2 consecutive scans; association gate 0.15 m; clusters cut at both ends are never flagged. Method: chi-square
innovation test (Bar-Shalom et al. 2001), sequential form after Wald 1945. The old test stays as `model="legacy"` for comparisons.

**Consequences.** Zero static flags in 20k+ twin track-scans at 8-30 mm noise (`sim/tracking_eval.py`); detection latency equal to the old
test under the consumers' ≥ 5-scan rule (0.4-1.3 s depending on speed) but reachable without that rule: the consumers should use
`Track.moving` directly (proposed in TODO P2-3). About 3 ms per scan on the laptop.

# ADR-032 Parallel parking on a continuous lane: gaps from the scan, nearest viable slot to the click

**Status:** Accepted (supersedes the fixed parking spaces S0..S7 of the Parking Simulator)

**Context.** The first Parking Simulator street had predefined spaces with occupied flags; the user rejected it: "no dedicated rectangles
should exist" - a real street is one lane along the kerb with cars at arbitrary spacing, and the driver points at where they want to park.

**Decision.** The HMI sends only the lane (road-edge polyline, side, depth) and the click (`PARK_TARGET`). The car finds the free gaps from
its own scan (`find_gaps`: returns in the lane band projected on the strip, 1-D clustering), a gap is viable if it is at least the car's
length + 0.30 m, a slot is placed in each viable gap as near to the click as the gap allows (at most 1.6 m long; a long gap parks where
clicked), the nearest slot with a manoeuvre is taken. When the car is beside the gap it measures it again at short range and re-plans
(`refine_parallel`, same rule among the gaps within 1.5 m): from afar parked cars hide parts of each other at grazing angles. The lane
itself (the kerb) is not visible to a 2D LiDAR, so its position comes from the car's pose.

**Consequences.** No spaces anywhere (world, HMI, planner, validation); validation is scored against the true parked cars. A near-tie between
two gaps (a few cm of difference in distance to the click) can go either way with perception noise - accepted within 10 cm.

# ADR-033 Tight gaps: the exit planned from the parked pose and driven backwards (Vorobieva et al. 2015)

**Status:** Accepted (replaces the Hybrid A* multi-point search for parallel parking)

**Context.** With the car's 0.67 m turning radius (twice its length) the reverse S needs ~1.3 m; the Hybrid A* multi-point search below that
took ~9 s and failed at 1.1 m, partly because its three-circle body model over-states the car by ~6 cm at each bumper.

**Decision.** Plan the EXIT: from the parked pose back up to the rear neighbour, then alternate full-lock forward and reverse strokes that
each end 5 cm before a neighbour or the kerb, until two arcs lead out to the lane; drive it backwards (Vorobieva, Glaser, Minoiu-Enache &
Mammar, IEEE T-ITS 2015). Clearance on the exact rectangle. The reverse S (80 % curvature, roomy margin first) stays the first choice.

**Consequences.** Deterministic, ~10 ms; gaps down to 0.55 m plan (9 gear changes); the minimum is set to car + 0.30 m (<= ~6 changes).
Forward entry (a dead end ahead) swings the long front end into the kerb with this steering: refused, documented.

# ADR-034 Scan-to-local-map odometry with degeneracy handling

**Status:** Accepted (replaces scan-to-scan `Odometry` in the relay)

**Context.** Scan-to-scan ICP drifted 3-5 % along a street (the parallel walls leave the along direction unconstrained, the speed model's
bias then decides) and ~1.5 deg heading over 9 m; parking after a long approach missed by 10-30 cm.

**Decision.** `pi/scanmatch.LocalMapOdometry`: each scan is registered to a voxel map of recent keyframes (KISS-ICP, Vizzo et al. RA-L 2023)
by point-to-line Gauss-Newton with Huber weights (Censi ICRA 2008); a direction with too little information (smallest eigenvalue of the
translation block) keeps the motion prediction (Zhang, Kaess & Singh ICRA 2016). Used for the manoeuvre frame and the world pose; the gate's
obstacle memory is re-anchored on it after each scan.

**Consequences.** Twin benchmark (6 streets, 9 m, speed model 5 % low): along 30.8 -> 3.1 cm, lateral 8.1 -> 1.4 cm, heading 1.8 -> 0.1 deg,
3.3 -> 1.7 ms per scan; return to start 11 -> 4 cm. To be confirmed on the car (MEASURE: straight 5 m drive, taped end mark).

# ADR-035 A deterministic twin

**Status:** Accepted

**Context.** The same simulated drive gave different results run to run (a bay park 3.8 / 8.2 / 25 cm; return to start failing 2 in 6): the
Hybrid A* budgets were wall-clock, the simulated Pi delay was measured time x 3.5, and the SLAM worker thread's timing decided which scans it
used - so tests were flaky and tuning chased noise.

**Decision.** In the twin (plan service inline with a latency factor): search budgets in nodes (`NodeBudget`, 1600 nodes per Pi-second), the
plan's release delay from the work done (nodes, grid cells, clearance evaluations) at measured Pi rates, SLAM inline. The car keeps
wall-clock budgets and the worker thread / process.

**Consequences.** Identical results on repeat; full test suite stable; the twin's timing is a model (rates in `adas/plan_service.py`).

# ADR-036 Bay parking: camera for WHICH bay, LiDAR for the wall and the stop

**Status:** Accepted

**Context.** Parking from the ArUco marker alone ended 6-9 cm short / long (range from four pixel corners, odometry on the way in) and tilted
by an askew marker (5-11 deg).

**Decision.** The marker gives the bay's place on the wall; the rear wall's line comes from the LiDAR (trimmed PCA around the marker,
neighbours' rear faces rejected); on the last reverse leg the stop follows the wall as the LiDAR measures it (parking-sensor behaviour).

**Consequences.** Bay suite 23/24 at 5 cm / 5 deg (all parks <= 4.4 cm, mean 2.8 cm); an askew marker no longer tilts the car.

# ADR-037 HMI: frames paced by the buffer swap; garbage only on the GUI thread; worker signals through bridges

**Status:** Accepted

**Context.** A free 16 ms timer beat against the display refresh (30 ms frames among 14 ms, visible judder); intermittent access violations
came from worker threads emitting on widgets that a window rebuild had destroyed, and from Python's collector freeing Qt objects in a worker.

**Decision.** The next frame starts when the 3D view's frame reached the screen (11 ms floor, 33 ms eco); `controls.ThreadSignal` bridges for
every worker-thread signal; `controls.gui_thread_gc()` (pyqtgraph GarbageCollector) at start-up.

**Consequences.** 62 -> 92 FPS, frame sd 6.3 -> 4.6 ms, on-screen shake p95 3.0 -> 2.3 mm (twin relay driving); no crash in repeated test runs.

# ADR-038 Parallel parking in phases: pull out, approach with the click-to-go planner, measure alongside, manoeuvre

**Status:** Accepted (amends ADR-032: the way to the slot is no longer part of the gap planner's path)

**Context.** Dashboard test of 30 Sep (user's relay log 10:16-10:19): the way from the car to the slot was one Dubins curve planned from one
scan and driven as fixed; from afar it was often a long reverse along the road (8 cm from a parked car, 21 cm from the wall, then stuck),
and from a car already parked in the lane no such curve exists ("no way to park there"). The slot choice itself depended on that curve: a
click was answered with a different gap when the curve to the nearest one failed.

**Decision.** As production park assists do (search, drive to the start of the manoeuvre, measure, park): the click chooses the slot from the
scan independently of where the car is (`plan_parallel(..., approach=False)`: the nearest viable gap with a manoeuvre, never a refusal while
one exists); a car standing in the lane first pulls out with the tiny-gap planner's exit strokes run forward (`plan_pull_out`); the
click-to-go planner (Hybrid A*, re-planned on blockage, stall recovery) drives to a point 0.8 m before the pull-up pose, beside the gap;
there the gap is measured at short range and the manoeuvre planned from the car's exact pose (run-in + reverse S, or multi-stroke). The map
shows the whole intended path (way + manoeuvre) and keeps it while a re-plan runs. The kerb has its own clearance (target 5 cm, 3 cm
along the path) apart from the neighbours'. A stroke the brake gate holds within 6 cm of its end counts as done (the gate keeps 6 cm +
1.2 cm at a crawl); a stall is measured as no progress, not as speed below a threshold (a car that creeps and stops for minutes is stalled).
A side offset left after the S (> 1.5 cm) is taken out with a two-arc shift and a straight move back, reversing when moving toward the kerb.

**Consequences.** See TODO AC8-AC15 for the dashboard results (tools/hmi_test.py). The approach costs a Hybrid A* search (the Pi's
planner process; the car holds still meanwhile) and the gap planner is used only from beside the gap, where its single-curve approach is
short.

# ADR-042 Parallel parking: the final position from the parked neighbours, not from the odometry

**Status:** Accepted (amends ADR-032's "the lane's position comes from the car's pose")

**Context.** With the actuation noise on, the car's picture of the lane drifted during the approach (along the street up to 18 cm, across
it up to ~4 cm on the dashboard) and during the manoeuvre (9 cm along in one run): cars parked 5-10 cm off-centre or on the kerb while
believing they were on target. A 2D LiDAR does not see the kerb; it sees the parked cars well at short range.

**Decision.** As production parallel-park systems place the target from the detected vehicle contours (Pohl, Sethsson & Ekmark, Proc.
IMechE D 220, 2006) and finish on the parking sensors: (1) when the gap is measured beside it, the goal across the lane is in line with the
neighbours' road-side faces (never nearer the believed kerb than 3.5 cm; without neighbours 5 cm off the kerb); (2) in the slot, near the
end of the path, the free distances to the neighbour ahead and behind and their end faces' road-side corners set the goal - centred
between them (plus the slot's planned offset in a long gap) and in line with them - before 'arrived' is decided; the side correction and
the creep then drive to it. The path keeps as much clearance to the kerb as the goal has (at most 3 cm).

**Consequences.** The final position no longer depends on the odometry since the click. It follows the neighbours: a row parked far from
the kerb is followed (clamped by the believed kerb), a single neighbour or none falls back to the odometry goal.

# ADR-043 A parking lane drawn as a rectangle on the map (simulator and car)

**Status:** Accepted

**Context.** The Park tool worked only on the simulator's street, whose lane comes from the simulator's layout; on the car there was no way
to say where the parking lane is. User, 30 Sep: "make it so I can define a parking lane as a rectangle in any simulation or the real thing".

**Decision.** A Parking lane map tool: drag a rectangle; its long axis is the lane, the long side nearer the car the road edge (the car
stands on the road), the other the kerb (a Flip tile swaps them), the short side the depth. It is stored in the relay's world (map) frame
in `gui/parking_lane.json`, turned into the same layout the simulator street uses (in the simulator frame through the current alignment;
on the car the alignment is the identity), so the Park tool, the map and 3D drawing and `PARK_TARGET` are unchanged; in the simulator the
street's cars and pavement stay on screen. A room switch or an origin reset deletes it (its coordinates would point elsewhere). A kerb the
LiDAR sees (a wall, a board) is handled: returns within 3 cm of the kerb line are the kerb (3.5 x the 8 mm range noise, and a 3 cm
drawing error), returns within 8 cm are not used to find parked cars, and with returns along the kerb inside the gap the car parks 6 cm
off it and the path keeps 5 cm (outside the brake gate's crawl envelope). In a gap longer than the slot, further positions 0.25 m apart
are tried, nearest to the click first, before a gap is given up.

**Consequences.** Works in every simulator room and on the car (dashboard: a lane over the street, 2-4 cm off the kerb in 4 of 5 runs,
0.4 cm in one - the drawn kerb was 1.9 cm off the true one; a lane along a room wall, 5.1 cm off it). The lane's precision is the
drawing's: the car cannot see a tape kerb - the neighbours (in line with them) limit the error.

# ADR-039 A fault in the autonomy never ends the relay

**Status:** Accepted

**Context.** 30 Sep: the relay process died on an AttributeError in unfinished AutoNav code (a restarted twin had loaded it); the dashboard
showed NO DATA and the brake gate was gone with the process.

**Decision.** The relay catches any exception from the autonomy's tick and from the GOTO / PARK_TARGET / PARK_MARKER / PARK / EXPLORE
commands, prints the traceback, cancels the autonomy (the car then holds until the throttle is let go, as after any autonomy end) and carries
on. Manual driving and the gate are unaffected.

**Consequences.** A planning bug becomes an "autonomy fault - stopped" message instead of a dead relay; bugs are still visible in the log.

# ADR-040 The displayed ego pose is filtered (One Euro)

**Status:** Accepted

**Context.** User, 30 Sep: the car keeps vibrating on the dashboard - LiDAR noise must move the points, not shake the screen. The displayed
pose was the relay's pose interpolated between packets; the scan matcher corrects it by millimetres / tenths of a degree ten times a second
and the chase camera magnifies a heading step across the whole view.

**Decision.** The pose for the car model, the chase camera, the mini-map and the map's follow view passes a One Euro filter (Casiez, Roussel &
Vogel, CHI 2012): 0.6 Hz cutoff at rest (heading 0.5 Hz), rising with speed (6 Hz per m/s, heading 3 Hz per rad/s) so the car does not
trail its own points; a jump (reset, relocalisation) is taken at once. The LiDAR points keep the packet's own pose, so their noise stays in the
points.

**Consequences.** Measured with `--measure` (TODO AC16). Up to ~2 cm lag of the car model at 0.3 m/s.

# ADR-041 The twin's car has imperfect actuation

**Status:** Accepted

**Context.** User, 30 Sep: the same steering and throttle must not always give the same curve in the simulator, since they do not on the car;
parking and click-to-go have to adapt. The twin's car executed every command exactly.

**Decision.** `sim.hw_sim.ActuationNoise` (seeded, on by default in every simulation): a steering-centre error (sd 0.8 deg) and left / right
gain errors (sd 6 %) per run, a speed gain error (sd 3 %) per run, slow wander of the steering offset (sd 0.5 deg, 2 s) and of the speed gain
(sd 4 %, 3 s) as first-order Gauss-Markov processes (Maybeck 1979), and 0.6 deg of play in the steering linkage. The Monte Carlo and the
scenario suite draw one car per seed; a simulator reset draws a new one.

**Consequences.** Results are reported with the noise on; open-loop accuracy no longer passes a test - the followers, the terminal
corrections and the re-planning must close the loop.

# ADR-044 The relay acts on the newest driving packet only
**Status:** Accepted

**Context.** Incident 1 Oct 2026 (TODO AH): during parking autonomy the relay handled 14-20 of the controller's 50 packets per second; the
UDP socket (212 kB) queued seconds of old commands and dropped 1943 packets. The driver's steering and throttle took effect about a second
late and went on after the trigger was released.

**Decision.** `pi/driver_input.py`: after each wait the relay takes everything already queued on its port, handles every control command
(PING, GOTO, ZONES, PARK, ...) in arrival order and, of the driving packets ("A", "M", "T" lines), only the newest - the keep-last, depth-1
policy used for teleoperation command streams (e.g. ROS 2 QoS for velocity commands). The receive buffer is set to 32 kB. Dropped stale
packets are counted in the drive log ("stale driver packets dropped").

**Consequences.** A slow tick lowers the control rate but can no longer delay commands; the heavy work that made ticks slow is still to be
moved out of the loop (TODO AH3). The twin runs the same loop, so the laptop simulator behaves the same.

# ADR-045 No LiDAR returns in the direction of travel is unknown, not free
**Status:** Accepted

**Context.** Incident 1 Oct 2026: for 9 s the LiDAR returned nothing across a 100-200 deg arc in front of the car (a dark or very close
surface); the path gate measured an infinite free distance and let the car push into the wall at full throttle until the wall appeared at
4 cm. The rear bevel differential broke.

**Decision.** `pi/path_gate.py`: each scan, the fraction of 2 deg bearing bins within 30 deg of straight ahead / behind that hold a return
(`sector_cover`). Driving towards a sector with less than half of its bins covered: speed capped at 0.10 m/s, and after 1 s of that the
throttle is held at zero ("LiDAR blind ahead - stopped") until the sector is seen again or the driver lets go. A bearing also counts as
seen when the obstacle memory holds a point there that is now too close to see (nosing up to a wall seen before is not blind; each
remembered point covers 3 cm either side). Unknown space is treated as
possibly occupied, the conservative reading of an occupancy grid (Thrun, Burgard & Fox 2005, ch. 9); ISO 3691-4 likewise asks driverless
vehicles to slow or stop when the protective field cannot be monitored.

**Consequences.** Replaying the whole incident session (22 073 driver ticks) through the gate: stopped from 402.7 s, 7 s before the
impact; elsewhere two 0.1 s creeps and one 0.9 s creep, no stop. Twin: goto_and_home runs 0 blind ticks, relay_scenarios 23/24 as before. A driver can
still reverse out of a blind spot when the rear is seen.

# ADR-046 Remembered obstacles are removed only on evidence that depends on distance and size
**Status:** Accepted

**Context.** User, 14 Oct 2026 (real car): thin objects were remembered when the car was close and deleted once it moved away, since
they are not returned from farther off. `adas/memory.py` dropped a point in one scan when no return lay within 10 cm of it, or when any
return within +-4 deg was farther; the dashboard map (ADR-less fix of 1 Oct) used a +-1 deg window.

**Decision.** `adas/visibility.py`, used by the gate's obstacle memory and the dashboard map: each remembered point / cell keeps a
clamped score (+1 per scan that hits it; memory max 8, map 6) and loses a miss weight when the scan sees through it. Only beams within
the point's own angular size count; that size comes from how many remembered points lie next to it (a thin post = 1-2 points, a wall =
many). A point is judged only when at least 4 beams should hit it from where the car is (a 2 cm post: within ~0.9 m; a wall: from
several metres), and the weight falls beyond 1 m (R_REF / r). Inside the LiDAR's minimum range a far return still removes the point at
once (a phantom wall from dead-reckoning drift). Clamped log-odds occupancy (Moravec & Elfes 1985; OctoMap, Hornung et al. 2013) with
Removert's visibility test (Kim & Kim 2020).

**Consequences.** A 2 cm post seen at 0.4 m stays when the car backs to 1 m with the wall seen beside it; a removed 20 cm box goes within
~15 scans; a wall that moved 1 m clears in ~6 scans at 2 m (was 2). Objects that really were removed but are only ever seen again from far
away stay on the map until the car comes closer. Dashboard cost ~11 ms per update at 13 000 cells.
