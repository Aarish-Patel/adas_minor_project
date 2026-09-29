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
