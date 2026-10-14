# How real LiDAR navigation stacks avoid our current problems (research, 14 Oct 2026)

Requested by the user after the real-car tests ("research how real autonomous car, turtlebot, rover etc. systems work using lidar data
... make the car as smooth and well tuned as possible, reduce unnecessary warnings and replans"). Parameter values marked (verified) were
checked against the current documentation; others are long-standing defaults to confirm against the version used.

## 1. Established stacks with a 2D LiDAR
- **Nav2 costmaps** (Macenski et al., "The Marathon 2", IROS 2020). ObstacleLayer: mark up to `obstacle_max_range` (2.5 m), clear by ray
  tracing up to `raytrace_max_range` (3.0 m) - clearing reaches further than marking; `inf_is_valid` (no-return beams may clear or not);
  `track_unknown_space` keeps free / occupied / unknown. DenoiseLayer drops small isolated occupied groups. Spatio-Temporal Voxel Layer:
  marks decay with time and are cleared only inside the sensor's view - the standard answer to "remember what the sensor can no longer
  see". InflationLayer: cost 252*exp(-k*(d - r_inscribed)) to `inflation_radius`; tuning guides advise a wide radius with a low k (e.g.
  1.0 m / 2.5) so the gradient fills free space and paths run down the middle of corridors.
- **Smac Hybrid-A*** (Macenski et al., arXiv 2024) (verified): `cost_penalty` 2.0 (cost-aware search keeps away from inflated cells),
  `non_straight_penalty` 1.2, `reverse_penalty` 2.0, `analytic_expansion_ratio` 3.5, `analytic_expansion_max_length` 3.0,
  `analytic_expansion_max_cost` 200 (shots to the goal that skim walls are refused), `allow_unknown` true.
- **Regulated Pure Pursuit** (Macenski et al., Autonomous Robots 2023) (verified): lookahead 0.6 (0.3-0.9) or velocity scaled
  (`lookahead_time` 1.5 s); slows in tight curves (`regulated_linear_scaling_min_radius` 0.9, min speed 0.25) and near obstacles
  (`cost_scaling_dist` 0.6 <= inflation radius); collision check up to the carrot (`max_allowed_time_to_collision_up_to_carrot` 1.0 s);
  approach slow-down 0.6 m. MPPI (Williams et al. 2016/17) is the best Nav2 controller for Ackermann but heavy for a Pi in Python;
  DWB / TEB are older. Progress checker: 0.5 m within 10 s; recoveries (clear costmap, back up, wait) only after a failure.
- **F1TENTH**: follow-the-gap, pure pursuit with curvature-scheduled speed; the **MAP controller** (Becker et al., ICRA 2023,
  github.com/ETH-PBL/MAP-Controller) maps pure pursuit's lateral acceleration to steering through an identified table: 4x lower lateral
  error than plain pure pursuit.
- **Autoware** (autoware_universe/vehicle): online steering-offset estimator, accel/brake map calibrator + feedforward converter, obstacle
  stop with `hold_stop_margin_distance` (no restart towards a stop point already close). **Apollo**: LQR lateral control with feedforward and
  integral action; steering/throttle calibration tables. **NASA/JPL ENav** (Toupet et al. 2020; Otsu et al., ACE, JFR 2020): cheap ranking,
  expensive collision checks only for the best candidates, the current arc kept until invalidated.

## 2. Fewer re-plans
Nav2's "consistent replanning and if path becomes invalid" tree replans only when `IsPathValid` fails (cost threshold on the remaining
poses, optionally unknown = invalid), the goal changes, or ~10 s pass; meanwhile the controller keeps the committed path and slows first.
Hysteresis: invalid only after N scans (e.g. 3, ~0.25 s); switch to a new path only if clearly cheaper (< 0.8x the remaining cost).
Incremental repair (D* Lite, Koenig & Likhachev 2002; AD*) or, simpler: replan locally from ~1 m ahead to a rejoin point, full search only
if that fails.

## 3. Planner clearance vs. the brake's protective field (brake flicker)
Industrial practice: warning (slow) and protective (stop) fields switched by speed (IEC 61496 / ISO 3691-4; ISO/TS 15066 separation
distance). Nav2 Collision Monitor (verified): STOP / SLOWDOWN / LIMIT / APPROACH zones, velocity-dependent polygons. Rule that removes
flicker: planner lethal clearance >= brake field at the planned speed + p95 tracking error + localisation error, plus a decaying cost
gradient beyond it, and the follower slows near obstacles before the brake field is reached - the brake becomes the rare last resort.
Brake hysteresis: release only at d + 3-5 cm after 2-3 clear scans. A speed-dependent field (v*T_latency + v^2/2a + margin) shrinks at
crawl speed, so narrow passages stop tripping it.

## 4. Unknown space
Frontier exploration (Yamauchi 1997; explore_lite; Nav2 wavefront frontiers). Unknown as moderately costly but traversable (1.5-3x free),
not lethal. Against dead ends: require the first ~1.5 m of a committed path to be in known free space, the rest may cross unknown space
but is re-validated as cells are seen; once a route through unknown space proves blocked, raise the cost there (frontier blacklisting).

## 5. LiDAR processing on embedded hardware
Angular binning keeping the minimum range per 0.5-1 deg; shadow / speckle filters (laser_filters) for mixed-pixel edges; vectorised
ray casting with clamped log-odds (Thrun et al. 2005), clearing only up to the return minus one cell; no-return beams never clear
(thin / dark objects); hits kept longer than free evidence; small-cluster denoise. Dynamic-object removal in 2D = visibility clearing
(the 3D methods Removert, ERASOR, peopleremover reduce to it). SLAM: Cartographer (submaps + branch-and-bound loop closure, what the car
already does), slam_toolbox, Hector; KISS-ICP's constant-velocity deskew and adaptive threshold carry over (the A3's motion distortion at
12 Hz is not negligible at 0.8 m/s).

## 6. Mechanical inconsistency through processing
Steering bias / gain by recursive least squares from the scan-matched yaw rate (kappa_meas = omega / v vs. commanded curvature, only
above ~0.15 m/s) - Autoware steer_offset_estimator; integral action on cross-track error with anti-windup; backlash inverse on steering
reversals (Tao & Kokotovic 1996). Speed: feedforward map PWM = f(v, battery) + dead-band jump + PI with anti-windup (Astrom & Hagglund 2006),
scheduled on battery voltage. Controllers at low speed: pure pursuit (Coulter 1992) robust, Stanley (Thrun et al. 2006) good at low speed but
heading-noise sensitive (Snider 2009 compares them); for this car RPP-style pure pursuit with velocity-scaled lookahead (min ~0.25 m,
0.6-0.8 s, max ~0.7 m) plus curvature feedforward; a kinematic MPC with identified steering lag/bias (acados) fits a Pi 5 at 20 Hz.
Iterative learning control (Bristow et al. 2006) for repeated manoeuvres such as parking.

## 7. AI / learning on a Pi 5
End-to-end imitation (TinyLidarNet, Zarrar et al., IROS 2024: 1D CNN, < 1 ms) suits reactive lap driving only - no goals, parking or
guarantees. RL local planners: large sim-to-real gap with this car's actuation noise, hard to verify - low priority. **Residual learning**
on top of the classical controller (Johannink et al. 2019; GP-MPC, Kabzan et al. 2019) is the best fit: little data, the safety layer
stays, outputs easy to clamp.

## Adopt next (priority order)
1. Re-plan only on invalidation (path validity each scan on the planning cost, invalid after >= 3 scans, slow first) - fewer re-plans.
2. Clearance hierarchy: planner lethal clearance >= brake field + tracking error + localisation error, plus a decaying cost gradient
   (Smac-style cost_penalty) - no brake flicker.
3. Brake hysteresis + speed-dependent protective field - fewer warnings and stop-go.
4. RPP regulation: proximity and curvature slow-down, minimum speed above the dead-band, velocity-scaled lookahead.
5. Online steering bias/gain RLS + backlash compensation.
6. Speed loop: feedforward map + dead-band jump + PI with anti-windup, battery scheduling.
7. Known-space commitment (first ~1.5 m known free) + dead-end blacklist (replaces the first AJ2 attempt, which slowed planning).
8. Cost cap on analytic Reeds-Shepp / Dubins shots + local path repair before a full re-plan.
9. Scan hygiene: min-range binning, shadow/speckle filter, deskew, asymmetric log-odds, no clearing by no-return beams.
10. Identified-model upgrade: MAP-style lateral-acceleration-to-steering table or kinematic MPC; later a residual model / ILC for parking.
Not now: end-to-end or RL controllers.

Sources: Nav2 RPP / Smac Hybrid / replanning BT / Collision Monitor / IsPathValid documentation (docs.nav2.org); Becker et al. 2023
(arXiv 2209.04346); ForzaETH race stack (arXiv 2403.11784); TinyLidarNet (arXiv 2410.07447); autowarefoundation/autoware_universe
vehicle packages; SteveMacenski/spatio_temporal_voxel_layer; hrnr/m-explore.
