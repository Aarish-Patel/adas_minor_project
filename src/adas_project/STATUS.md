# Real-hardware status (Pi + LiDAR + ESP32)

Read this first. Everything below is real hardware work from one long session, milestone by
milestone (`git log --oneline`, M1 through M17) — each one is a distinct commit you can check
out and test individually if something regresses.

**Important: the Pi went unreachable (SSH and even ICMP ping both timing out) partway through
the session, after M15 was deployed and tested live but before M16/M17 were deployed at all.**
Everything from M16 onward is committed and syntax-checked, but has never run on the real
hardware. **Start tomorrow's session with the deploy checklist below**, not by assuming
anything past M15 is actually live.

## Deploy checklist for tomorrow (do this first)

1. Confirm the Pi is reachable: `ping 192.168.1.3`, then `ssh pi@192.168.1.3`.
2. Stop the relay, push every file under `pi/` and `adas/` that changed since M15, restart:
   ```bash
   ssh pi@192.168.1.3 sudo systemctl stop rc-relay
   # scp/sftp everything (see "How this session deployed" below for the pattern used)
   ssh pi@192.168.1.3 sudo systemctl start rc-relay
   ssh pi@192.168.1.3 systemctl is-active rc-relay
   ```
3. Health check before driving: UDP `PING` to port 4210 should get `PONG`, and
   `http://192.168.1.3:8090/` should load the live GUI with a moving point cloud.
4. **Re-test the creep-collision fix specifically** (M16) before trusting it: creep toward an
   obstacle at low PWM and confirm it now stops at `CREEP_FLOOR_M` (5cm) instead of making
   contact. This was a real bug found live (see M16's commit message) — the fix is code-
   reviewed and logically sound but was never re-verified against real hardware.
5. Sanity-check the new imports don't crash on startup: `adas.tracking`, `adas.acc`,
   `pi.path_predict` are all new dependencies pulled into `wifi_drive_safety.py` tonight.
   Watch `relay.log` for a clean "listening for driver commands" line, not a traceback.
6. Only after 1-5 pass, test moving-object tracking (M16) and follow-mode (M17) — neither has
   ever run against real LiDAR data.

## What's verified working live tonight

- **RPLIDAR A3M1**, ESP32 dual serial+WiFi link, motor polarity handling — all solid,
  unchanged from earlier sessions.
- **LiDAR mount calibration**: `yaw_offset_deg` re-measured after the mount visibly shifted
  (96.5° → **95.1°**, from a tight 8-sample static-scan cluster). `min_valid_range_m=0.20`,
  overhangs front=0.16/rear=0.17/left=0.10/right=0.10 (in `pi/tuning_real_car.json`).
- **Servo center**: confirmed at **90.0°** across 3 independent converged passes (one showed
  just 0.4° drift) — high confidence, matches the current config exactly.
- **Speed model, finally trustworthy**: `v_max=0.68 m/s, deadband=0` — replaced the old
  simulator-derived guess (`v_max=1.0, deadband=40`). Measured via a properly ramp-aware
  burst duration (earlier attempts were corrupted by measuring mid-acceleration).
- **Turn radius** (real, not theoretical): +20°→2.184m, +35°→0.633m, -35°→1.351m (-20° was a
  discarded unreliable outlier). Left/right differ by >2x at the extremes — real asymmetry or
  still noise, not resolved. `pi/path_predict.py` uses each side's own data, not a symmetric
  assumption, since this feeds a safety-critical stop decision.
- **The relay's safety gate**, after several real bugs found and fixed from live testing:
  - Full-360° "wide backstop" (corner-strike protection) is split front/rear
    (`WIDE_CONE_DEG=90`), not one global flag — a single flag blocking *both* directions
    fired 68% of a real drive session on a wall the car was just parked next to on one side.
  - Body-overhang subtraction uses the car's actual rectangular footprint per bearing
    (`body_overhang()`), not one flat constant that overestimated closeness at the sides.
  - **Fixed a real creep-to-collision bug**: an obstacle closer than `MIN_VALID_RANGE_M` gets
    filtered out of the scan entirely, which flipped the body-alert to "false" (falsely
    "clear") right when the car was closest. Added the same hold-after-lost-reading
    protection that `front_track`/`rear_track` already had, which this path never got.
  - **Graduated creep vs. cancel**: inside the safety margin, a low-PWM command is still
    allowed through (down to `CREEP_FLOOR_M=0.05m`) for a deliberate slow approach; a high-PWM
    command in that same zone is cancelled outright, not softened to a cap.
  - **Active braking**: a real reverse pulse (not just coasting) when close AND still
    committing hard, speed-scaled and reduced if the driver's own recent commands show them
    already easing off (a lightweight intent-aware heuristic, not the trained ML model).
- **ADAS override**: a GUI button that passes every command through unmodified when toggled
  on, with an unmissable red banner. Verified round-trip live, both directions.
- **Live LiDAR GUI** (`pi/lidar_gui.html`, served by a small HTTP server embedded in the
  relay on port 8090, reusing its already-open LiDAR connection): real-time point cloud, car
  outline, front/rear cones tinted on block, override/follow buttons, tracked-object overlay.
  Verified rendering correctly against the real car in a browser.

## Built tonight, NOT yet verified live (Pi went offline mid-session)

- **Steering-aware predicted-path stop check** (`pi/path_predict.py`): reuses the simulator's
  already-tested arc-sweep footprint math (`adas/geometry.py`), fed the *measured* turn
  radius above instead of the theoretical bicycle-model angle (which was never independently
  checked against this hardware). Ran once in `pi/autonomous_drive.py`'s live test — 90s,
  46 segments, zero contact — but the arc check was never actually the deciding factor
  (the room's simple distance checks always fired first), so it has no live evidence yet of
  catching something the simpler checks would've missed. Needs a bigger space or a
  deliberately staged off-cone/in-arc obstacle to actually validate.
- **Moving-object tracking** (`adas/tracking.py`, ported into `Clearance._loop`): each scan
  feeds a `Tracker`, using an ego speed/yaw-rate estimate to separate real object motion from
  the car's own. `moving_object_contact()` blocks throttle if a moving object's predicted
  path meets the car's sooner than the required stopping distance — genuinely new capability
  (static distance checks only know *where* things are, not where they're headed), but
  completely unexercised against real moving objects.
- **Adaptive cruise / follow-the-leader** (`adas/acc.py`, opt-in via a GUI button): caps
  forward throttle to hold a time-gap behind whatever the tracker finds moving ahead. Depends
  entirely on the tracker above being correct, so equally unverified.
- All three share one dependency: the ego speed/yaw-rate estimate used to separate a
  tracked object's real motion from the car's own is only as good as the speed model and the
  measured-turn-radius interpolation above - if those drift, tracking quality drifts with them.

## Known gaps, unchanged or newly clarified

- **No webcam connected.** Still true - `lsusb`/`/dev/video*` only show the Pi's onboard ISP.
  Everything camera-dependent (ArUco parking, traffic-sign ISA, lane keeping) stays blocked.
  `pi/devices.py`'s `WebcamMarkers` and `adas/markers.py` are ready and simulator-tested,
  untested on real video - this is the single biggest remaining gap once the Pi is back.
- **No LiDAR-to-rear-axle measurement** - `pi/path_predict.py`'s `VehicleParams` uses the
  simulator's assumed `lidar_x=0.12, lidar_y=0.0`, never independently confirmed on this car.
  If the arc-sweep predictions look consistently offset once tested, measure this first.
- **`pi/calibrate.py --real`** still has never been run end-to-end (the newer
  `pi/lidar_steering_diag.py` iterative-convergence approach superseded it for tonight's
  actual calibration work, but the original procedure - with its own braking/latency
  measurement, which nothing else measures - is still sitting there unused).
- **Braking/deceleration is still `ASSUMED_DECEL=1.0 m/s²`, an uncalibrated guess.** Nothing
  tonight measured real stopping deceleration. This directly sizes `required_margin()` and
  therefore the stopping distance quoted to the user (~0.50m at top speed) - worth doing
  before trusting that number for anything beyond "reasonably conservative."
- **`pi/main.py`/`pi/runtime.py`** (the "proper" `AdasPipeline` architecture) is still not
  what's running - `wifi_drive_safety.py` remains a parallel, independently-built relay that
  now duplicates a fair amount of `adas/` logic (tracking, ACC, geometry) rather than reusing
  `adas/pipeline.py` directly. Worth unifying once the two together are more validated.

## Suggested order for the next session

1. Run the deploy checklist above. Confirm the relay starts cleanly with the M16/M17 imports.
2. Re-verify the creep-collision fix live - this was a real "drove into the wall" bug.
3. Verify moving-object tracking and follow-mode against an actual moving object (walk a box
   past the car, or push it by hand) before trusting either for anything.
4. Try to get real evidence for/against the path-prediction arc check specifically - stage an
   obstacle just outside the front cone but inside where a hard turn would swing the corner.
5. If there's spare time/patience: measure LiDAR-to-rear-axle, run a real braking-deceleration
   test, or get a webcam connected to finally unblock ArUco parking/ISA/lane-keeping.

## Quick reference

```bash
# Check the relay is running / watch it live
ssh pi@192.168.1.3 systemctl is-active rc-relay
ssh pi@192.168.1.3 tail -f /home/pi/relay.log

# Restart after deploying a code change
ssh pi@192.168.1.3 sudo systemctl restart rc-relay

# Drive (from the laptop, controller plugged in, same WiFi as the Pi)
python rc_controller.py

# Live LiDAR GUI (point cloud, override/follow toggles, tracked objects)
http://192.168.1.3:8090/

# Toggle ADAS override / follow mode without the GUI (UDP to port 4210)
# ADAS_OVERRIDE_ON / ADAS_OVERRIDE_OFF / FOLLOW_ON / FOLLOW_OFF
```
