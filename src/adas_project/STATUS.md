# Real-hardware status (Pi + LiDAR + ESP32)

Working notes from tonight's hardware bring-up and follow-on development. Everything below
is a distinct git commit (`git log --oneline`) so you can check out and test any point
individually. This file is the handoff between sessions — read it first.

## Verified working on real hardware

- **RPLIDAR A3M1** on the Pi (`/dev/ttyUSB0` or `/dev/ttyUSB1`, auto-detected): connects,
  spins, gives valid scans. Fixed a real power back-feed bug (Pi's own USB VBUS fighting the
  power bank on the barrel jack) by cutting the cable's VBUS wire and moving to two
  independent power banks (one for the Pi, one for the LiDAR) — see chat history if the
  motor starts refusing to spin again; it's almost certainly a power issue, not code.
- **ESP32** ("ROVER READY" firmware, `ESP32_RC/ESP32_RC.ino`): responds on both USB serial
  (`/dev/ttyUSB1` @ 115200) and WiFi UDP (port 4210) *simultaneously*, no re-flash needed for
  either. Motor polarity is **reversed** on this car (`M 90` drives backward) — already
  handled by `MOTOR_REVERSED=True` in `rc_controller.py` and `adas/servo.py`, and by
  `WIRE_MOTOR_REVERSED` in `pi/wifi_drive_safety.py`. If you ever touch the wiring, re-verify
  with `pi/motor_pulse_test.py`.
- **LiDAR mount calibration** (`adas/lidar_mount.py`, saved in `pi/tuning_real_car.json`):
  - `yaw_offset_deg = 96.5`: the LiDAR's own raw-angle zero is NOT the car's front. Measured
    by placing a single object dead-ahead and reading which raw angle saw it.
  - `front_overhang_m = 0.16`, `rear_overhang_m = 0.17`: LiDAR-to-bumper distances (measured
    with a ruler), used to convert raw sensor range into bumper clearance.
  - `min_valid_range_m = 0.20`: readings closer than this are ignored. There's a wire/mount
    bracket that sits in the beam path and reflects at a fixed ~16-19cm — this filter hides
    it. **Consequence**: the system is blind to real obstacles closer than ~4cm from the
    bumper (`0.20 - 0.16`). The stop margin must always stay above that floor or the car
    will drive through an obstacle it can no longer see. If you ever re-route that wire,
    this filter could probably be tightened.
- **`pi/obstacle_stop.py`**: standalone creep-and-stop demo. Confirmed working, including a
  real fix for a "no reading right after a stop = treated as clear, drives forward into the
  obstacle" bug (now holds "blocked" for 1s after losing the reading).
- **`pi/wifi_drive_safety.py`**: the main event. A relay between your WiFi-connected
  Xbox controller (`rc_controller.py` on the laptop) and the ESP32. Runs as a proper
  **systemd service** (`rc-relay`, auto-starts on boot, survives SSH disconnects — this took
  three attempts: plain `nohup`/`setsid` do NOT survive systemd's session cleanup on this
  Ubuntu image even with `loginctl enable-linger pi` enabled; only a real systemd unit does).
  - Blocks throttle **only in the direction you're actually commanding** (front cone when
    driving forward, rear cone when reversing) — steering is never touched, and the other
    direction always stays available, per your request.
  - Stop margin **scales with closing speed**, measured directly from consecutive LiDAR
    readings (no PWM-to-speed calibration needed). Fixes the "stops too close when
    approaching fast" bug you found. `ASSUMED_DECEL = 1.0 m/s²` is an uncalibrated guess —
    tighten or loosen it once you've run `pi/calibrate.py --real` for a measured number.
  - Logs every tick to `pi/rc_car/pi/drive_logs/drive_<timestamp>.csv` (steer, PWM commanded
    vs. sent, front/rear distance + speed + blocked flag) — raw material for training a
    real-world driver-intent model later (see `adas/intent.py` for the eventual pipeline).
- **Code layout on the Pi**: `/home/pi/rc_car/{adas,pi}/...` — the real `adas/` package is
  deployed there now (not just standalone copy-pasted scripts), loaded via
  `adas.config.load_tuning()`.

## Known gaps / deliberately not done tonight

- **No steering-aware curved-path prediction on real hardware yet.** The relay checks a
  fixed ±25° cone regardless of steering angle, unlike the simulator's proper Ackermann-arc
  sweep (`adas/geometry.py`). Porting that requires knowing the LiDAR's position **from the
  rear axle** specifically (not just from the bumpers) for the turn-radius math to be
  correct — I don't have that measurement. Doing this with a wrong/guessed number risks
  being confidently wrong about where the car's path actually goes, which is worse than the
  current simple (but honest) cone check. Get one more measurement (LiDAR to rear axle, or
  LiDAR to front axle + wheelbase) before attempting this.
- **No moving-object tracking (`adas/tracking.py`) ported to the real relay.** Reasoned
  through this: the existing closing-speed mechanism already reacts correctly to *any*
  fast-closing gap regardless of whether the car or the object is moving, so the marginal
  safety benefit of full multi-object tracking is smaller than it looks, and it's real added
  complexity in a safety-critical live system I can't physically supervise. Widened the
  detection cone instead (15° → 25°) for a cheap, low-risk improvement in warning lead time.
  Worth doing properly later if you want earlier warning on fast-crossing pedestrians
  specifically.
- **No webcam connected.** Checked (`lsusb`, `/dev/video*`) — only the Pi's onboard ISP
  virtual devices exist, no physical camera. Everything camera-dependent (ArUco parking,
  traffic-sign ISA, lane keeping) is blocked until one's plugged in. `pi/devices.py`'s
  `WebcamMarkers` and `adas/markers.py` are ready and simulator-tested; untested on real
  video.
- **`pi/calibrate.py --real` now has a working `RealPlatform`** (ESP32 over serial, LiDAR
  forward sector, manual `rewind()` that waits for you to push the car back and press
  Enter). Deployed to the Pi (`/home/pi/rc_car/pi/calibrate.py`) but **not yet run** — it
  needs the LiDAR/ESP32 ports free, which means stopping `rc-relay` first
  (`sudo systemctl stop rc-relay`), and it drives the car repeatedly and automatically at a
  wall, so it needs your direct supervision and a clear, open floor. Run it, sanity-check
  the numbers in `tuning_suggested.json`, then merge `speed_model`/`aeb` into
  `tuning_real_car.json` by hand.
- **`pi/main.py` / `pi/runtime.py` (the "proper" `AdasPipeline`-based architecture) is not
  what's actually running.** `wifi_drive_safety.py` is a deliberately simpler, independently
  built relay — faster to get right and verify live, but it duplicates some logic instead of
  reusing `adas/pipeline.py`. Worth unifying later once the curved-path and tracking gaps
  above are closed, so the real car and the simulator are provably running the *same* ADAS
  code, not two parallel implementations.

## Suggested order for next session

1. **Run `pi/calibrate.py --real`** (stop `rc-relay` first, clear floor, stay ready to
   intervene) -> real speed/braking numbers instead of guesses. Merge the result into
   `tuning_real_car.json`.
2. Measure LiDAR-to-rear-axle distance -> unlock steering-aware curved-path checking.
3. Plug in a webcam -> test `pi/devices.py`'s `WebcamMarkers` on real ArUco markers, then
   revisit auto-parking / ISA.
4. Decide whether to port `adas/tracking.py` for real multi-object prediction, now that
   there's real drive-log data (`pi/rc_car/pi/drive_logs/`) to sanity-check it against.
5. Longer-term: migrate `wifi_drive_safety.py`'s logic into `pi/runtime.py` +
   `adas/pipeline.py` so the real car runs the exact same ADAS code as the simulator.

## Quick reference

```bash
# Check the relay is running
ssh pi@192.168.1.3 systemctl status rc-relay

# Watch it live
ssh pi@192.168.1.3 tail -f /home/pi/relay.log

# Restart after deploying a code change
ssh pi@192.168.1.3 sudo systemctl restart rc-relay

# Drive (from the laptop, controller plugged in, same WiFi as the Pi)
python rc_controller.py
```
