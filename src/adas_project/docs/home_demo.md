# Home demo (TODO AN)

Showing the car in a house instead of on a course: a free floor, a few boxes, a wall for the bay, and a small ArUco marker. Everything below is
tested in the digital twin; the car-side steps are marked **on the car** because the Pi was off the network when this was written.

## What the car shows
| Feature | How to show it | What to say |
|---|---|---|
| Emergency braking | drive at a wall or a box with RT held | the relay brakes from the LiDAR, the driver cannot drive into it; the dashboard counts down to contact |
| Collision / side warnings | drive past furniture | warnings before braking, the brake is the last resort |
| Evasive steering (Assists drawer) | switch it on, drive at a box | it steers round and rejoins the line |
| Speed zone | draw a 5-10 km/h zone on the map, drive through | the car holds the zone's limit whatever the stick says |
| Click-to-go | Go to tool, click the map, hold RT | plans round obstacles, RT is the dead-man switch, any steering or brake hands control back |
| Return to start | "Return to start" tile (ORIGIN first) | drives back to where the map began |
| Thin objects stay on the map | a table leg or a 16 mm post | the map keeps what the LiDAR saw, removes what it saw past |
| LiDAR-blind object | a box in black felt, crawl into it | the virtual bumper notices it is not moving and marks the obstacle |
| Parallel parking | Parking lane tool over two boxes with a 1.0-1.5 m gap, Park tool, click in the gap, hold RT | finds the gap with the LiDAR, parks in a few strokes |
| Bay parking with the camera | "Park at the marker" tile, hold RT | the front camera finds the ArUco marker on the bay wall, the car drives in nose first and stops 13 cm from the wall |

## Set-up
**Floor.** About 3 x 2 m free of cables and rugs with fringe. Hard floor, not thick carpet (wheel slip spoils the odometry). Put the car on a start mark
and press **ORIGIN** on the dashboard once, with the car standing there.

**LiDAR.** Window clear, power bank charged (it ran flat once during testing: the relay then shows FAULT "no LiDAR scans"). Objects the LiDAR must see
are at least **23 cm** tall (its scan plane is ~20 cm above the floor; measure it): shoe boxes stood upright, a chair leg, a bag.

**Bay.** A wall or a cardboard panel (30 cm tall, 60 cm wide) as the back wall, two boxes (about 22 cm wide, 25 cm tall, 40 cm deep) standing against it
**46 cm apart centre to centre** (a 70 cm opening between them is the twin's bay; anything from 40 cm up works for the 20 cm wide car).
The car starts **1.2 to 1.8 m** in front of the bay, facing the wall within about 30 degrees, with the marker in view.

**Marker.** `python tools/make_marker.py` writes `docs/home_demo/aruco_23_80mm.png` / `.pdf`: an A4 page with ArUco id 23 (DICT_4X4_50), the black
square exactly 80 mm. Print at 100 % ("actual size"), check the 50 mm scale bar with a ruler, cut along the dotted line, stick it flat on the bay's
back wall **centred in the bay, the centre of the square at the height of the camera lens**, matt paper (gloss reflects the room's lights).
A different size or id: edit `marker_size_m` / `marker_id` in `pi/camera.json` and run the script again.

**Camera on the front of the car.** Any USB webcam. Fix it rigidly (a wobbling camera is a wobbling estimate) looking straight ahead, a little down at most.
Measure from the car, in metres, and put them in `pi/camera.json` **on the Pi**:
- `x`: lens ahead of the rear axle (the front bumper is at 0.28; a camera just behind the nose is about 0.25),
- `z`: lens height above the floor (about 0.10),
- `pitch_deg`: tilt down (0 if level),
- `hfov_deg`: the webcam's horizontal field of view (the box says; 60-78 degrees for most; 70 is assumed).

A 5 cm error in `x` or `z` moves the estimated bay by about as much; the LiDAR measures the wall itself and sets the final stop, so the camera only has to say
*which bay* and roughly where. A calibration (`tools/camera_calibrate.py`: checkerboard, intrinsics and mounting) improves it; run it only when you decide to.

## On the car (first time)
1. The Pi's address is now 192.168.1.10 (it was .15 and .6); `rc_controller.py` has `ESP32_IP = "192.168.1.10"`.
2. Deploy the current code (relay, `adas/`, `pi/`) and the new files `pi/camera.json`, `pi/marker_camera.py`, `pi/marker_client.py`, `adas/camera_config.py`.
   Keep the car's own JSON files (`pi/tuning_real_car.json`, `pi/car_model.json`, ...).
3. The camera service needs OpenCV with the `aruco` module on the Pi (`python3 -c "import cv2; cv2.aruco"`). If it is missing it has to be installed:
   ask first (nothing is downloaded without permission).
4. Start the service: `python3 ~/rc_car/pi/marker_camera.py` (first with `--bench 50` to see the frame time; without a camera: `--sim`).
   The dashboard's **Camera** drawer then shows the stream with the marker outlined, "marker 23: seen, 1.2 m away".
5. Restart the relay once so it reads `pi/camera.json` (control panel -> Drive). It writes the tuned wheel centres on start.

## Before every show
```
python tools/preflight.py --host 192.168.1.10
```
It reads the car's live state (nothing is sent to the motor) and prints OK / WARN / FAIL per item with the fix: relay answering, health, LiDAR rate,
ESP32, Pi temperature, what the LiDAR sees, the camera service and whether it sees the marker. The dashboard's **Park at the marker** tile also says
"seen 1.2 m" or "not seen" live.

Then: car on its mark, **ORIGIN**, `rc_controller.py` running, a slow look round (or "Explore the room") so the map shows the room.

## Script (about 10 minutes)
1. **Safe by default:** drive at a box with RT held; the gate brakes; point at the countdown and the warnings.
2. **Zone:** draw a slow zone, drive through at full throttle.
3. **Click-to-go and home:** click across the room round a box; then "Return to start".
4. **Thin object:** drive past a chair leg; show it stays on the map after it has gone behind.
5. **Parallel parking:** lane over two boxes, Park tool, click in the gap.
6. **Bay parking with the camera:** place the car 1.5 m from the bay, facing it; the Camera drawer shows the marker found; press "Park at the marker", hold RT.
   Let go of RT at any time and the car stops.
7. **Optional, for the engineers in the room:** the black box; crawl into it; the bumper marks it.

## If something goes wrong
| Dashboard says | Cause | Do |
|---|---|---|
| HEALTH LIMP: Pi supply under-voltage | the Pi's power supply sags (it throttles, the LiDAR and ESP32 USB adapters time out, the car feels laggy) | use a 5 V / 5 A supply or a UBEC from the battery with a short, thick cable; `journalctl -k \| grep -i volt` on the Pi shows each event |
| HEALTH FAULT: no LiDAR scans | the LiDAR's USB adapter dropped (loose cable, power bank empty) | reseat the cable, charge the bank; the relay restarts itself after 8 s and finds it again |
| the car only drives forward / never reverses | the motor driver's direction line (ESP32 GPIO 26) | check that wire and the driver's ground; it worked again after reseating on 9 Nov |
| "Park at the marker . not seen" | marker out of view, too far (> 1.8 m), too oblique, glare | move the car closer and square to the wall; shade the marker |
| "camera off" | the camera service is not running | start `pi/marker_camera.py` |
| "stuck - no way through from here" | the planner gave up after three stalled re-plans near a wall | take over with the stick, back off, click again |
| the car stops 10-13 cm from the wall | that is the brake gate's distance, by design | none |

## What the twin says (so you know what to expect)
Bay parking, nose in, 8 cm marker, 132 randomised runs (random start +-12 cm / +-6 degrees, camera noise 0.3-1.2 px, LiDAR noise 0-4 cm, dropped frames,
odometry drift, a box beside the path): no collisions; of 118 runs that parked, 108 ended within 6 cm and 10 degrees of the goal (median 2.3 cm, 2.8 degrees;
95th percentile 7.5 cm, 7.5 degrees); 92 % of all runs within 6 cm / 10 degrees. A marker that is hidden or beyond ~2 m is refused (the car does not move).
`python tools/front_bay_validation.py mc 6` reproduces it.
