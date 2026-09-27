# RC-ADAS

An intent-aware ADAS for an RC car (Raspberry Pi 5 + RPLIDAR A3 + ESP32): path-predicted emergency braking,
Hybrid A* evasive steering, a learned driver-intent model that decides when to take the wheel, driving assists and
click-to-go autonomy. The car's own relay code also runs on the laptop against a digital twin (fitted car model +
simulated LiDAR), so every feature is tested there before the car. Current state: `STATUS.md`; open work:
`TODO.md`; methods and papers: `RESEARCH.md`.

## The digital twin (the car's code on the laptop)

```bash
python tools/sim_car.py --world doorway      # worlds: doorway, room, corridor, gap, open, log:<drive log>
python rc_controller.py --ip 127.0.0.1       # drive it (Xbox pad or keyboard)
```
Open http://localhost:8090/ (2D view: assist buttons, predicted path + collision X, planned manoeuvre; click the
map to drive there autonomously while holding the throttle) or http://localhost:8090/dash (3D EV dashboard).

| Check | Command | What it shows |
|---|---|---|
| Assist scenarios | `python -m sim.relay_scenarios` | 11 pass/fail cases on the relay code |
| Monte Carlo | `python -m sim.relay_mc 48` then `python -m sim.mc_stats` | crashes, goals, needless interventions and their burden; intent-aware vs not, paired tests |
| Intent model | `python -m sim.train_intent_net 80` | trains `pi/intent_net.json`, held-out AUC |
| Autonomy | `python -m sim.autonav_eval` | click-to-go on 6 goals -> `reports/autonav.png` |
| Twin accuracy | `python -m sim.twin_report` | replays a real drive log: path and LiDAR error |
| Car model | `python -m sim.car_model_eval` | physics vs physics+ML motor/steering model on held-out legs |

Key files: `pi/wifi_drive_safety.py` (the relay), `pi/path_gate.py` (braking), `pi/relay_assists.py` (assists,
intent, autonomy in the relay), `adas/hybrid_astar.py`, `adas/autonav.py`, `adas/intent_net.py`,
`adas/car_model.py`, `sim/hw_sim.py` (virtual car + LiDAR), `pi/dash/` (3D dashboard).

## The earlier laptop simulator (3D viewer)

Everything below is the first simulator (`server.py`), kept for its scenario, parking and camera work.

## Run it

```bash
pip install numpy scipy scikit-learn matplotlib opencv-python pygame pyserial websockets playwright joblib
python server.py            # then open http://localhost:8765
```

The viewer: pick a scenario, drive with **W/S/A/D** or an Xbox controller (stick + triggers), switch ADAS
**Off / Warn / Active** with `X`, camera views `1-4`, auto-park `G`, follow-the-leader `C`, traffic signs `I`.
Side panel tabs: **Live** (charts, driver-intent bars, events), **Tuning** (sliders, download `tuning.json`),
**Camera** (OpenCV running on the frames the 3D scene renders), **Results** (all evaluations).
**Bypass** (`B`) drives straight, goes around the obstacle on the more open side and rejoins the line, using the same controller and scan-matching odometry as the real car (`pi/bypass_core.py`). **Car: REAL** (`K`) switches the simulator to the measured car (`sim/real_car.py`). **Monte Carlo lab** (top right) replays many random drives side by side, ADAS off (red) vs on (teal).

## What is where

| Folder / file | What it is |
|---|---|
| `adas/` | The ADAS itself. No hardware or simulator imports. Runs unchanged on the Pi. |
| `sim/` | Simulator: car, arena, LiDAR, camera markers, scenarios, evaluations. |
| `web/`, `server.py` | The 3D viewer (three.js, vendored, works offline) and its server. |
| `pi/` | The real car: control loop, device drivers, ESP32 link, calibration procedures, simulator-in-the-loop. |
| `ESP32_RC/ESP32_RC.ino` | ESP32 firmware. Flash once; it only executes servo angles and motor PWM. |
| `rc_controller.py` | Laptop-only Xbox -> ESP32 driving (no ADAS). |
| `models/` | Trained models and saved evaluation results. |
| `reports/`, `web/data/` | Generated figures and the Results tab data. |

Key `adas/` modules: `aeb.py` (braking, speed scaling), `tracking.py` (moving objects), `memory.py` (LiDAR blind
ring), `intent.py` + `adaptive.py` + `warning.py` (driver intent and warnings), `parking.py`, `acc.py`, `isa.py`,
`markers.py` (ArUco), `pipeline.py` (everything wired together), `config.py` (the tuning file).

## Tests and evaluations

```bash
python -m sim.run_scenarios      # 14 fixed scenarios, ADAS off/on, +/-25% speed-calibration error
python -m sim.monte_carlo 200    # random drives, crash rate by throttle
python -m sim.faults             # fault injection: LiDAR unplugged, link delay, camera lost, Pi dies ...
python -m sim.parking_eval 60    # auto-park from random starting poses
python -m sim.sweeps 40          # robustness: crash rate vs calibration error, delay, noise ...
python -m sim.intent_data 240    # train + evaluate the driver-intent model
python -m sim.warning_eval 400 100   # does intent help the warnings? (long: ~20 min)
python -m sim.run_scenarios --real   # the same scenarios with the car as measured (sim/real_car.py)
python -m sim.bypass_eval 60     # obstacle bypass: random obstacles, walls, gaps; default car and real-car profile
python -m unittest discover -s tests -t .   # regression suite (23 tests, no car needed)
python -m sim.report --run       # everything above -> Results tab + reports/index.html
python -m pi.hil wall            # the real Pi runtime driving the simulated car + emulated ESP32
python -m pi.calibrate --sim     # rehearse the calibration procedure on the simulator
python tools/validate_camera.py  # 3D-rendered camera vs geometric sensor vs OpenCV (server must be running)
```

## From simulator to the real car

1. **Measure and replace the guesses** (`adas/vehicle_params.py`): body overhangs, LiDAR position on the car,
   camera position (`adas/markers.py` `Camera`). The LiDAR blind zone depends on where the LiDAR sits.
2. **Calibrate the speed model** (no encoder): the ADAS estimates speed from PWM. `pi/calibrate.py` does timed runs
   against a wall and writes `speed_model` and `aeb` suggestions. Rehearse it first with `--sim`. The real-car
   `Platform` (ESP32 command + LiDAR forward range) still has to be filled in.
3. **Tune in the viewer**: sliders in the Tuning tab; "Reality" sliders stress-test the ADAS. Download `tuning.json`.
4. **Flash `ESP32_RC/ESP32_RC.ino`** (WiFi credentials are inside it).
5. **On the Pi**: `python -m pi.main --tuning tuning.json --link udp --lidar rplidar:/dev/ttyUSB0 --camera 0`.
   Xbox buttons: X = ADAS mode, A = auto-park, Y = follow, B = e-stop.
6. **First drive checks**: wheels straight at rest; stick right turns the car right (if not, flip `servo_sign` in the
   `servo` section of `tuning.json`); forward is forward (`motor_reversed`); e-stop works; wheels off the ground first.

Untested on hardware: `pi/devices.py` (RPLIDAR driver, webcam, gamepad) and the real-car calibration platform.
The A3M1 Python driver in particular may need swapping for Slamtec's SDK.

## Honest limits

* Simulation numbers describe the simulator. The real car has slop, wheel slip and sensor quirks that are only
  approximated. The robustness sweeps show which mismatches matter most.
* The LiDAR sees one plane and nothing closer than ~20 cm from itself, so the last few cm before the bumper are blind
  (the stopping margin accounts for it). Obstacles lower or higher than the scan plane are invisible.
* The driver-intent model is trained on a *simulated* driver. Record real driving with the **Rec** button / `--log`
  and retrain (`adas/intent.py`) before trusting it on people.
