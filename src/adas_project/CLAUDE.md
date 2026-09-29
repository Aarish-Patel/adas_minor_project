# CLAUDE.md - Minimal Sensor ADAS (1:14 RC car)

## Project
Intent-aware driver assistance for a 1:14 electric RC car. One 2D LiDAR is the primary sensor. The driver keeps control; the ADAS
intervenes only when the driver, left alone, would (nearly) collide. Verified on a digital twin that runs the car's own code.
Car: Raspberry Pi 5 (relay) + RPLIDAR A3 + ESP32 (servo/ESC PWM, 500 ms failsafe) + optional rear USB camera. Laptop: HMI, twin, training.

## Read first (in this order, only what the task needs)
1. `docs/requirements.md` - requirement IDs (FR-xxx, NFR-xx), acceptance criteria, status
2. `docs/architecture.md` - modules, data objects, interfaces, frames, state machines
3. `docs/decisions.md` - ADRs; do not reverse an accepted decision without a new ADR
4. `docs/TODO.md` - open work (log new tasks here, tick when done)
Background only when needed: `docs/RESEARCH.md` (methods + references), `docs/REPORT.md` (results), `docs/STATUS.md` (what is on the car).

## Tech stack
Python 3.11+ · NumPy, SciPy · scikit-learn (trees), PyTorch+CUDA (training only; car inference is NumPy) · PySide6 + pyqtgraph/OpenGL (HMI)
· OpenCV (optional camera) · UDP/HTTP/serial · Arduino C++ (ESP32) · unittest.

## Repository structure
```
adas/        algorithm core, NO hardware / simulator / GUI imports (planners, gate helpers, intent, tracking, EKF, SLAM, park...)
adas/vision/ rear-camera algorithms
pi/          car-side: wifi_drive_safety.py (relay main loop), relay_assists.py, path_gate.py, health, esp_link, zones, services
pi/legacy/   old experiments - do not use or extend
sim/         digital twin, scenario suite, Monte Carlo, evaluations, ML data + training
gui/         HMI: theme.py (design system), controls.py (tiles), ev.py (main window), lab_*.py (labs), dashboard.py (entry)
tools/       log pull/replay, camera calibration, benchmarks, sim_car.py (twin relay)
tests/       unittest suite (must stay green)
models/      trained models + evaluation JSON · reports/ figures · docs/ documentation · ESP32_RC/ firmware · web/ old browser viewer
```
Dependency rule: `adas` <- `pi` <- `sim`/`tools`. `gui` talks to the car only over the network. Nothing on the car imports sim/tools/gui/tests.

## Commands
- All tests: `python -m unittest discover -s tests` (~2 min, 160+ tests)
- Scenario suite: `python -m sim.relay_scenarios` (must print all passed)
- Monte Carlo: `python -m sim.relay_mc --live <rooms> off,brake-only,adas,adas+intent <styles>`; bound/paired: `python -m sim.mc_bound`
- Twin relay: `python tools/sim_car.py --world room` · HMI: `python -m gui.dashboard [--host IP] [--open map|assists|mc|train]`

## Coordinate frames
- Vehicle (base_link): origin rear-axle centre, x ahead, y left, angles/curvature + = counter-clockwise (left).
- LiDAR: 0.12 m ahead of the axle; relay scan format = (angle deg CLOCKWISE from ahead, range m). Convert only at the edge.
- World: pose at start / last ORIGIN reset. Manoeuvre frame: pose when a plan began. Camera: rear-facing, yaw 180 deg.

## Units
SI inside all algorithms (m, s, rad, m/s). Degrees only at interfaces and in the HMI.
Servo: degrees, centre ~87, + = right at the servo. Throttle: PWM -255..255, + = forward (after motor-direction correction).
HMI speed: full-size km/h = m/s x 14 x 3.6 (CAR_SCALE 14). Zones are in full-size km/h.

## Coding standards
- Match the surrounding code: comment density, naming, docstring style (module docstring states purpose + method + reference).
- Cite the published method in the docstring when implementing an algorithm; no ad-hoc logic where a standard method exists.
- Small functions, pure where possible; no global state in `adas/`; configuration via dataclasses / tuning JSON, not literals scattered.
- Every new feature: a unit test in `tests/` and, if behavioural, a twin scenario or evaluation script that prints the numbers.
- Report numbers honestly (sample size, seeds, twin vs car); tune on some seeds, confirm on untouched ones.
- GUI: build panels from `gui/controls.py` (FeatureTile, Segmented, Card, section_label) and colours from `theme.C`; support dark
  and light; no flat blue fills; drawers must fit 1100x700; check with `tests/test_gui_smoke.py`.

## Naming conventions
- Modules/functions/variables snake_case; classes PascalCase; constants UPPER_CASE.
- Units in names where ambiguous: `_m`, `_s`, `_deg`, `_kph`, `_pwm`; `v` speed m/s, `k`/`kappa` curvature 1/m, `th` heading rad.
- Relay/twin variants: `off`, `brake-only`, `adas`, `adas+intent`. Requirement IDs FR-/NFR-/HW-, decisions ADR-NNN, TODO sections letters.
- Evaluation scripts `sim/<topic>_eval.py` writing `models/<topic>.json` (+ `reports/<topic>.png`).

## Hard rules
1. Never modify modules unrelated to the task. No drive-by refactors.
2. Keep modules independent: respect the dependency rule; `adas/` stays hardware-free.
3. Car and twin run the SAME classes (RelayAssists, PathGate, RelaySpeed, RelayIntent) - change behaviour there, never fork a sim copy.
4. The path gate (emergency brake) has the last word; learned models may only withhold takeovers or change the display, never
   suppress a brake. Safety cuts bypass comfort filters.
5. Heavy work (planning, SLAM, vision) never runs in the control loop - use worker/budgeted services.
6. Write or update tests with every change; run the affected tests + `sim.relay_scenarios` before committing.
7. Log every task in `docs/TODO.md` before working and tick it when done; update `docs/requirements.md` status when a requirement's
   evidence changes; new design choices get an ADR in `docs/decisions.md`.
8. Calibrations (LiDAR offset, servo centre, speed/stopping, turning) only when the user asks. Never apply a calibration silently.
9. Do not deploy to the Pi, restart its services or move the car without the user's OK. Pi benchmarks go in `~/rc_bench`, never `~/rc_car`.
10. Do not download packages or model weights without permission. Never enter passwords.
11. Commit locally with the attribution trailer. Publishing: the GitHub repo `Aarish-Patel/adas_minor_project`, branch
    `real-car-adas`, holds this folder under `src/adas_project/` (local repo has no remote - copy changed files into that layout).
    No README for now.
