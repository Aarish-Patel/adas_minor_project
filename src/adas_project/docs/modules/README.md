# Subsystem documents

Each file describes one subsystem completely enough to work on it without loading the rest of the project. Shared conventions
(frames, units, hard rules) are in `CLAUDE.md`; requirement IDs refer to `docs/requirements.md`.

| Document | Subsystem | Main folders |
|---|---|---|
| [perception.md](perception.md) | LiDAR processing, blind-zone memory, free space, ego-motion, localisation + SLAM, tracking, rear-camera vision | `adas/`, `adas/vision/`, `pi/scanmatch.py` |
| [safety_decision.md](safety_decision.md) | Prediction, path gate (emergency brake), assists and arbitration, driver-intent use, warnings, health limits | `pi/path_gate.py`, `pi/relay_assists.py`, `adas/assists.py` |
| [planning.md](planning.md) | Hybrid A*, Dubins, MPPI, planner service, click-to-go, return to start, exploration, parking, speed control | `adas/` planners |
| [ml.md](ml.md) | Driver-intent models, datasets, training, calibration, conformal threshold, online adaptation | `adas/intent_net.py`, `sim/train_*`, `models/` |
| [monte_carlo.md](monte_carlo.md) | Digital twin, scenario suite, repeatability, fault injection, Monte Carlo, evaluations | `sim/` |
| [gui.md](gui.md) | Qt EV-cockpit HMI and the two labs | `gui/` |
| [integration.md](integration.md) | Car runtime: relay process, ESP32 link and firmware, health, logging, network interfaces, services, deployment | `pi/`, `ESP32_RC/`, `tools/` |

Boundaries: perception produces data objects -> safety_decision consumes them and owns the final command -> planning serves
paths on request -> integration wires everything to hardware and the network -> monte_carlo runs the same classes on the twin ->
gui only talks to the car over the network. ML provides models and is consumed by safety_decision (intent) and gui (displayed risk).
