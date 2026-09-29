One-off experiments and early bring-up scripts from the first weeks of the project. Nothing in the running system imports
them (checked by name across the repository); they are kept for reference and can be run from the repository root
(`python pi/legacy/<name>.py`) - some need the LiDAR/ESP32 attached. The maintained tools are in `pi/` (relay, gate,
health, ESP32 link, calibration entry points used by the control panel) and `tools/`.
