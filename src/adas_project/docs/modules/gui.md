# GUI (EV-cockpit HMI and labs)

## Purpose
Show the driver and the operator what the car and the ADAS are doing at a glance, let them configure assists, zones and autonomy,
and present validation and training as live research consoles. SRS: FR-040, FR-041, NFR-04, NFR-20.

## Responsibilities
| Area | Content |
|---|---|
| Main window | top bar (mode chip, health / ESP32 / link chips, theme switch) · 3D drive scene (hot-rod car with LiDAR puck, obstacles coloured grey / blue on path / red contact, predicted-path ribbon, proximity arcs, zones, goal) · speed cluster (full-size km/h, PRND, limit sign, steering range) · driver-risk card · status pill (path clear / contact / RSS) · mini-map · bottom app bar |
| Drawers | Map & zones (map with zoom / pan / follow, zone tools, autonomy tiles: return to start, explore, park) · Assists (approved template: tiles, drive-mode segmented control) · Car setup (Pi control panel) · Rear camera (guidelines, ghost car) · Diagnostics (stat cards, trip card, charts) · Events |
| Labs | Monte Carlo lab (all systems driving one room in 3D, system cards, replay timeline, results, paired tests, RUN tab) · ML training lab (pipeline stepper, three randomised twin cars, KPIs, curves, RUN tab) |
| Design system | dark (slate blue-grey) and light themes, persisted; one type scale; blue = paths / active lines, never a flat fill; green / amber / red fixed meaning |

## Inputs
StateMessage over UDP (subscribe on port 4210, renewed every 2 s) · HTTP: control panel :8080, rear camera :8091 · files:
`models/mc_live/`, `models/intent_training/`, `models/intent_v3_report.json`.

## Outputs
UDP text commands to the relay: `ASSIST <name> ON|OFF`, `MODE eco|normal|sport`, `ZONES <json>`, `ORIGIN`, `GOTO x y [heading]`,
`GOTO CANCEL`, `HOME`, `EXPLORE ON|OFF`, `PARK [PARALLEL|PERPENDICULAR]`, `FOLLOW_ON|OFF`, `ADAS_OVERRIDE_ON|OFF`, `GUI_SUBSCRIBE <port>`
· HTTP POSTs to the control panel · child processes for lab runs (Monte Carlo, data generation, training).

## Dependencies
PySide6, pyqtgraph + PyOpenGL, NumPy, OpenCV (rear camera). Geometry helpers from `gui.dashboard` (GEO, polar_xy). Talks to the car
only over the network; never imports `pi/` runtime modules.

## Public interfaces
| Interface | Contract |
|---|---|
| `python -m gui.dashboard [--host IP] [--open map|assists|setup|rear|diag|events|mc|train] [--snapshot PNG --after s] [--classic]` | entry point |
| `EVWindow(link)` (`gui/ev.py`) | `refresh()` (30 Hz), `toggle(drawer, on)`, `toggle_theme()`, `add_event(text)`, `panels`, `docks` |
| `theme` (`gui/theme.py`) | `C` colours, `PALETTES`, `set_mode`, `load_mode`, `save_mode`, `on_change`, `STYLE`, `font/light/semibold`, `rgb`, `css_rgba`, `chip_css`, `style_plot`, `Tween` |
| `controls` (`gui/controls.py`) | `FeatureTile(glyph, title, hint, danger, state, momentary, compact)`, `TileGroup`, `Segmented`, `Card`, `ZoneRow`, `Chip`, `section_label`, `style_primary`, `style_danger` |
| `widgets` (`gui/widgets.py`) | `Kpi`, `Stepper`, `RiskBar` |
| `MonteCarloWindow`, `TrainingWindow` (`gui/lab_windows.py`); `MonteCarloTab`, `TrainingTab` (`gui/lab_tabs.py`) | labs |

## Rules for changes
- Build panels from `gui/controls.py`; the Assists drawer is the approved template.
- Colours only from `theme.C`; register module-level colour tables with `theme.on_change`.
- Support dark and light; drawers must fit at 1100×700 and scroll; plain wheel scrolls drawers, Ctrl + wheel zooms maps.
- Every empty / no-data state is designed (message or placeholder), never a blank box.
- Background I/O in threads; UI updates via Qt signals.

## Test strategy
`tests/test_gui_smoke.py` (offscreen): drawers fit at three sizes, live and no-data state, theme switch, map zoom / pan / fit,
wheel scroll vs Ctrl-zoom, car geometry inside the real footprint, labs build and tick. Visual review with `--snapshot` screenshots
in both themes.

## Performance targets
| Metric | Target |
|---|---|
| Refresh | 30 FPS, data ≥ 20 Hz |
| NO DATA shown | after 2.5 s without packets, not before |
| Drawer layout | no clipping from 1100×700 to 2560×1440 |

## Files
`gui/dashboard.py`, `gui/ev.py`, `gui/theme.py`, `gui/controls.py`, `gui/widgets.py`, `gui/lab_windows.py`, `gui/lab_tabs.py`;
user state `gui/zones.json` (git-ignored). Older browser dashboards: `pi/dash/`, `pi/lidar_gui.html`, `web/`.

## Design references and user feedback (keep)
- References given by the user: Huawei ADS screens (3D view with blue path ribbon and glow ring under the car, grey object cars, split 3D /
  map view, valet-parking view) and Tesla's cluster / FSD view (grey and blue tones, speed-limit sign, lane ribbons, PRND).
- Approved: the Assists drawer (icon tiles lit when on, segmented drive mode, small-caps section headings) - template for all panels.
- Rejected: pitch-black backgrounds, rows of wide plain buttons, flat bright-blue button fills, empty grey boxes, cut-off text,
  the plain "path clear" pill, a white-block car.
- Car model: hot rod with fenders, exposed wheels, side exhausts, hood scoop and a spinning LiDAR puck, kept inside the real footprint.
- Rear camera: guidelines + ghost car (footprint where the car will be after 0.3 / 0.6 / 1.0 m of reversing), hazards in the swept path,
  STOP strip under 35 cm, floor-patch (puddle / hole) warning - warning only.
- Labs: research consoles - several cars actually moving in 3D, per-system status cards, replay timeline, results below.
- Palettes: dark bg #232D3D / surface #2B3648 / accent #3D8BFF; light bg #E9EEF4 / surface #FFFFFF / accent #1E6BFF;
  ok / warn / bad green / orange / red. Fonts: Bahnschrift (numerals, labels), Segoe UI Variable (text).
