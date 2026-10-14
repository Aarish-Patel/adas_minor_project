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
| Parking Simulator | simulator only: room switch (ROOM / BAYS / STREET), ArUco marker id, `Park` map tool. STREET: one continuous lane, no rectangles - a car-sized ghost follows the pointer on the lane, a click sends `PARK_TARGET` (lane + click) at once, the slot the car chose (`nav.slot`) is drawn blue on the map and in 3D. BAYS: the nearest free bay is highlighted, "Park at the marker" sends `PARK_MARKER`. Parked vehicles, road, sidewalk and marker are 3D objects in the drive scene |
| Parking lane | any room and the car (ADR-043): the Parking lane map tool - drag a rectangle; the long side nearer the car is the road edge, the far side the kerb (Flip tile), the short side the depth; card with Flip / Delete; saved in `gui/parking_lane.json`; replaces the street's lane for the Park tool; deleted on a room switch / origin reset |
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
| `EVWindow(link)` (`gui/ev.py`) | `frame()` (60 Hz timer), `refresh()` (process the newest packet + one frame, for tests), `toggle(drawer, on)`, `toggle_theme()`, `add_event(text)`, `panels`, `docks`, `clock` (FPS / profile), `parking_model`, `parking_published` |
| `render` (`gui/render.py`) | `FrameClock` (fixed timestep, FPS, per-stage profile), `StateInterpolator` (entity interpolation of the pose), `Smooth`, `SmoothAngle` |
| `parking` (`gui/parking.py`) | `ParkingModel` (strip, spaces, `click(xy)` -> published dict, `nearest_free`), `park_meshes` (3D objects) |
| `theme` (`gui/theme.py`) | `C` colours, `PALETTES`, `set_mode`, `load_mode`, `save_mode`, `on_change`, `STYLE`, `font/light/semibold`, `rgb`, `css_rgba`, `chip_css`, `style_plot`, `Tween` |
| `controls` (`gui/controls.py`) | `FeatureTile(glyph, title, hint, danger, state, momentary, compact)`, `TileGroup`, `Segmented`, `Card`, `ZoneRow`, `Chip`, `LegendPill(items)`, `ChartCard(title, plot, legend)`, `EmptyHint(parent, glyph, title, text)`, `section_label`, `style_primary`, `style_danger` |
| `widgets` (`gui/widgets.py`) | `Kpi`, `Stepper`, `RiskBar` |
| `MonteCarloWindow`, `TrainingWindow` (`gui/lab_windows.py`); `MonteCarloTab`, `TrainingTab` (`gui/lab_tabs.py`) | labs |

## Rules for changes
- Build panels from `gui/controls.py`; the Assists drawer is the approved template.
- Colours only from `theme.C`; register module-level colour tables with `theme.on_change`.
- Support dark and light; drawers must fit at 1100×700 and scroll; plain wheel scrolls drawers, Ctrl + wheel zooms maps.
- Every empty / no-data state is designed (message or placeholder), never a blank box: main window = range rings + pulse in the 3D
  scene, radar mini-map, "waiting for the car" pill, risk card "no data"; lab charts = `EmptyHint` until data arrives.
- Charts: `lab_tabs.plot()` (no title / legend of its own, 13 px ticks, 12 px axis captions) inside `chart_card()`; the legend is a
  `LegendPill` above the plot, never drawn over the curves. Bar charts print their values on the bars.
- Solid 3D objects (room walls, furniture, lab obstacles) are closed boxes (`ev.solid_box`, `ev.wall_slabs`) drawn with
  `Scene.mesh(..., lit=True)` (per-face Lambert shading), never a single flat quad.
- Background I/O in threads; UI updates via Qt signals - a worker thread emits only through a `controls.ThreadSignal` bridge (the widget
  may be destroyed meanwhile: a closed window, the rebuild on a theme switch), and Python garbage is collected only on the GUI thread
  (`controls.gui_thread_gc()` at start-up) - both caused access violations before (ADR-037).

## Test strategy
`tests/test_gui_smoke.py` (offscreen): drawers fit at three sizes, live and no-data state, theme switch, map zoom / pan / fit,
wheel scroll vs Ctrl-zoom, car geometry inside the real footprint, labs build and tick. Visual review with `--snapshot` screenshots
in both themes.

## Performance targets
| Metric | Target |
|---|---|
| Refresh | ≥ 30 FPS, even frame pacing (`python -m gui.dashboard --measure 20`, twin relay weaving, 1707x1067 @ 165 Hz): 92 FPS bare scene, frame-time sd 4.6 ms, on-screen shake p95 2.3 mm; 64 FPS with the map drawer; relay health NORMAL throughout |
| NO DATA shown | after 2.5 s without packets, not before |
| Drawer layout | no clipping from 1100×700 to 2560×1440 |

## Rendering model (TODO G3-1, G3-5)
- The UDP receive thread only stores the newest packet. The GUI thread runs one 60 Hz `QTimer` (`frame()`): `_process()` when a packet
  is new (texts, chips, meshes of the LiDAR objects and paths), `_animate()` every frame (interpolated pose, camera, gauges).
  OpenGL cannot be drawn from another thread, so "separate rendering thread" = separate receive thread + a renderer that only reads the
  latest snapshot. Fixed timestep: `FrameClock` (Fiedler, "Fix Your Timestep").
- Smoothing: `StateInterpolator` shows the pose one packet interval and a half in the past, interpolated between the bracketing packets
  (entity interpolation, Bernier / Valve); a late packet is extrapolated for at most 0.1 s then held; a jump > 1.5 m (origin reset)
  is not glided. `DriveScene.set_view()` moves the newest packet's meshes as one group by the pose difference and slides a world-fixed
  floor grid under the car, so obstacles and paths glide at 60 Hz. Map camera = exponential follower (`Smooth`, tau 0.1 s).
- Only changed objects update: the world map cells are cached per version, the map scatter is set only when they change, the parking
  meshes live in a world-frame group and are rebuilt only when the layout / target / hover changes (the group matrix follows the car).
- Costs found by profiling and removed: pyqtgraph axis tick labels (~8 ms a frame while the map follows -> `MetricGrid`, one pass, no
  text), per-point Python loops in `point_walls` / `walls_mesh` (2 ms -> 0.05 ms), 40 000-cell map rebuild every frame, PyOpenGL's
  `glGetError` after every call (`OpenGL.ERROR_CHECKING = False`), Python loop of `drawPoint` in the mini-map.
- The FPS readout (top right of the 3D scene) has the per-stage timings in its tooltip.
- Frame pacing (ADR-037): the next frame starts when the 3D view's previous frame reached the screen (floor 11 ms, 33 ms in eco); a free
  16 ms timer beat against the display refresh (30 ms frames among 14 ms). The map drawer follows the car every other frame.

## Files
`gui/dashboard.py`, `gui/ev.py`, `gui/render.py`, `gui/parking.py`, `gui/theme.py`, `gui/controls.py`, `gui/widgets.py`, `gui/lab_windows.py`, `gui/lab_tabs.py`;
user state `gui/zones.json`, `gui/parking_lane.json` (git-ignored); simulator rooms for the Parking Simulator: `sim/parking_worlds.py` (`park_perp`, `park_street`), switched at run time by `tools/sim_car.py`. Older browser dashboards: `pi/dash/`, `pi/lidar_gui.html`, `web/`.

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
