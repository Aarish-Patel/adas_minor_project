# Demo course (TODO AL)

A small course that shows everything the car can do, one station after another, with as few printed parts as possible. The same geometry
(`sim/demo_course.py`) drives three things: the twin world `demo` (dashboard: Parking Simulator -> DEMO, or `python tools/sim_car.py --world demo`),
the print files (`python tools/demo_course_parts.py` -> `docs/demo_course/`) and the scripted check (`python tools/hmi_test.py demo_course`).

## Layout (5.0 x 2.4 m, top view; x ahead from the start mark, y left)

```
 y=+1.2 +-------------------------------------------------------------+
        |                              T (post, 2.0/1.05)     D (dark box, optional)
        |                         O (box 30x30)                       |
        |   S --->   [ Z speed zone 10 km/h ]           G (goal)      | B  brake wall
        |  start    x 0.4..1.2, y -0.5..0.5                           |   (x = 4.0)
        |  +-------------- parking lane ----------------+   free:     |
        |  | [car 1]        gap 1.5 m        [car 2]    |   turning   |
 y=-1.2 +-------------------------------------------------------------+   <- real wall = kerb
      x=-1.0  x=0                                    x=3.1          x=4.0
```

| Station | Where (m) | What it is | What the audience sees |
|---|---|---|---|
| S start | (0, 0), facing +x | printed arrow | the car's map starts here; "return to start" comes back here |
| P parallel parking | lane x 0.1..3.1 along the wall; cars at x 0.40..0.73 and 2.23..2.56, 5 cm off the wall | 2 shoe-box "cars", 4 printed corner marks | click in the gap, hold RT: the car finds the space and parks in a few strokes, a few cm from the kerb |
| Z speed zone | x 0.4..1.2, y -0.5..0.5, 10 km/h | drawn on the dashboard only | full throttle, the car still holds the zone limit |
| O obstacle | centre (1.8, 0.45), 30 x 30 cm | a cardboard box | click-to-go plans round it |
| G goal | (3.0, 0.2): one left turn from where the car leaves the space | printed cross | the car arrives within ~5 cm |
| T thin post | (2.0, 1.05), between the box and the side wall | printed 16 mm post | the thin post stays on the map (obstacle memory) |
| home | back to S | - | return-to-start, heading within a few degrees |
| B brake | the end wall, x = 4.0 | the room's wall or a cardboard panel | full throttle straight at the wall: the emergency brake stops it 20-30 cm short |
| D dark box (optional) | (3.75, 1.0), 25 x 25 cm, the far corner | a box wrapped in black felt | the LiDAR does not see it; the virtual bumper marks it after a soft touch (show only on purpose, at a crawl; take it away for the other stations - twin world `demo_dark`) |

Everything the LiDAR has to see must reach above its scan plane: the LiDAR height (measure it on the car, default 0.20 m) + 3 cm.
Shoe boxes stood on end, or two stacked, are tall enough; flat boxes are not.

## What to print (Creality K1, PLA, 0.2 mm layers, no supports)

| File | Count | Filament | Notes |
|---|---|---|---|
| `thin_post.stl` | 1 | ~18 g | 16 mm tube, 240 mm tall (4 cm above the scan plane), 60 mm foot; 2 perimeters, 0 % infill, brim helps |
| `panel_foot.stl` | 2 per cardboard panel | ~7 g each | U-channel for 3-5 mm cardboard; only needed where there is no real wall (kerb or brake wall) |
| `marker_start.stl` | 1 | ~4 g | flat arrow, 1.2 mm |
| `marker_corner.stl` | 4 | ~2 g each | the lane corners, so the lane can be drawn on the map in the same place every time |
| `marker_goal.stl` | 1 | ~2 g | flat cross |

About 46 g in all (solid volume; with 2 walls and sparse infill the slicer will report less). The `.scad` files hold the same shapes for
changes in OpenSCAD; if the LiDAR height differs, change `LIDAR_PLANE_M` in `sim/demo_course.py` and run the generator again.
No print: the obstacle and the two parked cars (cardboard boxes), tape for the speed-zone corners if you want them visible.

## Set-up
1. Put the course along a straight wall: the wall is the kerb of the parking lane. Tape the start arrow 1 m from the room's back end
   and 4 m from the brake wall, pointing along the wall, 1.2 m from it. Keep the last 1.4 m before the end wall empty: the car turns
   there (its turning circle is ~1.3 m).
2. Place the corner marks, the two boxes (5 cm off the wall, 1.5 m gap), the obstacle box, the post and the goal cross by the table.
3. Car on the arrow, relay running, dashboard open: ORIGIN (resets the world frame to the start), then a slow look around (or one AUTO
   EXPLORE lap) so the map shows the room.
4. Draw the speed zone (Zones tool, 10 km/h) and the parking lane (Parking lane tool, over the corner marks).

## Demo script
1. **Parking:** Park tool, click in the gap, hold RT. Let go any time: the car stops (the driver keeps control).
2. **Click-to-go:** click the goal cross. The car pulls out, goes through the speed zone at the zone's limit, round the obstacle, past the
   thin post, to the goal. Point out the post on the map.
3. **Return to start:** the home tile. The car drives back to the arrow.
4. **Emergency brake:** steer straight at the end wall, full throttle; the gate brakes.
5. **(Optional) LiDAR-blind object:** crawl into the dark box. The bumper marks it on the map; it is then avoided.

## Twin check
`python tools/hmi_test.py demo_course` runs the whole script on the real dashboard against the twin (actuation noise on) and prints per
station: park inside the lane, kerb gap, heading; goal error and contacts; peak speed in the zone vs. its cap; thin post on the map; home
error; brake gap.

Latest twin result (26 Oct 2026, 4 runs, actuation noise on): parking 4/4 (3.1-4.9 cm from the kerb, 0 contacts), speed zone held
(peak 0.205-0.214 m/s against 0.198), thin post on the map 4/4, emergency brake 18-22 cm from the wall, click-to-go 3/4 (3.6-5.4 cm),
return to start 2/3; the whole course in one go 2/4. The failures are the planner giving up after three stalled re-plans close to a
wall or a parked car (TODO AK12), not the course. On the car: rehearse each station once; if the car says "stuck", take over with the
stick, back it off a little and click again.
