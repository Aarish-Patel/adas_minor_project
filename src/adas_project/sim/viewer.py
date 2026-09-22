"""Top-down viewer for the simulator.

    python -m sim.viewer                       drive the virtual car with the keyboard
    python -m sim.viewer --scenario wall --pwm 255 --at 1.3 --out snap.png
                                               run a scenario headless and save a picture

Keys: UP/DOWN = throttle/reverse, LEFT/RIGHT = steer, X = ADAS on/off, R = reset, ESC = quit
"""

import argparse
import math

import numpy as np

from adas.geometry import path_pose
from adas.vehicle_params import steer_to_delta

from .scenarios import (constant_driver,
                        curve_into_wall, head_on_wall, pedestrian_crossing, reverse_into_wall)
from .simulator import Simulator
from .world import Box, Cone, MovingCircle, Wall, World

LEVEL_COLOURS = ["#2e9e44", "#e6c229", "#f08a1c", "#d62828"]
LEVEL_NAMES = ["CLEAR", "CAUTION", "WARNING", "BRAKING"]


def demo_arena():
    w = World()
    w.add_room(-0.5, 4.5, -1.5, 1.5)
    w.add(Cone(1.4, 0.15, 0.03))
    w.add(Cone(2.4, -0.35, 0.03))
    w.add(Box(3.4, 0.7, 0.5, 0.25, 0.2))
    w.add(MovingCircle(2.0, 1.3, 0.0, -0.15, 0.05))
    return w, (0.0, 0.0, 0.0)


def draw(ax, sim, title=""):
    import matplotlib.patches as mp

    ax.clear()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    for x1, y1, x2, y2 in sim.world.segments():
        ax.plot([x1, x2], [y1, y2], color="#222", lw=2)
    for cx, cy, r in sim.world.circles():
        ax.add_patch(mp.Circle((cx, cy), r, color="#d97706"))

    c = sim.car
    if len(sim.points):
        sx, sy, sth = sim.scan_pose
        px, py = sim.points[:, 0], sim.points[:, 1]
        wx = sx + math.cos(sth) * px - math.sin(sth) * py
        wy = sy + math.sin(sth) * px + math.cos(sth) * py
        ax.scatter(wx, wy, s=3, color="#3b82f6", alpha=0.6)

    colour = LEVEL_COLOURS[sim.level]
    p = sim.p
    body = np.array([[p.rear_x, -p.width / 2], [p.front_x, -p.width / 2],
                     [p.front_x, p.width / 2], [p.rear_x, p.width / 2]])
    ct, st = math.cos(c.theta), math.sin(c.theta)
    body_w = np.column_stack([c.x + ct * body[:, 0] - st * body[:, 1],
                              c.y + st * body[:, 0] + ct * body[:, 1]])
    ax.add_patch(mp.Polygon(body_w, closed=True, fc=colour, ec="black", alpha=0.85))
    ax.plot([c.x, c.x + ct * p.front_x], [c.y, c.y + st * p.front_x], color="black", lw=1)

    direction = sim.info.get("direction", 1)
    D = sim.info.get("D", math.inf)
    length = min(D if math.isfinite(D) else 1.0, 1.0)
    s = direction * np.linspace(0.0, max(length, 0.05), 40)
    delta = steer_to_delta(sim.applied_steer, p)
    gx, gy, _ = path_pose(delta, s, p)
    ax.plot(c.x + ct * gx - st * gy, c.y + st * gx + ct * gy, color=colour, lw=2, ls="--")

    seg = sim.world.segments()
    circ = sim.world.circles()
    xs = [c.x] + list(seg[:, 0]) + list(seg[:, 2]) + list(circ[:, 0])
    ys = [c.y] + list(seg[:, 1]) + list(seg[:, 3]) + list(circ[:, 1])
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)

    D_text = f"{D * 100:.0f} cm" if math.isfinite(D) else "clear"
    status = "CRASH" if c.collided else LEVEL_NAMES[sim.level]
    ax.set_title(f"{title}  t={sim.t:.1f}s  speed={c.v:.2f} m/s  path clear: {D_text}  "
                 f"[{status}]  ADAS {'ON' if sim.adas_on else 'OFF'}", fontsize=9)


SCENARIOS = {
    "wall": lambda pwm: head_on_wall(pwm),
    "reverse": lambda pwm: reverse_into_wall(pwm),
    "curve": lambda pwm: curve_into_wall(pwm, 0.5),
    "pedestrian": lambda pwm: pedestrian_crossing(pwm, 0.4),
}


def snapshot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scn = SCENARIOS[args.scenario](args.pwm)
    world, start = scn.build()
    sim = Simulator(world, start, adas_on=not args.no_adas)
    while sim.t < args.at and not sim.car.collided:
        sim.step(*scn.driver(sim.t, sim))
    fig, ax = plt.subplots(figsize=(9, 5))
    draw(ax, sim, args.scenario)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


def interactive():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    state = {"keys": set(), "sim": None, "steer": 0.0}

    def reset(adas=True):
        world, start = demo_arena()
        state["sim"] = Simulator(world, start, adas_on=adas)
        state["steer"] = 0.0

    reset()
    fig, ax = plt.subplots(figsize=(10, 6))

    def on_key(event, down):
        key = event.key
        if key is None:
            return
        if down:
            state["keys"].add(key)
            if key == "x":
                state["sim"].adas_on = not state["sim"].adas_on
            elif key == "r":
                reset(state["sim"].adas_on)
            elif key == "escape":
                plt.close(fig)
        else:
            state["keys"].discard(key)

    fig.canvas.mpl_connect("key_press_event", lambda e: on_key(e, True))
    fig.canvas.mpl_connect("key_release_event", lambda e: on_key(e, False))

    def frame(_):
        sim, keys = state["sim"], state["keys"]
        pwm = 200 if "up" in keys else -200 if "down" in keys else 0
        target = 1.0 if "left" in keys else -1.0 if "right" in keys else 0.0
        state["steer"] += (target - state["steer"]) * 0.25
        for _ in range(5):
            sim.step(state["steer"], pwm)
        draw(ax, sim, "demo arena")

    anim = FuncAnimation(fig, frame, interval=50, cache_frame_data=False)
    plt.show()
    return anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--pwm", type=int, default=255)
    ap.add_argument("--at", type=float, default=1.5, help="simulation time to capture (s)")
    ap.add_argument("--out", default="snapshot.png")
    ap.add_argument("--no-adas", action="store_true")
    args = ap.parse_args()

    if args.scenario:
        snapshot(args)
    else:
        interactive()


if __name__ == "__main__":
    main()
