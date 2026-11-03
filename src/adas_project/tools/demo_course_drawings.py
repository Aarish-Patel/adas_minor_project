"""Build package for the demo course (TODO AL7; layout sim/demo_course.py, parts tools/demo_course_parts.py, guide docs/demo_course.md).

    python tools/demo_course_drawings.py        # writes docs/demo_course/: layout_plan.png, layout_3d.png, parts_printed.png,
                                                # parts_cardboard.png, placement.csv, course_layout.dxf, course_assembly.scad

Everything a mechanical engineer needs to CAD the course and make its parts, generated from the same geometry as the twin world and the
STL files, so drawings, CAD files and simulation cannot disagree. Units mm. Datum: the start mark = the car's rear-axle centre at the
start, x along the kerb wall towards the end (brake) wall, y to the left (away from the kerb wall), z up from the floor.
"""
import csv
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "demo_course")

import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle           # noqa: E402

import sim.demo_course as D                                          # noqa: E402
from tools.demo_course_parts import parts                            # noqa: E402

MM = 1000.0
INK, GREY, BLUE, ORANGE, RED, GREEN = "#1f2933", "#7b8794", "#2563eb", "#d97706", "#dc2626", "#059669"


def mm(v):
    return v * MM


def rot(poly, deg, dx=0.0, dy=0.0):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(dx + c * x - s * y, dy + s * x + c * y) for x, y in poly]


# ------------------------------------------------------------------ the course as a list of objects (mm, datum = start mark)
def objects():
    """[dict(id, name, kind, x, y, L, W, H, rot, shape, note)] - x, y = centre (markers: their reference point), L along x, W along y."""
    x0, x1, y0, y1 = (mm(v) for v in D.ROOM)
    bh, ph = mm(D.BOX_HEIGHT_M), mm(D.PANEL_HEIGHT_M)
    out = [dict(id="W1", name="kerb wall (parking side)", kind="wall", x=(x0 + x1) / 2, y=y0, L=x1 - x0, W=0, H=ph, rot=0, shape="wall",
                note="a real straight wall, or cardboard panels; the parking lane's kerb"),
           dict(id="W2", name="end wall = brake wall", kind="wall", x=x1, y=(y0 + y1) / 2, L=0, W=y1 - y0, H=ph, rot=0, shape="wall",
                note="a real wall, or panels; the car brakes in front of it"),
           dict(id="W3", name="side wall", kind="wall", x=(x0 + x1) / 2, y=y1, L=x1 - x0, W=0, H=ph, rot=0, shape="wall",
                note="recommended (the twin has it): a real wall or panels"),
           dict(id="W4", name="back wall", kind="wall", x=x0, y=(y0 + y1) / 2, L=0, W=y1 - y0, H=ph, rot=0, shape="wall",
                note="recommended (the twin has it): a real wall or panels")]
    for i, (cx, cy, ln, wd) in enumerate(D.PARKED, 1):
        out.append(dict(id=f"C{i}", name=f"parked car {i} (cardboard box)", kind="cardboard", x=mm(cx), y=mm(cy), L=mm(ln), W=mm(wd), H=bh,
                        rot=0, shape="box", note="any closed box of this footprint, standing upright"))
    cx, cy, ln, wd = D.OBSTACLE
    out.append(dict(id="O", name="obstacle (cardboard box)", kind="cardboard", x=mm(cx), y=mm(cy), L=mm(ln), W=mm(wd), H=bh, rot=0,
                    shape="box", note="click-to-go plans round it"))
    cx, cy, ln, wd = D.DARK_BOX
    out.append(dict(id="D", name="dark box (optional)", kind="cardboard", x=mm(cx), y=mm(cy), L=mm(ln), W=mm(wd), H=bh, rot=0,
                    shape="box", note="covered in matte black felt; only for the LiDAR-blind / virtual bumper station, removed otherwise"))
    px, py, pr = D.THIN_POST
    post_h = max(v[2] for t in parts()["thin_post"][0] for v in t)
    out.append(dict(id="T", name="thin post (printed thin_post.stl)", kind="print", x=mm(px), y=mm(py), L=mm(2 * pr), W=mm(2 * pr), H=post_h,
                    rot=0, shape="post", note="Ø16 tube on a Ø60 x 2 foot"))
    out.append(dict(id="S", name="start arrow (printed marker_start.stl)", kind="print", x=0.0, y=0.0, L=120, W=50, H=1.2, rot=0,
                    shape="arrow", note="reference point = middle of the arrow's tail edge = the car's rear-axle centre; points along +x"))
    gx, gy = D.GOAL
    out.append(dict(id="G", name="goal cross (printed marker_goal.stl)", kind="print", x=mm(gx), y=mm(gy), L=60, W=60, H=1.2, rot=0,
                    shape="cross", note="reference point = the cross's centre (the car's rear axle arrives over it)"))
    (lx0, lyr), (lx1, lyk) = D.LANE
    for i, ((x, y), r) in enumerate(zip([(lx0, lyr), (lx1, lyr), (lx0, lyk), (lx1, lyk)], D.CORNER_ROT_DEG), 1):
        out.append(dict(id=f"L{i}", name=f"lane corner {i} (printed marker_corner.stl)", kind="print", x=mm(x), y=mm(y), L=60, W=60, H=1.2,
                        rot=r, shape="corner", note="reference point = the L's outer corner = the lane corner; legs along the lane edges"))
    z = D.SPEED_ZONE
    out.append(dict(id="Z", name="speed zone 10 km/h (optional tape outline)", kind="tape", x=mm(z["x0"] + z["x1"]) / 2, y=mm(z["y0"] + z["y1"]) / 2,
                    L=mm(z["x1"] - z["x0"]), W=mm(z["y1"] - z["y0"]), H=0, rot=0, shape="zone",
                    note="only drawn on the dashboard; tape on the floor is for the audience"))
    out.append(dict(id="P", name="parking lane (optional tape outline)", kind="tape", x=mm(lx0 + lx1) / 2, y=mm(lyr + lyk) / 2,
                    L=mm(lx1 - lx0), W=mm(abs(lyk - lyr)), H=0, rot=0, shape="zone", note="drawn on the dashboard over the four corner markers"))
    return out


def from_walls(o):
    """Distances from the two reference walls (what a tape measure reads): end wall to the object's nearest face, kerb wall likewise."""
    x1, y0 = mm(D.ROOM[1]), mm(D.ROOM[2])
    if o["shape"] == "wall":
        return None, None
    half_l, half_w = (o["L"] / 2, o["W"] / 2) if o["shape"] in ("box", "zone", "post") else (0.0, 0.0)
    return round(x1 - (o["x"] + half_l)), round((o["y"] - half_w) - y0)


# ------------------------------------------------------------------ helpers for the drawings
def dim(ax, p, q, off, text=None, side=1, fs=7, color=INK):
    """A dimension line between p and q (mm), offset `off` perpendicular to it, with extension lines and the length as text."""
    (x1, y1), (x2, y2) = p, q
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy)
    nx, ny = -dy / ln * off, dx / ln * off
    a, b = (x1 + nx, y1 + ny), (x2 + nx, y2 + ny)
    ax.plot([x1, a[0]], [y1, a[1]], color=color, lw=0.4)
    ax.plot([x2, b[0]], [y2, b[1]], color=color, lw=0.4)
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90 or ang <= -90:
        ang += 180
    m = ((a[0] + b[0]) / 2 + nx / max(abs(off), 1e-9) * 0 , (a[1] + b[1]) / 2)
    ax.text(m[0], m[1], text or f"{ln:.0f}", rotation=ang, ha="center", va="bottom" if side > 0 else "top", fontsize=fs, color=color,
            rotation_mode="anchor", bbox=dict(fc="white", ec="none", pad=0.4))


def draw_course(ax, labels=True):
    x0, x1, y0, y1 = (mm(v) for v in D.ROOM)
    t = 40
    for xa, ya, w, h in ((x0 - t, y0 - t, x1 - x0 + 2 * t, t), (x0 - t, y1, x1 - x0 + 2 * t, t), (x0 - t, y0, t, y1 - y0), (x1, y0, t, y1 - y0)):
        ax.add_patch(Rectangle((xa, ya), w, h, fc="#cbd2d9", ec=INK, lw=0.8, hatch="////"))
    for o in objects():
        s = o["shape"]
        if s == "box":
            dark = o["id"] == "D"
            ax.add_patch(Rectangle((o["x"] - o["L"] / 2, o["y"] - o["W"] / 2), o["L"], o["W"], fc="#111827" if dark else "#e4c99b",
                                   ec=INK, lw=0.8, ls="--" if dark else "-", alpha=0.85 if dark else 1.0))
        elif s == "post":
            ax.add_patch(Circle((o["x"], o["y"]), 30, fc="#bfdbfe", ec=BLUE, lw=0.6))
            ax.add_patch(Circle((o["x"], o["y"]), 8, fc=BLUE, ec=INK, lw=0.6))
        elif s in ("arrow", "cross", "corner"):
            name = {"arrow": "marker_start", "cross": "marker_goal", "corner": "marker_corner"}[s]
            poly = _outline(name)
            ax.add_patch(Polygon(rot(poly, o["rot"], o["x"], o["y"]), closed=True, fc=GREEN, ec=INK, lw=0.5))
        elif s == "zone":
            col = BLUE if o["id"] == "Z" else ORANGE
            ax.add_patch(Rectangle((o["x"] - o["L"] / 2, o["y"] - o["W"] / 2), o["L"], o["W"], fc="none", ec=col, lw=1.0, ls=":"))
    # the RC car on the start mark (outline) and its LiDAR
    r, f, w = (mm(v) for v in D.CAR_BODY)
    ax.add_patch(Rectangle((r, -w / 2), f - r, w, fc="none", ec=RED, lw=1.0, ls="-."))
    ax.add_patch(Circle((mm(D.LIDAR_X_M), 0), 35, fc="none", ec=RED, lw=0.8))
    if labels:
        for o in objects():
            if o["shape"] == "wall":
                continue
            dy = {"Z": 380, "P": -10, "S": -150, "T": 70, "G": 60}.get(o["id"], 0)
            ax.text(o["x"] + (90 if o["id"] in ("T", "G") else 0), o["y"] + dy, o["id"], fontsize=9, weight="bold",
                    color=INK if o["id"] != "D" else "white", ha="center", va="center")


def _outline(name):
    """The flat marker's outline (mm, its own frame) - the polygon the STL was extruded from."""
    import tools.demo_course_parts as P
    src = P.parts()[name][2]
    pts = src[src.index("polygon(") + 8: src.rindex(")")]
    import ast
    return [tuple(p) for p in ast.literal_eval(pts)]


# ------------------------------------------------------------------ 1) dimensioned plan
def plan(path):
    x0, x1, y0, y1 = (mm(v) for v in D.ROOM)
    fig, ax = plt.subplots(figsize=(16.5, 11.7))                      # A3 landscape
    draw_course(ax)
    # overall size and the start mark from the walls
    dim(ax, (x0, y0 - 40), (x1, y0 - 40), -380, "5000 (inside, back wall to end wall)")
    dim(ax, (x1 + 40, y0), (x1 + 40, y1), -330, "2400 (inside)", side=-1)
    dim(ax, (x0, 0), (0, 0), 0.0001, "1000", fs=7)
    dim(ax, (0, y0), (0, 0), 300, "1200", side=-1)
    dim(ax, (0, -640), (x1, -640), 0.0001, "4000 start mark → end wall")
    # chain along the kerb wall: cars, gap, the free end
    c1, c2 = [o for o in objects() if o["id"] in ("C1", "C2")]
    xs = [0, c1["x"] - c1["L"] / 2, c1["x"] + c1["L"] / 2, c2["x"] - c2["L"] / 2, c2["x"] + c2["L"] / 2, x1]
    for a, b in zip(xs, xs[1:]):
        dim(ax, (a, y0), (b, y0), -230, f"{b - a:.0f}" + (" GAP" if abs(b - a - 1500) < 1 else ""))
    dim(ax, (c1["x"] + c1["L"] / 2 + 20, y0), (c1["x"] + c1["L"] / 2 + 20, c1["y"] - c1["W"] / 2), 60, "50", side=-1, fs=6)
    dim(ax, (c1["x"] - c1["L"] / 2, c1["y"] - c1["W"] / 2), (c1["x"] - c1["L"] / 2, c1["y"] + c1["W"] / 2), 70, "200", side=-1, fs=6)
    # the lane and its markers
    (lx0, lyr), (lx1, lyk) = ((mm(a), mm(b)) for a, b in D.LANE)
    dim(ax, (lx0, lyr), (lx1, lyr), 90, "lane 3000")
    dim(ax, (lx1, lyk), (lx1, lyr), -90, "400", side=-1, fs=6)
    dim(ax, (lx1 + 150, y0), (lx1 + 150, lyk), 0.0001, "30", fs=6)
    dim(ax, (0, lyr + 160), (lx0, lyr + 160), 0.0001, "100", fs=6)
    # obstacle, post, goal, dark box: ordinate dimensions from the datum
    for o in objects():
        if o["id"] in ("O", "T", "G", "D"):
            ax.plot([o["x"]], [o["y"]], "+", color=RED, ms=8, mew=1.0)
            left = o["id"] == "D"
            ax.text(o["x"] + (-180 if left else 50), o["y"] - 70, f"x {o['x']:.0f}\ny {o['y']:+.0f}", fontsize=7, color=RED, va="top",
                    ha="right" if left else "left")
    o = next(o for o in objects() if o["id"] == "O")
    dim(ax, (o["x"] - 150, o["y"] + 150), (o["x"] + 150, o["y"] + 150), 60, "300", fs=6)
    dim(ax, (o["x"] - 150, o["y"] - 150), (o["x"] - 150, o["y"] + 150), 60, "300", fs=6, side=-1)
    t = next(o for o in objects() if o["id"] == "T")
    dim(ax, (t["x"] - 200, t["y"]), (t["x"] - 200, y1), 0.0001, "150", fs=6)
    d = next(o for o in objects() if o["id"] == "D")
    dim(ax, (d["x"] + 125, d["y"] - 160), (x1, d["y"] - 160), 0.0001, "125", fs=6)
    dim(ax, (d["x"] - 160, d["y"] + 125), (d["x"] - 160, y1), 0.0001, "75", fs=6)
    z = next(o for o in objects() if o["id"] == "Z")
    dim(ax, (z["x"] - z["L"] / 2, z["y"] + z["W"] / 2), (z["x"] + z["L"] / 2, z["y"] + z["W"] / 2), 60, "zone 800 (x 400..1200)", fs=6)
    dim(ax, (z["x"] - z["L"] / 2, z["y"] - z["W"] / 2), (z["x"] - z["L"] / 2, z["y"] + z["W"] / 2), 60, "1000 (y ±500)", fs=6, side=-1)
    # datum symbol and axes
    ax.annotate("", xy=(450, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    ax.annotate("", xy=(0, 450), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    ax.text(470, -30, "+x", color=RED, fontsize=9, va="top")
    ax.text(30, 470, "+y", color=RED, fontsize=9)
    ax.text(-60, 60, "DATUM (0,0)\nstart mark = rear-axle\ncentre of the car", color=RED, fontsize=7, ha="right")
    ax.text(x1 + 120, 0, "END WALL\n= BRAKE WALL", rotation=90, ha="left", va="center", fontsize=8, weight="bold")
    ax.text((x0 + x1) / 2, y0 - 120, "KERB WALL (parking side) - reference wall for y", ha="center", va="top", fontsize=8, weight="bold")
    ax.text((x0 + x1) / 2, y1 + 70, "SIDE WALL", ha="center", fontsize=8, weight="bold")
    ax.text(x0 - 90, 0, "BACK WALL", rotation=90, ha="right", va="center", fontsize=8, weight="bold")
    ax.text(3300, -300, "keep x 2600..4000\nfree: the car turns here\n(turning circle ~1300)", fontsize=7, color=GREY, ha="center",
            style="italic")
    key = ("S start arrow (print)   G goal cross (print)   L1-L4 lane corners (print)   T thin post (print)\n"
           "C1/C2 parked cars, O obstacle (cardboard 330x200 / 300x300, H 250)   D dark box (optional, black felt)\n"
           "Z speed zone, P parking lane (tape, optional)   red dash-dot: the RC car on the start mark (330 x 200), circle = LiDAR")
    ax.text(x0 - 40, y0 - 560, key, fontsize=7.5, va="top")
    ax.text(x0 + 3000, y0 - 560, "All dimensions mm. Floor inside the walls.\nTolerances: positions ±10, the parking gap 1500 ±20,\n"
            "walls straight ±10 over their length.\nHeights: boxes 250, post 240, wall panels 300\n(all ≥ LiDAR scan plane 200 + 30).",
            fontsize=7.5, va="top", ha="left")
    ax.set_xlim(x0 - 500, x1 + 650)
    ax.set_ylim(y0 - 950, y1 + 250)
    ax.set_aspect("equal")
    ax.set_xticks(range(-1000, 4001, 500))
    ax.set_yticks(range(-1000, 1001, 500))
    ax.tick_params(labelsize=7)
    ax.grid(color="#e4e7eb", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("RC-car ADAS demo course - plan view (top), dimensions in mm, datum = start mark", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2) 3D view
def _box3d(ax, cx, cy, L, W, H, z0=0.0, color="#e4c99b", alpha=1.0, edge=INK):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    x0, x1, y0, y1, z1 = cx - L / 2, cx + L / 2, cy - W / 2, cy + W / 2, z0 + H
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [[v[i] for i in f] for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=edge, linewidths=0.4, alpha=alpha))


def _cyl3d(ax, cx, cy, r, h, z0=0.0, color=BLUE):
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    a = np.linspace(0, 2 * np.pi, 25)
    side = [[(cx + r * math.cos(a[i]), cy + r * math.sin(a[i]), z0), (cx + r * math.cos(a[i + 1]), cy + r * math.sin(a[i + 1]), z0),
             (cx + r * math.cos(a[i + 1]), cy + r * math.sin(a[i + 1]), z0 + h), (cx + r * math.cos(a[i]), cy + r * math.sin(a[i]), z0 + h)]
            for i in range(len(a) - 1)]
    top = [[(cx + r * math.cos(t), cy + r * math.sin(t), z0 + h) for t in a]]
    ax.add_collection3d(Poly3DCollection(side + top, facecolor=color, edgecolor="none"))


def view3d(path):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    x0, x1, y0, y1 = (mm(v) for v in D.ROOM)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [0, 0, 0, 0, 0], color=INK, lw=0.8)        # the floor's edge (a floor polygon
    ph, t = mm(D.PANEL_HEIGHT_M), 20                                                                   # would be drawn over the boxes)
    for cx, cy, L, W, a in (((x0 + x1) / 2, y0 - t / 2, x1 - x0, t, 0.55), (x1 + t / 2, 0, t, y1 - y0, 0.55),
                            ((x0 + x1) / 2, y1 + t / 2, x1 - x0, t, 0.08), (x0 - t / 2, 0, t, y1 - y0, 0.08)):
        _box3d(ax, cx, cy, L, W, ph, color="#9aa5b1", alpha=a, edge="#52606d")
    for o in objects():
        if o["shape"] == "box":
            _box3d(ax, o["x"], o["y"], o["L"], o["W"], o["H"], color="#1f2933" if o["id"] == "D" else "#e4c99b")
        elif o["shape"] == "post":
            _cyl3d(ax, o["x"], o["y"], 30, 2, color="#93c5fd")
            _cyl3d(ax, o["x"], o["y"], 8, o["H"], color=BLUE)
        elif o["shape"] in ("arrow", "cross", "corner"):
            name = {"arrow": "marker_start", "cross": "marker_goal", "corner": "marker_corner"}[o["shape"]]
            p = rot(_outline(name), o["rot"], o["x"], o["y"])
            ax.add_collection3d(Poly3DCollection([[(x, y, 2) for x, y in p]], facecolor=GREEN, edgecolor=INK, lw=0.3))
        elif o["shape"] == "zone":
            xa, xb, ya, yb = o["x"] - o["L"] / 2, o["x"] + o["L"] / 2, o["y"] - o["W"] / 2, o["y"] + o["W"] / 2
            ax.plot([xa, xb, xb, xa, xa], [ya, ya, yb, yb, ya], [1, 1, 1, 1, 1], color=BLUE if o["id"] == "Z" else ORANGE, lw=1, ls=":")
    r, f, w = (mm(v) for v in D.CAR_BODY)
    _box3d(ax, (r + f) / 2, 0, f - r, w, 120, z0=30, color="#f8fafc", edge=RED)
    _cyl3d(ax, mm(D.LIDAR_X_M), 0, 38, 50, z0=mm(D.LIDAR_PLANE_M) - 25, color="#334155")
    for o in objects():
        if o["shape"] in ("box", "post"):
            ax.text(o["x"], o["y"], o["H"] + 60, o["id"], fontsize=10, weight="bold", ha="center")
    ax.text(0, -150, 5, "S", fontsize=10, weight="bold")
    ax.text(mm(D.GOAL[0]), mm(D.GOAL[1]) - 120, 5, "G", fontsize=10, weight="bold")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0 - 500, y1 + 500)
    ax.set_zlim(0, 600)
    ax.set_box_aspect((x1 - x0, y1 - y0 + 1000, 600))
    ax.view_init(elev=30, azim=128)                                # from the back / side-wall corner: kerb and end wall behind
    ax.set_xlabel("x mm", fontsize=8)
    ax.set_ylabel("y mm", fontsize=8)
    ax.set_zlabel("z mm", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title("Demo course - 3D view (walls 300, boxes 250, post 240 high; the car on the start mark)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 3) printed parts
def printed_parts(path):
    P = parts()
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 11.7))
    # thin post: half section (revolved profile)
    ax = axs[0, 0]
    prof = eval(P["thin_post"][2][P["thin_post"][2].index("polygon(") + 8: P["thin_post"][2].rindex(")")])
    ax.add_patch(Polygon(prof, closed=True, fc="#bfdbfe", ec=INK, lw=0.8, hatch="////"))
    ax.add_patch(Polygon([(-x, y) for x, y in prof], closed=True, fc="#bfdbfe", ec=INK, lw=0.8, hatch="////"))
    h = max(y for _, y in prof)
    ax.plot([0, 0], [-10, h + 10], color=GREY, lw=0.6, ls="-.")
    dim(ax, (-30, 0), (30, 0), -12, "Ø60 foot", side=-1)
    dim(ax, (-8, h), (8, h), 10, "Ø16")
    dim(ax, (30, 0), (30, 2), 20, "2", fs=6)
    dim(ax, (8, 0), (8, h), -32, f"{h:.0f}", side=-1)
    ax.text(10, h * 0.55, "wall 0.8\n(ID Ø14.4,\nopen both ends)", fontsize=7)
    ax.set_title("thin_post.stl - section A-A (revolved)\n1 off, PLA, 2 perimeters, 0 % infill, brim 5, upright", fontsize=9)
    ax.set_xlim(-60, 60), ax.set_ylim(-35, h + 30)
    # panel foot: end view of the U-channel
    ax = axs[0, 1]
    foot = eval(P["panel_foot"][2][P["panel_foot"][2].index("polygon(") + 8: P["panel_foot"][2].rindex(")")])
    ax.add_patch(Polygon(foot, closed=True, fc="#d9e2ec", ec=INK, lw=0.8))
    dim(ax, (0, 0), (50, 0), -6, "50", side=-1)
    dim(ax, (50, 0), (50, 2), 8, "2", fs=6)
    dim(ax, (22.4, 22), (27.6, 22), 6, "5.2 slot")
    ax.annotate("wall 2.4 (both)", xy=(29, 15), xytext=(35, 27), fontsize=7, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    dim(ax, (27.6, 2), (27.6, 22), -12, "20", side=-1)
    ax.text(25, -15, "extruded 30 deep (into the page). The 5.2 slot takes 3-5 cardboard:\nwedge thinner board with a strip of card.",
            ha="center", fontsize=7)
    ax.set_title("panel_foot.stl - end view\n2 per panel + 1 per joint, PLA, 3 perimeters, 15 % infill, on its base", fontsize=9)
    ax.set_xlim(-8, 62), ax.set_ylim(-20, 34)
    # markers
    for ax, name, title, dims in ((axs[0, 2], "marker_start", "marker_start.stl - top view, 1 off, t = 1.2", "arrow"),
                                  (axs[1, 0], "marker_corner", "marker_corner.stl - top view, 4 off, t = 1.2", "corner"),
                                  (axs[1, 1], "marker_goal", "marker_goal.stl - top view, 1 off, t = 1.2", "cross")):
        poly = _outline(name)
        ax.add_patch(Polygon(poly, closed=True, fc="#c6f6d5", ec=INK, lw=0.8))
        if dims == "arrow":
            dim(ax, (0, -25), (120, -25), -10, "120", side=-1)
            dim(ax, (0, -12), (0, 12), 10, "24", side=-1)
            dim(ax, (120, -25), (120, 25), -14, "50")
            dim(ax, (80, 25), (120, 25), 8, "40")
            ax.plot([0], [0], "o", color=RED, ms=5)
            ax.text(-4, 30, "reference point (red dot):\ncar's rear-axle centre", color=RED, fontsize=7, ha="right")
            ax.set_xlim(-50, 145), ax.set_ylim(-55, 50)
        elif dims == "corner":
            dim(ax, (0, 0), (60, 0), -8, "60", side=-1)
            dim(ax, (0, 0), (0, 60), 8, "60")
            dim(ax, (60, 0), (60, 12), -8, "12", fs=6)
            ax.plot([0], [0], "o", color=RED, ms=5)
            ax.text(3, -14, "reference point: lane corner", color=RED, fontsize=7)
            ax.set_xlim(-25, 85), ax.set_ylim(-25, 75)
        else:
            dim(ax, (-30, -30), (30, -30), -6, "60", side=-1)
            dim(ax, (-6, 30), (6, 30), 6, "12", fs=6)
            ax.plot([0], [0], "o", color=RED, ms=5)
            ax.text(8, 8, "reference point:\ngoal", color=RED, fontsize=7)
            ax.set_xlim(-45, 45), ax.set_ylim(-45, 45)
        ax.set_title(title + "\nPLA, 100 % infill, contrasting colour, flat on the bed", fontsize=9)
    # print summary
    ax = axs[1, 2]
    ax.axis("off")
    from tools.demo_course_parts import PLA_G_PER_CM3, volume_mm3
    rows = [(n, c, volume_mm3(t) / 1000 * PLA_G_PER_CM3) for n, (t, c, _, _) in P.items()]
    txt = "Printer: Creality K1 (bed 220 x 220, height 250)\nPLA, 0.4 nozzle, 0.2 layers, no supports\n\n"
    txt += "\n".join(f"{n:15s} x{c}   {g:5.1f} g each" for n, c, g in rows)
    txt += f"\n\ntotal (solid volume) {sum(c * g for _, c, g in rows):.0f} g\n(+ 1 panel foot per extra panel joint)\n\n"
    txt += "Files: docs/demo_course/<part>.stl (mm)\nand <part>.scad (OpenSCAD source)\nfrom tools/demo_course_parts.py"
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=9)
    for a in axs.flat[:5]:
        a.set_aspect("equal")
        a.tick_params(labelsize=6)
        a.grid(color="#eef0f2", lw=0.4)
    fig.suptitle("Printed parts - dimensions in mm", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 4) cardboard parts
def cardboard_parts(path):
    bh, ph, pl = mm(D.BOX_HEIGHT_M), mm(D.PANEL_HEIGHT_M), mm(D.PANEL_LENGTH_M)
    fig, axs = plt.subplots(2, 2, figsize=(16.5, 11.7))
    for ax, (title, L, W, H, col, note) in zip(axs.flat[:3], (
            ("C1, C2 parked cars - 2 off", 330, 200, bh, "#e4c99b", "closed box, upright; long side along the kerb wall,\n50 from it"),
            ("O obstacle - 1 off", 300, 300, bh, "#e4c99b", "closed box"),
            ("D dark box (optional) - 1 off", 250, 250, bh, "#1f2933", "closed box covered in matte black felt\n(all sides facing the course);"
                                                                         " check on the\ndashboard that the LiDAR returns no points"))):
        ax.add_patch(Rectangle((0, 0), L, H, fc=col, ec=INK, lw=0.8))                           # front view
        ax.add_patch(Rectangle((L + 120, 0), W, H, fc=col, ec=INK, lw=0.8))                     # side view
        ax.add_patch(Rectangle((0, H + 120), L, W, fc=col, ec=INK, lw=0.8))                     # top view
        dim(ax, (0, 0), (L, 0), -25, f"{L}", side=-1)
        dim(ax, (L + 120, 0), (L + 120 + W, 0), -25, f"{W}", side=-1)
        dim(ax, (0, 0), (0, H), 25, f"{H:.0f}")
        dim(ax, (L, H + 120), (L, H + 120 + W), -25, f"{W}", side=-1)
        ax.text(L / 2, -95, "front (x)", ha="center", fontsize=7)
        ax.text(L + 120 + W / 2, -95, "side (y)", ha="center", fontsize=7)
        ax.text(L / 2, H + 120 + W + 20, "top", ha="center", fontsize=7)
        ax.text(L + 160 + W, H + 120, note, fontsize=7.5, va="bottom")
        ax.set_title(title + f": {L} x {W} x {H:.0f} (L x W x H), tolerance ±10, height ≥ 230", fontsize=9)
        ax.set_xlim(-80, L + W + 700), ax.set_ylim(-140, H + W + 220)
    ax = axs[1, 1]                                                                              # wall panel with its feet
    ax.add_patch(Rectangle((0, 2), pl, ph, fc="#cbd2d9", ec=INK, lw=0.8))
    for x in (0, pl - 30):
        ax.add_patch(Rectangle((x, 0), 30, 22, fc="#52606d", ec=INK, lw=0.6))
        ax.add_patch(Rectangle((x - 10, 0), 50, 2, fc="#52606d", ec=INK, lw=0.6))
    dim(ax, (0, 2), (pl, 2), -60, f"{pl:.0f}", side=-1)
    dim(ax, (pl, 2), (pl, ph + 2), -40, f"{ph:.0f}", side=-1)
    ax.text(pl / 2, ph / 2, "corrugated cardboard 3-5 thick\n(flutes vertical = stiffer)", ha="center", fontsize=8)
    ax.text(15, 40, "panel_foot\n30 long", fontsize=6.5)
    ax.text(0, -160, "Only where the room has no wall. Butt joints: one foot straddles the joint (15 of each panel in it).\n"
            "Kerb + end wall from a room corner, side + back wall from panels: 5 + 3 = 8 panels (one cut to 400), about 11 feet.\n"
            "Corners: each panel has its own foot 20 from the corner; tape the corner on the outside.", fontsize=7.5, va="top")
    ax.set_title(f"Wall panel (only where there is no wall): {pl:.0f} x {ph:.0f}, 2 panel_foot per panel", fontsize=9)
    ax.set_xlim(-80, pl + 120), ax.set_ylim(-330, ph + 80)
    for a in axs.flat:
        a.set_aspect("equal")
        a.tick_params(labelsize=6)
        a.grid(color="#eef0f2", lw=0.4)
    fig.suptitle("Cardboard parts - dimensions in mm (front / side / top views)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 5) CSV, DXF, OpenSCAD
def placement_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "kind", "x_mm", "y_mm", "rot_deg", "L_mm", "W_mm", "H_mm", "from_end_wall_mm", "from_kerb_wall_mm", "note"])
        for o in objects():
            e, k = from_walls(o)
            w.writerow([o["id"], o["name"], o["kind"], round(o["x"], 1), round(o["y"], 1), o["rot"], round(o["L"], 1), round(o["W"], 1),
                        round(o["H"], 1), "" if e is None else e, "" if k is None else k, o["note"]])


class _Dxf:
    """Minimal ASCII DXF (R12 entities: LINE, CIRCLE, TEXT; layers implicit) in mm."""

    def __init__(self):
        self.e = []

    def line(self, a, b, layer):
        self.e += ["0", "LINE", "8", layer, "10", f"{a[0]:.2f}", "20", f"{a[1]:.2f}", "30", "0", "11", f"{b[0]:.2f}", "21", f"{b[1]:.2f}", "31", "0"]

    def poly(self, pts, layer):
        for a, b in zip(pts, pts[1:] + pts[:1]):
            self.line(a, b, layer)

    def circle(self, c, r, layer):
        self.e += ["0", "CIRCLE", "8", layer, "10", f"{c[0]:.2f}", "20", f"{c[1]:.2f}", "30", "0", "40", f"{r:.2f}"]

    def text(self, p, h, s, layer="TEXT"):
        self.e += ["0", "TEXT", "8", layer, "10", f"{p[0]:.2f}", "20", f"{p[1]:.2f}", "30", "0", "40", f"{h:.2f}", "1", s]

    def save(self, path):
        head = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "9", "$INSUNITS", "70", "4", "0", "ENDSEC"]
        with open(path, "w", encoding="ascii") as f:
            f.write("\n".join(head + ["0", "SECTION", "2", "ENTITIES"] + self.e + ["0", "ENDSEC", "0", "EOF"]) + "\n")


def dxf(path):
    d = _Dxf()
    x0, x1, y0, y1 = (mm(v) for v in D.ROOM)
    d.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "WALLS_INSIDE")
    for o in objects():
        s = o["shape"]
        if s in ("box", "zone"):
            xa, xb, ya, yb = o["x"] - o["L"] / 2, o["x"] + o["L"] / 2, o["y"] - o["W"] / 2, o["y"] + o["W"] / 2
            d.poly([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], {"box": "CARDBOARD", "zone": "TAPE"}[s])
        elif s == "post":
            d.circle((o["x"], o["y"]), 30, "PRINTED")
            d.circle((o["x"], o["y"]), 8, "PRINTED")
        elif s in ("arrow", "cross", "corner"):
            name = {"arrow": "marker_start", "cross": "marker_goal", "corner": "marker_corner"}[s]
            d.poly(rot(_outline(name), o["rot"], o["x"], o["y"]), "PRINTED")
        if s != "wall":
            d.text((o["x"] + 20, o["y"] + 20), 40, o["id"])
    r, f, w = (mm(v) for v in D.CAR_BODY)
    d.poly([(r, -w / 2), (f, -w / 2), (f, w / 2), (r, w / 2)], "CAR_AT_START")
    d.circle((mm(D.LIDAR_X_M), 0), 35, "CAR_AT_START")
    d.line((0, 0), (300, 0), "DATUM")
    d.line((0, 0), (0, 300), "DATUM")
    d.text((20, -80), 30, "DATUM 0,0 (start mark, +x to the end wall, +y away from the kerb wall)", "DATUM")
    d.save(path)


def scad(path):
    P = parts()
    poly = lambda name: P[name][2][P[name][2].index("polygon("):].rstrip(";")
    lines = ["// Demo course assembly (generated by tools/demo_course_drawings.py) - mm, datum = start mark (car rear-axle centre),",
             "// +x towards the end (brake) wall, +y away from the kerb wall, z up. Open in OpenSCAD; export STL / 3MF / DXF (projection).",
             f"wall_h = {mm(D.PANEL_HEIGHT_M):.0f}; wall_t = 20;",
             f"module post() {{ rotate_extrude($fn=72) {poly('thin_post')}; }}",
             f"module marker_start() {{ linear_extrude(1.2) {poly('marker_start')}; }}",
             f"module marker_goal() {{ linear_extrude(1.2) {poly('marker_goal')}; }}",
             f"module marker_corner() {{ linear_extrude(1.2) {poly('marker_corner')}; }}",
             "module box(c, s) { translate([c[0] - s[0] / 2, c[1] - s[1] / 2, 0]) cube(s); }"]
    x0, x1, y0, y1 = (mm(v) for v in D.ROOM)
    lines += [f"color(\"Silver\", 0.5) {{ translate([{x0}, {y0} - wall_t, 0]) cube([{x1 - x0}, wall_t, wall_h]);   // W1 kerb wall",
              f"  translate([{x1}, {y0}, 0]) cube([wall_t, {y1 - y0}, wall_h]);                       // W2 end wall",
              f"  translate([{x0}, {y1}, 0]) cube([{x1 - x0}, wall_t, wall_h]);                       // W3 side wall",
              f"  translate([{x0} - wall_t, {y0}, 0]) cube([wall_t, {y1 - y0}, wall_h]); }}            // W4 back wall",
              f"color(\"WhiteSmoke\") translate([{x0}, {y0}, -5]) cube([{x1 - x0}, {y1 - y0}, 5]);   // floor"]
    for o in objects():
        s = o["shape"]
        if s == "box":
            c = "Black" if o["id"] == "D" else "BurlyWood"
            lines.append(f"color(\"{c}\") box([{o['x']:.1f}, {o['y']:.1f}], [{o['L']:.0f}, {o['W']:.0f}, {o['H']:.0f}]);   // {o['id']} {o['name']}")
        elif s == "post":
            lines.append(f"color(\"RoyalBlue\") translate([{o['x']:.1f}, {o['y']:.1f}, 0]) post();   // T")
        elif s in ("arrow", "cross", "corner"):
            m = {"arrow": "marker_start", "cross": "marker_goal", "corner": "marker_corner"}[s]
            lines.append(f"color(\"SeaGreen\") translate([{o['x']:.1f}, {o['y']:.1f}, 0]) rotate([0, 0, {o['rot']:.0f}]) {m}();   // {o['id']}")
    r, f, w = (mm(v) for v in D.CAR_BODY)
    lines.append(f"%translate([{r}, {-w / 2}, 30]) cube([{f - r}, {w}, 120]);   // the RC car on the start mark (reference only)")
    lines.append(f"%translate([{mm(D.LIDAR_X_M)}, 0, {mm(D.LIDAR_PLANE_M) - 25}]) cylinder(r=38, h=50, $fn=48);   // its LiDAR "
                 f"(scan plane z = {mm(D.LIDAR_PLANE_M):.0f})")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    plan(os.path.join(OUT, "layout_plan.png"))
    view3d(os.path.join(OUT, "layout_3d.png"))
    printed_parts(os.path.join(OUT, "parts_printed.png"))
    cardboard_parts(os.path.join(OUT, "parts_cardboard.png"))
    placement_csv(os.path.join(OUT, "placement.csv"))
    dxf(os.path.join(OUT, "course_layout.dxf"))
    scad(os.path.join(OUT, "course_assembly.scad"))
    for o in objects():
        e, k = from_walls(o)
        print(f"{o['id']:3s} {o['name']:45s} x {o['x']:7.1f}  y {o['y']:7.1f}  rot {o['rot']:5.0f}  {o['L']:.0f} x {o['W']:.0f} x {o['H']:.0f}"
              + ("" if e is None else f"   end wall {e}, kerb wall {k}"))
    print("written to", OUT)


if __name__ == "__main__":
    main()
