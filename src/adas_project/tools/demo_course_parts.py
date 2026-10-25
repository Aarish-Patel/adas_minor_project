"""Printable parts for the demo course (TODO AL3; layout sim/demo_course.py, guide docs/demo_course.md).

    python tools/demo_course_parts.py            # writes docs/demo_course/*.stl and *.scad, prints volume and filament per part

Sized for a Creality K1 (220 x 220 x 250 mm) and as little filament as possible: the parked "cars" and the obstacle are cardboard boxes
(no print); printed are only the parts that make the course repeatable or show a capability:
  thin_post        16 mm hollow post (2 walls) on a 60 mm foot - the thin object of the memory demo; tall enough for the LiDAR plane
  panel_foot       U-channel foot that holds 3-5 mm cardboard upright (a kerb / brake wall where there is no real wall), 2 per panel
  marker_start     flat arrow: where the car starts and which way it faces (return to start comes back here)
  marker_corner    flat L: the four corners of the parking lane (draw the lane on the map over them)
  marker_goal      flat cross: the click-to-go goal
Each STL is one watertight solid (an extruded polygon or a revolved profile), written directly - no CAD program needed; the .scad files
are the same shapes for editing in OpenSCAD.
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "demo_course")
PLA_G_PER_CM3 = 1.24


# ------------------------------------------------------------------ geometry -> triangles
def _area(poly):
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]))


def _triangulate(poly):
    """Ear clipping of a simple polygon (any orientation) -> index triangles, counter-clockwise."""
    pts = list(poly)
    idx = list(range(len(pts)))
    if _area(pts) < 0:
        idx.reverse()
    tris = []

    def inside(p, a, b, c):
        def cross(o, u, v):
            return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue                                                     # reflex corner
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
    tris.append(tuple(idx))
    return tris


def extrude(poly, height):
    """A polygon (x, y in mm) extruded from z = 0 to height -> triangles (outward normals)."""
    poly = list(poly)
    if _area(poly) < 0:
        poly.reverse()
    n = len(poly)
    tris = []
    for a, b, c in _triangulate(poly):
        tris.append(((*poly[a], height), (*poly[b], height), (*poly[c], height)))      # top, CCW from above
        tris.append(((*poly[c], 0.0), (*poly[b], 0.0), (*poly[a], 0.0)))              # bottom, reversed
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        tris.append(((x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height)))
        tris.append(((x0, y0, 0.0), (x1, y1, height), (x0, y0, height)))
    return tris


def revolve(profile, segments=72):
    """A closed profile [(r, z)] (r > 0, counter-clockwise in the r-z plane) revolved about z -> triangles (outward normals)."""
    if _area(profile) < 0:
        profile = profile[::-1]
    tris = []
    ring = lambda r, z, k: (r * math.cos(2 * math.pi * k / segments), r * math.sin(2 * math.pi * k / segments), z)
    for i in range(len(profile)):
        (r0, z0), (r1, z1) = profile[i], profile[(i + 1) % len(profile)]
        for k in range(segments):
            a, b = ring(r0, z0, k), ring(r1, z1, k)
            c, d = ring(r1, z1, k + 1), ring(r0, z0, k + 1)
            tris.append((a, d, c))
            tris.append((a, c, b))
    return tris


def volume_mm3(tris):
    return abs(sum((a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
                   for a, b, c in tris))


def write_stl(path, name, tris):
    with open(path, "w", encoding="ascii") as f:
        f.write(f"solid {name}\n")
        for a, b, c in tris:
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            f.write(f"  facet normal {nx / ln:.6f} {ny / ln:.6f} {nz / ln:.6f}\n    outer loop\n")
            for p in (a, b, c):
                f.write(f"      vertex {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


# ------------------------------------------------------------------ the parts (mm)
def parts():
    from sim.demo_course import LIDAR_PLANE_M
    post_h = min(240.0, (LIDAR_PLANE_M + 0.04) * 1000.0)         # reaches 4 cm above the scan plane; the K1 builds up to 250 mm
    r_out, wall, foot_r, foot_t = 8.0, 0.8, 30.0, 2.0
    post = [(r_out - wall, 0.0), (foot_r, 0.0), (foot_r, foot_t), (r_out, foot_t), (r_out, post_h), (r_out - wall, post_h)]
    slot, w, h, base_w, base_t, length = 5.2, 2.4, 20.0, 50.0, 2.0, 30.0
    xs0 = (base_w - slot) / 2.0
    foot = [(0, 0), (base_w, 0), (base_w, base_t), (xs0 + slot + w, base_t), (xs0 + slot + w, base_t + h), (xs0 + slot, base_t + h),
            (xs0 + slot, base_t), (xs0, base_t), (xs0, base_t + h), (xs0 - w, base_t + h), (xs0 - w, base_t), (0, base_t)]
    t = 1.2                                                      # flat markers: 1.2 mm (3 layers at 0.4, or 6 at 0.2)
    arrow = [(0, -12), (80, -12), (80, -25), (120, 0), (80, 25), (80, 12), (0, 12)]
    corner = [(0, 0), (60, 0), (60, 12), (12, 12), (12, 60), (0, 60)]
    cross = [(-6, -30), (6, -30), (6, -6), (30, -6), (30, 6), (6, 6), (6, 30), (-6, 30), (-6, 6), (-30, 6), (-30, -6), (-6, -6)]
    # the U profile is drawn in x-z and extruded along y, so the foot prints standing on its base (swapping y and z mirrors the solid:
    # the triangles are reversed to keep the normals outward)
    foot_tris = [tuple((x, y, z) for x, z, y in tri[::-1]) for tri in extrude(foot, length)]
    return {
        "thin_post": (revolve(post), 1, f"rotate_extrude($fn=72) polygon({[list(p) for p in post]});",
                      "post 16 mm x %.0f mm, 2 walls (0.8 mm), 60 mm foot; no infill needed" % post_h),
        "panel_foot": (foot_tris, 2, f"rotate([90,0,0]) linear_extrude({length}) polygon({[list(p) for p in foot]});",
                       "holds 3-5 mm cardboard; 2 per panel, prints standing on its 50 mm base"),
        "marker_start": (extrude(arrow, t), 1, f"linear_extrude({t}) polygon({[list(p) for p in arrow]});", "start arrow, 120 mm"),
        "marker_corner": (extrude(corner, t), 4, f"linear_extrude({t}) polygon({[list(p) for p in corner]});", "lane corner L, 60 mm"),
        "marker_goal": (extrude(cross, t), 1, f"linear_extrude({t}) polygon({[list(p) for p in cross]});", "goal cross, 60 mm"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    total = 0.0
    print(f"{'part':15s} {'count':>5s} {'cm3 each':>9s} {'g each':>7s} {'g total':>8s}")
    for name, (tris, count, scad, note) in parts().items():
        write_stl(os.path.join(OUT, f"{name}.stl"), name, tris)
        with open(os.path.join(OUT, f"{name}.scad"), "w", encoding="utf-8") as f:
            f.write(f"// {note} (generated by tools/demo_course_parts.py)\n{scad}\n")
        v = volume_mm3(tris) / 1000.0
        g = v * PLA_G_PER_CM3
        total += g * count
        print(f"{name:15s} {count:5d} {v:9.2f} {g:7.1f} {g * count:8.1f}   {note}")
    print(f"filament (solid volume, PLA {PLA_G_PER_CM3} g/cm3): {total:.0f} g in all")


if __name__ == "__main__":
    main()
