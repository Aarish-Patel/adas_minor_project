"""Demo course (TODO AL): the printed parts are closed solids that fit the Creality K1, the twin world matches the layout."""
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class PartsTest(unittest.TestCase):
    def test_watertight_outward_and_fits_k1(self):
        from tools.demo_course_parts import parts, volume_mm3
        for name, (tris, count, _scad, _note) in parts().items():
            edges = Counter()
            for t in tris:
                p = [tuple(round(c, 4) for c in v) for v in t]
                for i in range(3):
                    edges[(p[i], p[(i + 1) % 3])] += 1
            open_edges = [e for e, c in edges.items() if e[0] != e[1] and edges.get((e[1], e[0]), 0) != c]
            self.assertEqual(open_edges, [], name)
            signed = sum(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
                         for a, b, c in tris) / 6.0
            self.assertGreater(signed, 0.0, name)                       # normals outward
            xs, ys, zs = zip(*[v for t in tris for v in t])
            self.assertLessEqual(max(xs) - min(xs), 220.0, name)
            self.assertLessEqual(max(ys) - min(ys), 220.0, name)
            self.assertLessEqual(max(zs) - min(zs), 250.0, name)
            self.assertAlmostEqual(min(zs), 0.0, places=3, msg=name)    # sits on the bed
            self.assertLess(volume_mm3(tris) * count / 1000.0, 20.0, name)

    def test_post_reaches_above_the_scan_plane(self):
        from sim.demo_course import MIN_OBJECT_HEIGHT_M
        from tools.demo_course_parts import parts
        zs = [v[2] for t in parts()["thin_post"][0] for v in t]
        self.assertGreaterEqual(max(zs) / 1000.0, MIN_OBJECT_HEIGHT_M)


class WorldTest(unittest.TestCase):
    def test_world_and_layout(self):
        from sim import demo_course as dc
        from sim.hw_worlds import WORLDS
        self.assertIn("demo", WORLDS)
        w, start = dc.world()
        self.assertEqual(tuple(start), dc.START)
        x0, x1, y0, y1 = dc.ROOM
        for cx, cy, *_ in [dc.OBSTACLE, dc.THIN_POST, dc.DARK_BOX, dc.GOAL, dc.PARK_CLICK] + dc.PARKED:
            self.assertTrue(x0 < cx < x1 and y0 < cy < y1)
        gap = (dc.PARKED[1][0] - dc.PARKED[1][2] / 2) - (dc.PARKED[0][0] + dc.PARKED[0][2] / 2)
        self.assertGreater(gap, 1.1)                                    # an easy parallel slot for the 0.36 m car


if __name__ == "__main__":
    unittest.main()
