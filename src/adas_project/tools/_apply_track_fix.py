def edit(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        if old not in s:
            raise SystemExit(f"pattern not found in {path}: {old[:70]!r}")
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)


edit("adas/tracking.py", [
    ("    def __init__(self, gate=0.30, min_samples=4, max_missed=3, move_threshold=0.10, persist=3):",
     "    def __init__(self, gate=0.30, min_samples=4, max_missed=3, move_threshold=0.10, persist=3,\n                 max_fit_rms=0.03, max_range=3.2, max_speed=1.6):"),
    ("        self.persist = persist\n", "        self.persist = persist\n        self.max_fit_rms = max_fit_rms      # real movers move smoothly; jumpy fragments (wall ends) do not\n        self.max_range = max_range          # beyond this LiDAR positions are too coarse to trust a velocity\n        self.max_speed = max_speed\n"),
    ('''        vx = float(np.polyfit(ts, [h[1] for h in tr.hist], 1)[0])
        vy = float(np.polyfit(ts, [h[2] for h in tr.hist], 1)[0])
        tr.vel_rel = (vx, vy)
''', '''        hx = np.array([h[1] for h in tr.hist])
        hy = np.array([h[2] for h in tr.hist])
        (vx, bx), (vy, by) = np.polyfit(ts, hx, 1), np.polyfit(ts, hy, 1)
        vx, vy = float(vx), float(vy)
        tr.vel_rel = (vx, vy)
        rms = float(np.sqrt(np.mean((hx - (vx * ts + bx)) ** 2 + (hy - (vy * ts + by)) ** 2)))
        tr.fit_rms = rms
'''),
    ('''        if math.hypot(rx, ry) > self.move_threshold:
            tr.moving_count += 1''', '''        smooth = rms <= self.max_fit_rms and math.hypot(x, y) <= self.max_range and math.hypot(rx, ry) <= self.max_speed
        if smooth and math.hypot(rx, ry) > self.move_threshold:
            tr.moving_count += 1'''),
    ("    moving_count: int = 0\n", "    moving_count: int = 0\n    fit_rms: float = 0.0\n"),
])
print("ok")
