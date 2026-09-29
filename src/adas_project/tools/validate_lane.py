"""Does the real OpenCV lane detector, run on the 3D-rendered camera, recover the true lane position?

Server must be running (python server.py). Captures frames at known poses on the curving road and compares the
detected lane-centre offset and heading with ground truth.
"""
import base64, math, sys, time
import cv2, numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, ".")
from adas.lane import detect_lane_points, fit_lane
from adas.markers import Camera

cam = Camera()
centre = lambda x: 0.30 * math.sin(0.55 * x)
slope = lambda x: 0.30 * 0.55 * math.cos(0.55 * x)
POSES = [(1.0, 0.0), (1.0, 0.08), (2.0, -0.06), (3.0, 0.10), (4.5, 0.0), (6.0, -0.09)]    # (x, lateral offset from centre)

with sync_playwright() as pw:
    b = pw.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1000, "height": 600})
    pg.goto("http://localhost:8765/", wait_until="load"); time.sleep(2)
    pg.select_option("#scenario", "lane"); time.sleep(2.5)
    pg.evaluate("window.__send({type:'cmd', cmd:'pause'})"); time.sleep(0.5)
    print(f"{'x':>4} {'true offset':>12} {'detected':>10} {'true head':>10} {'detected':>9} {'lines':>6}")
    errs, herrs = [], []
    for k, (x, off) in enumerate(POSES):
        y = centre(x) + off                       # car left of centre by `off`
        th = math.atan(slope(x))                  # aligned with the road
        data = pg.evaluate("""([x, y, th]) => {
            const snap = { car: { x, y, th, v: 0, wl: 0, wr: 0 }, adas: { level: 0, D: -1, mode: 'off' }, cmd: { out: 0 },
                           dyn: [], tracks: [], path: [[x, y], [x + 0.05, y]], lidar: null };
            window.__world.update(snap, 0.03);
            const img = window.__world.captureCarCamera(); const c = document.createElement('canvas');
            c.width = 640; c.height = 480; c.getContext('2d').putImageData(img, 0, 0); return c.toDataURL('image/png'); }""", [x, y, th])
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        if k == 3 and len(sys.argv) > 1:
            cv2.imwrite(sys.argv[1], frame)
        pts = detect_lane_points(frame, cam)
        est = fit_lane(pts)
        if est.valid:
            errs.append(abs(est.offset - off)); herrs.append(abs(est.heading))
            print(f"{x:4.1f} {off * 100:9.1f} cm {est.offset * 100:7.1f} cm {0.0:9.1f} {math.degrees(est.heading):8.1f} {est.lines:6d}")
        else:
            print(f"{x:4.1f} {off * 100:9.1f} cm   not detected ({len(pts)} points)")
    if errs:
        print(f"mean offset error {np.mean(errs) * 100:.1f} cm, mean heading error {math.degrees(np.mean(herrs)):.1f} deg")
    b.close()
