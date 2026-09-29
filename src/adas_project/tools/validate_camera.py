"""Does the 3D-rendered camera agree with the geometric marker sensor and with OpenCV?

Captures the browser's virtual camera frame at several poses, runs the real ArUco detector on it,
and compares the recovered marker position/size with the analytic prediction.
"""
import base64, math, sys, time
import cv2, numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, ".")
from adas.markers import Camera, detect_image
from sim.car_sim import Dynamics, SimCar
from sim.library import parking
from sim.marker_sensor import MarkerSensor
from adas.vehicle_params import VehicleParams

POSES = [(2.2, 0.0, 0), (2.8, 0.15, 6), (3.2, -0.10, -8), (3.6, 0.05, 0), (3.9, 0.0, 12), (1.9, 0.25, -5)]
p = VehicleParams()
world, _ = parking()
sensor = MarkerSensor(pixel_noise=0.0, detect_prob=1.0, seed=0)
marker = next(o for o in world.objects if hasattr(o, "marker_id"))
cam = Camera()

with sync_playwright() as pw:
    b = pw.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1000, "height": 600})
    pg.goto("http://localhost:8765/", wait_until="load"); time.sleep(2)
    pg.select_option("#scenario", "parking"); time.sleep(2.5)
    pg.evaluate("window.__send({type:'cmd', cmd:'pause'})"); time.sleep(0.5)
    print(f"{'pose':22s} {'analytic dist':>13s} {'OpenCV dist':>12s} {'corner err px':>14s}  detected")
    for (x, y, th) in POSES:
        # set the 3D scene pose directly (no network) and grab a frame in the same JS task
        data = pg.evaluate("""([x, y, th]) => {
            const snap = { car: { x, y, th, v: 0, wl: 0, wr: 0 }, adas: { level: 0, D: -1, mode: 'off' }, cmd: { out: 0 },
                           dyn: [], tracks: [], path: [[x, y], [x + 0.05, y]], lidar: null };
            window.__world.update(snap, 0.03);
            const img = window.__world.captureCarCamera(); const c = document.createElement('canvas');
            c.width = 640; c.height = 480; c.getContext('2d').putImageData(img, 0, 0); return c.toDataURL('image/png'); }""",
                       [x, y, math.radians(th)])
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        cv = [o for o in detect_image(frame, cam, {7: marker.size}) if o.id == 7]
        car = SimCar(x, y, math.radians(th), p, Dynamics())
        sensor.next_time = 0
        ana = sensor.sense(world, car, 0.0)
        tag = f"({x}, {y}, {th} deg)"
        if cv and ana:
            err = float(np.linalg.norm(cv[0].corners - ana[0].corners, axis=1).mean())
            print(f"{tag:22s} {ana[0].dist:11.3f} m {cv[0].dist:10.3f} m {err:12.1f}    yes")
        else:
            print(f"{tag:22s} {'-' if not ana else f'{ana[0].dist:.3f} m':>13s} {'-':>12s} {'-':>14s}  OpenCV {'yes' if cv else 'NO'} / analytic {'yes' if ana else 'no'}")
        if x == 3.2:
            cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "camera_frame.png", frame)
    b.close()
