"""Drive the car's own relay (pi/wifi_drive_safety.py, unchanged) on the laptop, against the virtual car.

    python tools/sim_car.py [--world doorway|room|corridor|gap|open|log:<file>] [--usb-stalls]
    python rc_controller.py --ip 127.0.0.1        (in a second terminal: drive with your controller)
    open http://localhost:8090/                   (the same GUI as on the car, plus the true walls in grey)

Logs go to RC_Car/logs/sim/. The GUI's RESET button puts the car back at the start.
"""
import argparse
import os
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="doorway")
    ap.add_argument("--usb-stalls", action="store_true", help="reproduce the LiDAR USB stalls seen on the car")
    a = ap.parse_args()
    os.environ.setdefault("RC_LOG_DIR", os.path.join(ROOT, "logs", "sim"))

    import pi.wifi_drive_safety as R
    from pi.relay_assists import car_params
    from sim.hw_sim import SimESP32, SimLidar, VirtualCar, truth_overlay
    from sim.hw_worlds import build

    world, start = build(a.world)
    car = VirtualCar(world, car_params(R.TUNING.mount), start, R.TUNING.servo.motor_reversed)
    R.find_ports = lambda *args, **kw: ("SIM_LIDAR", "SIM_ESP32")
    R.serial.Serial = lambda *args, **kw: SimESP32(car)
    R.open_lidar = lambda port, *args, **kw: SimLidar(car, R.TUNING.mount.yaw_offset_deg, usb_stalls=a.usb_stalls)
    cur = {"name": a.world, "start": start}

    def switch(name, marker_id=23):
        """Swap the room under the running relay (the HMI's Parking Simulator switch): the car goes to the new start pose;
        the HMI then sends ORIGIN so the relay's world frame is the new start."""
        new_world, new_start = build(name, marker_id)
        cur.update(name=name, start=new_start)
        car.world = new_world
        teleport(new_start)

    def teleport(pose):
        car.reset(pose)
        R.TELEPORT_PENDING.set()                                   # the relay forgets its obstacle memory / manoeuvre (not on the car),
                                                                   # in its control loop's thread at the next tick

    R.EXTRA_POST["/api/sim/reset"] = lambda: teleport(cur["start"])
    for nm in ("park_perp", "park_street", "room", "doorway", "open", "corridor", "gap", "lounge"):
        R.EXTRA_POST[f"/api/sim/world/{nm}"] = lambda nm=nm: switch(nm)
    for i in range(50):                                            # the ArUco marker id of the perpendicular room
        R.EXTRA_POST[f"/api/sim/world/park_perp/{i}"] = lambda i=i: switch("park_perp", i)

    def publish_truth():
        from sim.parking_worlds import layout_for_hmi
        while True:
            time.sleep(0.05)                                       # 20 Hz: the HMI pairs sim.pose with the relay's pose
            data = dict(truth_overlay(car), world=cur["name"])
            lay = layout_for_hmi(car.world, (0.0, 0.0, 0.0))
            if lay:
                data["parking"] = lay                              # docs/interfaces.md section 15 (simulator-only key)
            with R.GUI_STATE["lock"]:
                R.GUI_STATE["data"]["sim"] = data
    threading.Thread(target=publish_truth, daemon=True).start()

    # the rear camera: the twin's marker sensor (sim/marker_sensor.py) sees the wall marker of the perpendicular room; the relay
    # plans from the observations of the last 1.5 s (the car is standing when PARK_MARKER is given)
    from types import SimpleNamespace
    from adas.park_select import rear_camera
    from sim.marker_sensor import MarkerSensor
    sensor, recent = MarkerSensor(rear_camera(), seed=1), []

    def watch_markers():
        while True:
            time.sleep(0.07)
            x, y, th = car.pose()[:3]
            obs = sensor.sense(car.world, SimpleNamespace(x=x, y=y, theta=th), time.time())
            if obs:
                recent.extend((time.time(), o) for o in obs)
            recent[:] = [(t, o) for t, o in recent if time.time() - t < 1.5]
    threading.Thread(target=watch_markers, daemon=True).start()
    R.MARKER_SOURCE = lambda: [o for _, o in recent]

    print(f"simulated car in world '{a.world}'  |  GUI http://localhost:8090/  |  "
          f"drive: python rc_controller.py --ip 127.0.0.1")
    R.main()


if __name__ == "__main__":
    main()
