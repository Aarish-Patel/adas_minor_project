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
    R.EXTRA_POST["/api/sim/reset"] = lambda: car.reset(start)

    def publish_truth():
        while True:
            time.sleep(0.15)
            with R.GUI_STATE["lock"]:
                R.GUI_STATE["data"]["sim"] = dict(truth_overlay(car), world=a.world)
    threading.Thread(target=publish_truth, daemon=True).start()

    print(f"simulated car in world '{a.world}'  |  GUI http://localhost:8090/  |  "
          f"drive: python rc_controller.py --ip 127.0.0.1")
    R.main()


if __name__ == "__main__":
    main()
