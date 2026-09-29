"""Run the ADAS on the real car.

    python -m pi.main --link udp                       WiFi, finds the ESP32 automatically
    python -m pi.main --link serial:/dev/ttyACM0       USB serial
    python -m pi.main --lidar rplidar:/dev/ttyUSB0 --camera 0 --tuning tuning.json

Buttons (Xbox):  X = ADAS off/warn/active   A = auto-park   Y = follow the leader   B = e-stop

Try the whole chain on a laptop with no hardware first:  python -m pi.hil wall
"""

import argparse
import signal

from adas.config import load_tuning
from adas.intent import IntentLogger
from pi.link import Esp32Actuator, SerialLink, UdpLink
from pi.runtime import Runtime


def build(args):
    tuning = load_tuning(args.tuning)

    if args.link.startswith("serial:"):
        link = SerialLink(args.link.split(":", 1)[1])
    else:
        host = args.link.split(":", 1)[1] if ":" in args.link else None
        link = UdpLink(host)
    actuator = Esp32Actuator(link, tuning.servo, tuning.vehicle.wheelbase, tuning.vehicle.pivot_track)

    from pi import devices
    lidar = None
    if args.lidar != "none":
        lidar = devices.RPLidarSource(args.lidar.split(":", 1)[1] if ":" in args.lidar else "/dev/ttyUSB0",
                                      angle_offset_deg=args.lidar_offset)
    camera = None if args.camera == "none" else devices.WebcamMarkers(int(args.camera))
    driver = devices.XboxDriver()
    logger = IntentLogger(path=args.log) if args.log else None
    return Runtime(tuning, lidar, camera, driver, actuator, logger, mode=args.mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuning", default="tuning.json", help="tuning file exported from the simulator")
    ap.add_argument("--link", default="udp", help="udp | udp:IP | serial:PORT")
    ap.add_argument("--lidar", default="rplidar:/dev/ttyUSB0", help="rplidar:PORT | none")
    ap.add_argument("--lidar-offset", type=float, default=0.0, help="degrees: where the LiDAR's zero points relative to the car's front")
    ap.add_argument("--camera", default="0", help="camera index | none")
    ap.add_argument("--mode", default="active", choices=["off", "advisory", "active"])
    ap.add_argument("--log", default="", help="write driving data (for the intent model) to this CSV")
    args = ap.parse_args()

    rt = build(args)
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    print("running - Ctrl+C to stop (motor and steering return to neutral)")
    rt.run(hz=50.0, stop=lambda: stop["now"])
    if rt.logger:
        rt.logger.close()


if __name__ == "__main__":
    main()
