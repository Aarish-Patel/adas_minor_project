"""End-to-end processing latency of the relay (pi/wifi_drive_safety.py, unchanged) on the machine it runs on.

    python3 tools/relay_latency.py [--seconds 14] [--world doorway]

The relay runs against the virtual car + simulated LiDAR (as tools/sim_car.py) - no motor, no real LiDAR. A scripted
driver sends 20 Hz packets carrying their send time ("T <t>" line, passed through by the relay); the virtual ESP32
records when each arrives: that is the relay's input-to-motor processing time. Evasive steer is on and the driver
drives full throttle at the box in front of the doorway, so real path searches run in the planner worker meanwhile.
Logs go to a scratch folder (RC_LOG_DIR), never to the real /home/pi/logs.
"""
import argparse
import os
import socket
import sys
import threading
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--world", default="doorway")
    ap.add_argument("--profile", action="store_true", help="profile the relay's own thread")
    a = ap.parse_args()
    os.environ["RC_LOG_DIR"] = os.path.join(ROOT, "logs_latency_test")
    os.environ["RC_TIMING"] = "1"                         # the relay's per-stage timing

    import pi.wifi_drive_safety as R
    from pi.relay_assists import car_params
    from sim.hw_sim import SimESP32, SimLidar, VirtualCar
    from sim.hw_worlds import build

    world, start = build(a.world)
    car = VirtualCar(world, car_params(R.TUNING.mount), start, R.TUNING.servo.motor_reversed)
    lat, arrivals, queued, processing = [], [], [], []
    prof_holder = []
    received = {}                                         # send stamp -> when the relay's recvfrom returned it

    class ProbeESP32(SimESP32):
        def write(self, data):
            now = time.time()
            rest = []
            for line in data.decode(errors="replace").splitlines():
                if line.startswith("T "):
                    stamp = line.split()[1]
                    lat.append((now - float(stamp)) * 1000)
                    arrivals.append(now)
                    if stamp in received:
                        queued.append((received[stamp] - float(stamp)) * 1000)
                        processing.append((now - received[stamp]) * 1000)
                else:
                    rest.append(line)
            if rest:
                super().write(("\n".join(rest) + "\n").encode())

    real_socket = R.socket.socket

    class ProbeSocket(real_socket):
        def recvfrom(self, *args, **kw):
            data, addr = super().recvfrom(*args, **kw)
            for line in data.decode(errors="ignore").splitlines():
                if line.startswith("T "):
                    received[line.split()[1]] = time.time()
            return data, addr
    R.socket.socket = ProbeSocket

    R.find_ports = lambda *args, **kw: ("SIM_LIDAR", "SIM_ESP32")
    R.serial.Serial = lambda *args, **kw: ProbeESP32(car)
    R.open_lidar = lambda port, *args, **kw: SimLidar(car, R.TUNING.mount.yaw_offset_deg)

    def driver():
        try:
            run_driver()
        except Exception as e:                            # never leave the relay running without a driver
            print("probe failed:", repr(e), f"({len(lat)} stamps received)")
        finally:
            sys.stdout.flush()
            try:                                          # the planner's worker process must not outlive us
                import gc
                from adas.plan_service import PlanService
                for o in gc.get_objects():
                    if isinstance(o, PlanService):
                        o.shutdown()
            except Exception:
                pass
            os._exit(0)

    def run_driver():
        time.sleep(5.0)                                   # the relay spins the LiDAR up and starts the planner
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b"ASSIST evasive ON", ("127.0.0.1", R.UDP_PORT))
        centre = R.TUNING.servo.left_center
        wire = -255 if R.WIRE_MOTOR_REVERSED else 255
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            s.sendto(f"A {centre:.0f} {centre:.0f}\nM {wire}\nT {time.time():.6f}\n".encode(), ("127.0.0.1", R.UDP_PORT))
            time.sleep(0.05)
        x = np.array(lat)
        gaps = np.diff(arrivals) * 1000 if len(arrivals) > 1 else np.zeros(1)
        print(f"\n{len(x)} packets: relay input-to-motor latency median {np.median(x):.1f} ms, 95th "
              f"{np.percentile(x, 95):.1f} ms, 99th {np.percentile(x, 99):.1f} ms, max {x.max():.1f} ms")
        print(f"motor command interval: median {np.median(gaps):.1f} ms, max {gaps.max():.1f} ms (sent every 50 ms)")
        if queued:
            q, pr = np.array(queued), np.array(processing)
            print(f"  of which waiting in the socket: median {np.median(q):.1f} ms, 95th {np.percentile(q, 95):.1f} ms; "
                  f"relay processing: median {np.median(pr):.1f} ms, 95th {np.percentile(pr, 95):.1f} ms, "
                  f"max {pr.max():.1f} ms")
        x_, y_, th_, v_, *_r, crashed = car.pose()
        print(f"virtual car: x {x_:.2f} m, y {y_:+.2f} m, crashes {car.crash_count} (doorway box at x 1.7, wall at 2.8)")
        for st in R.STAGE_TIMERS:
            st.report()
        if prof_holder:
            import io
            import pstats
            pr = prof_holder[0]
            pr.create_stats()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
            print(s.getvalue()[-6000:])

    threading.Thread(target=driver, daemon=True).start()
    if "--profile" in sys.argv:                           # where the relay's own thread spends its time
        import cProfile
        import pstats
        pr = cProfile.Profile()
        prof_holder.append(pr)
        pr.enable()
    R.main()


if __name__ == "__main__":
    main()
