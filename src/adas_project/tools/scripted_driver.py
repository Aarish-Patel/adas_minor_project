"""A scripted driver for the laptop simulator (instead of rc_controller.py): sends the relay 20 Hz packets.

    python tools/scripted_driver.py [--pwm 200] [--seconds 20] [--assist evasive ...] [--steer-deg 0]

Used to demo and test the dashboard without a gamepad: e.g. full throttle at the box in front of the doorway with
the evasive steer on.
"""
import argparse
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pwm", type=float, default=200.0, help="throttle 0-255 (forward)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--centre", type=float, default=87.0, help="the servo's straight-ahead angle")
    ap.add_argument("--steer-deg", type=float, default=0.0, help="servo degrees from straight (+ = right)")
    ap.add_argument("--assist", nargs="*", default=[], help="assists to switch on first, e.g. evasive nudge")
    ap.add_argument("--start-delay", type=float, default=0.0)
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for name in a.assist:
        s.sendto(f"ASSIST {name} ON".encode(), (a.host, 4210))
    time.sleep(a.start_delay)
    servo = a.centre + a.steer_deg
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        s.sendto(f"A {servo:.0f} {servo:.0f}\nM {-int(a.pwm)}\n".encode(), (a.host, 4210))   # motor reversed
        time.sleep(0.05)
    for _ in range(5):
        s.sendto(f"A {a.centre:.0f} {a.centre:.0f}\nM 0\n".encode(), (a.host, 4210))
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
