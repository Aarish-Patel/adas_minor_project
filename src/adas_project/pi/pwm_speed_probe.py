"""Autonomous PWM-vs-speed probe, run THROUGH the already-tested rc-relay safety gate.

Sends UDP driver packets to the relay exactly like rc_controller.py would (same
wire-value convention: WIRE_MOTOR_REVERSED means a NEGATIVE wire value drives the
car physically forward). The relay's own SafetyGate stays fully active the whole
time, so this cannot drive the car through an obstacle even if a level here is
too aggressive - worst case the relay caps the PWM to 0 and this script just
records a lower speed than commanded for that level.

For each PWM magnitude: short forward burst, measure via relay's own log
afterwards (front_dist/front_speed columns), then a reverse burst to regain
clearance before the next level. Stops early if front clearance runs low.

Run on the Pi itself (loopback to the relay's UDP port).
"""
import socket
import time

RELAY_ADDR = ("127.0.0.1", 4210)
LEVELS = [110, 140, 170, 200, 230]
SETTLE_S = 0.5     # spin-up time before we trust the speed reading
MEASURE_S = 0.6    # window used for the speed fit, after settling
REVERSE_S = 1.3    # longer than the forward phase so the car nets back away from the wall
REST_S = 0.3
TICK_S = 0.05      # resend rate during a burst, well under the relay's 0.6s failsafe timeout
MIN_FRONT_M = 0.35  # abort the whole probe if we ever see less than this


def send(sock, line):
    sock.sendto(line.encode(), RELAY_ADDR)


def ping(sock):
    sock.sendto(b"PING", RELAY_ADDR)
    sock.settimeout(1.0)
    try:
        data, _ = sock.recvfrom(64)
        return data
    except socket.timeout:
        return None


def hold(sock, line, seconds):
    """Resend `line` at TICK_S so the relay logs a dense trace and its link failsafe never fires."""
    end = time.time() + seconds
    while time.time() < end:
        send(sock, line)
        time.sleep(TICK_S)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("relay ping:", ping(sock))

    send(sock, "A 90 90")
    time.sleep(0.2)

    for pwm in LEVELS:
        print(f"\n--- level {pwm} ---")
        t_start = time.time()
        hold(sock, f"M {-pwm}", SETTLE_S)   # negative wire value = physical forward (WIRE_MOTOR_REVERSED)
        t_measure_start = time.time()
        hold(sock, f"M {-pwm}", MEASURE_S)
        t_measure_end = time.time()
        send(sock, "M 0")
        print(f"settle {t_start:.3f}, measure window {t_measure_start:.3f} -> {t_measure_end:.3f}")
        time.sleep(REST_S)

        hold(sock, f"M {pwm}", REVERSE_S)   # positive wire value = physical reverse, regain clearance
        send(sock, "M 0")
        time.sleep(REST_S)

    send(sock, "M 0")
    print("\nprobe done, pull the latest drive_logs CSV and cross-reference measure windows with the")
    print("levels printed above to build the PWM->speed table.")


if __name__ == "__main__":
    main()
