"""
Xbox controller -> ESP32 RC car.  ALL tuning lives in this file.

The ESP32 firmware (ESP32_RC/ESP32_RC.ino) only executes "A <s1> <s2>"
(servo angles) and "M <pwm>" (motor) commands. Ackermann geometry, servo
calibration, limits, pin swap, steering feel and controls are all set below.

Controls:
  Left stick X   steering (stick right = turn right)
  Right trigger  forward
  Left trigger   reverse

Usage:
  python rc_controller.py                drive
  python rc_controller.py --ping         test the link to the ESP32
  python rc_controller.py --debug        print live controller axes
  python rc_controller.py --table        print steering -> servo angles (no hardware)
  python rc_controller.py --center       move both servos to their centre and exit
  python rc_controller.py --angles 95 90 set the LEFT and RIGHT servo to raw angles and exit

Requirements:  pip install pygame pyserial
"""

import math
import socket
import sys
import time

import pygame

# ============================================================
# LINK TO THE ESP32
# ============================================================

TRANSPORT = "wifi"          # "wifi" or "serial"

SERIAL_PORT = "COM7"
BAUD_RATE = 115200

ESP32_IP = "192.168.1.6"    # the PI's address (pi/wifi_drive_safety.py), NOT the ESP32's directly:
                            # driving straight to the ESP32 bypasses the obstacle safety relay.
                            # (leaving this on auto-discover risks nondeterministically finding
                            # the ESP32 itself instead, since both would answer a broadcast PING)
ESP32_HOSTNAME = "rccar.local"
ESP32_PORT = 4210

# ============================================================
# CAR GEOMETRY (any unit, as long as both use the same one)
# ============================================================

WHEELBASE = 20.0            # front axle centre to rear axle centre
TRACK = 7.0                 # distance between the two steering PIVOTS
                            # (not the outside tyre-to-tyre width)

# 1.0 = ideal Ackermann. 0.0 = parallel wheels. Values above 1 exaggerate it,
# negative values give anti-Ackermann.
ACKERMANN_FACTOR = 1

# ============================================================
# SERVO CALIBRATION (degrees, as used by the Arduino Servo library)
# ============================================================

LEFT_RIGHT, LEFT_CENTER, LEFT_LEFT = 30, 87, 140    # center 90 -> 87: measured with the final grips (pi/center_fine.py: 86.9, t=17.7)
RIGHT_RIGHT, RIGHT_CENTER, RIGHT_LEFT = 30, 87, 140

# Servo degrees per wheel degree (1.0 = servo shaft drives the wheel 1:1).
LEFT_SCALE = 1.0
RIGHT_SCALE = 1.0

# Which ESP32 output each servo is wired to: 1 = GPIO18, 2 = GPIO19.
# Currently swapped (left wheel is on GPIO19). Change if you rewire.
LEFT_SERVO_CHANNEL = 2
RIGHT_SERVO_CHANNEL = 1

# ============================================================
# STEERING FEEL
# ============================================================

STEERING_SIGN = 1          # stick right reads +1; the maths wants left = +. Flip if reversed.
STEERING_DEADZONE = 0.08
STEERING_EXPO = 1.0         # 1.0 = linear (half stick = half angle); >1 = gentler near centre
STEERING_SLEW_PER_SEC = 250.0   # max change of steering (-100..100) per second

# ============================================================
# MOTOR
# ============================================================

MAX_MOTOR_SPEED = 255       # cap top speed (0-255)
MOTOR_REVERSED = True     # flip if forward runs backward
TRIGGER_DEADZONE = 0.05
MOTOR_SLEW_PER_SEC = 600.0

# ============================================================
# CONTROLLER AXES (standard Xbox on Windows; check with --debug)
# ============================================================

AXIS_LEFT_X = 0
AXIS_LEFT_TRIGGER = 4
AXIS_RIGHT_TRIGGER = 5

SEND_HZ = 50.0

# ============================================================


def clamp(value, low, high):
    return max(low, min(high, value))


def steering_to_servo_angles(steer):
    """steer in -1..+1 (+ = left). Returns (left_servo_deg, right_servo_deg)."""
    steer = clamp(steer, -1.0, 1.0)
    turning_left = steer > 0

    # Inner wheel angle is proportional to steering and reaches the inner
    # servo's full travel at full steer.
    max_inner = (LEFT_LEFT - LEFT_CENTER) if turning_left else (RIGHT_CENTER - RIGHT_RIGHT)
    inner = (abs(steer) ** STEERING_EXPO) * max_inner

    if inner < 0.01:
        outer = 0.0
    else:
        # Ackermann: cot(outer) - cot(inner) = TRACK / WHEELBASE
        ideal = math.degrees(math.atan(1.0 / (1.0 / math.tan(math.radians(inner)) + TRACK / WHEELBASE)))
        outer = inner - ACKERMANN_FACTOR * (inner - ideal)

    if turning_left:
        left_wheel, right_wheel, sign = inner, outer, 1
    else:
        left_wheel, right_wheel, sign = outer, inner, -1

    left = LEFT_CENTER + sign * left_wheel * LEFT_SCALE
    right = RIGHT_CENTER + sign * right_wheel * RIGHT_SCALE

    return (clamp(left, LEFT_RIGHT, LEFT_LEFT),
            clamp(right, RIGHT_RIGHT, RIGHT_LEFT))


def build_command(left_deg, right_deg, motor):
    by_channel = {LEFT_SERVO_CHANNEL: left_deg, RIGHT_SERVO_CHANNEL: right_deg}
    motor = -motor if MOTOR_REVERSED else motor
    return f"A {by_channel[1]:.1f} {by_channel[2]:.1f}\nM {int(motor)}\n"


def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def trigger_to_unit(raw):
    return clamp((raw + 1.0) / 2.0, 0.0, 1.0)


def slew_toward(current, target, max_delta):
    if target > current:
        return min(current + max_delta, target)
    if target < current:
        return max(current - max_delta, target)
    return current


# ============================================================
# LINKS
# ============================================================

class SerialLink:
    def __init__(self):
        import serial
        self.ser = serial.Serial()
        self.ser.port = SERIAL_PORT
        self.ser.baudrate = BAUD_RATE
        self.ser.timeout = 0.1
        self.ser.dtr = False    # keep the ESP32 from resetting when the port opens
        self.ser.rts = False
        try:
            self.ser.open()
        except serial.SerialException as e:
            print(f"Could not open {SERIAL_PORT}: {e}")
            print("Close the Arduino Serial Monitor / other scripts using this port.")
            sys.exit(1)
        time.sleep(0.5)
        print(f"Serial link on {SERIAL_PORT}")

    def send(self, text):
        self.ser.write(text.encode())

    def ping(self):
        self.ser.reset_input_buffer()
        start = time.time()
        self.ser.write(b"PING\n")
        while time.time() - start < 1.0:
            if b"PONG" in self.ser.readline():
                return (time.time() - start) * 1000
        return None

    def close(self):
        self.ser.close()


class UdpLink:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ESP32_IP or self.discover()
        self.sock.settimeout(0.0)
        print(f"WiFi link to {self.ip}:{ESP32_PORT}")

    def discover(self):
        print("Looking for the ESP32 on the network...")
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.settimeout(0.5)
        for _ in range(6):
            try:
                self.sock.sendto(b"PING", ("255.255.255.255", ESP32_PORT))
                _, addr = self.sock.recvfrom(64)
                return addr[0]
            except socket.timeout:
                pass
            except OSError:
                break
        try:
            return socket.gethostbyname(ESP32_HOSTNAME)
        except OSError:
            pass
        print("Could not find the ESP32. Check it is powered and this laptop is on the")
        print("same WiFi (or the ESP32's own RC_CAR network), or set ESP32_IP above.")
        sys.exit(1)

    def send(self, text):
        try:
            self.sock.sendto(text.encode(), (self.ip, ESP32_PORT))
        except OSError as e:
            print(f"\nSend failed (check WiFi): {e}")

    def ping(self):
        self.sock.settimeout(1.0)
        start = time.time()
        self.sock.sendto(b"PING", (self.ip, ESP32_PORT))
        try:
            self.sock.recvfrom(64)
            result = (time.time() - start) * 1000
        except socket.timeout:
            result = None
        self.sock.settimeout(0.0)
        return result

    def close(self):
        self.sock.close()


def open_link():
    return SerialLink() if TRANSPORT == "serial" else UdpLink()


# ============================================================
# MODES
# ============================================================

def find_controller():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller detected. Connect the Xbox controller and try again.")
        sys.exit(1)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller: {joystick.get_name()}")
    return joystick


def print_table():
    print("steer   left servo  right servo   (left servo drives the LEFT wheel)")
    for pct in (100, 75, 50, 25, 0, -25, -50, -75, -100):
        left, right = steering_to_servo_angles(pct / 100.0)
        print(f"{pct:>5}   {left:9.1f}   {right:10.1f}")


def debug_axes():
    joystick = find_controller()
    print("Move sticks/triggers to see live axis values. Ctrl+C to stop.")
    try:
        while True:
            pygame.event.pump()
            values = [round(joystick.get_axis(i), 2) for i in range(joystick.get_numaxes())]
            print(values, end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def ping_test():
    link = open_link()
    for i in range(10):
        ms = link.ping()
        print(f"  reply {i + 1}: " + (f"{ms:.1f} ms" if ms is not None else "no reply"))
        time.sleep(0.2)
    link.close()


def send_angles_once(left, right):
    link = open_link()
    for _ in range(5):
        link.send(build_command(left, right, 0))
        time.sleep(0.05)
    print(f"Sent left servo {left}, right servo {right}")
    link.close()


def drive():
    joystick = find_controller()
    link = open_link()

    send_interval = 1.0 / SEND_HZ
    steering = 0.0      # -100..100, + = left
    motor = 0.0         # -255..255
    last_send = 0.0
    last_tick = 0.0

    print("Driving. Ctrl+C to stop.")

    try:
        while True:
            pygame.event.pump()

            steer_in = apply_deadzone(joystick.get_axis(AXIS_LEFT_X), STEERING_DEADZONE)
            target_steering = STEERING_SIGN * steer_in * 100.0

            rt = apply_deadzone(trigger_to_unit(joystick.get_axis(AXIS_RIGHT_TRIGGER)), TRIGGER_DEADZONE)
            lt = apply_deadzone(trigger_to_unit(joystick.get_axis(AXIS_LEFT_TRIGGER)), TRIGGER_DEADZONE)
            target_motor = clamp((rt - lt) * MAX_MOTOR_SPEED, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED)

            now = time.time()
            dt = now - last_tick if last_tick else send_interval
            last_tick = now

            steering = slew_toward(steering, target_steering, STEERING_SLEW_PER_SEC * dt)
            motor = slew_toward(motor, target_motor, MOTOR_SLEW_PER_SEC * dt)

            if now - last_send >= send_interval:
                last_send = now
                left, right = steering_to_servo_angles(steering / 100.0)
                link.send(build_command(left, right, motor))
                print(f"steer {steering:+6.1f}  L {left:6.1f}  R {right:6.1f}  motor {int(motor):+4d}   ", end="\r")

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass

    finally:
        print("\nStopping: motor off, steering centred.")
        left, right = steering_to_servo_angles(0.0)
        link.send(build_command(left, right, 0))
        time.sleep(0.1)
        link.close()


def main():
    global ESP32_IP, TRANSPORT, SERIAL_PORT
    args = sys.argv[1:]
    if "--ip" in args:                      # e.g. --ip 127.0.0.1 to drive the laptop simulator (tools/sim_car.py)
        ESP32_IP = args[args.index("--ip") + 1]
    # controller only, NO safety relay (bench tests): straight to the ESP32 over USB or WiFi
    if "--serial" in args:                  # --serial COM7
        TRANSPORT, SERIAL_PORT = "serial", args[args.index("--serial") + 1]
        print("CONTROLLER ONLY over USB - no obstacle braking, no ADAS")
    if "--direct" in args:                  # find the ESP32 itself on WiFi (broadcast PING / rccar.local)
        ESP32_IP = None
        print("CONTROLLER ONLY over WiFi - no obstacle braking, no ADAS")

    if "--table" in args:
        print_table()
    elif "--debug" in args:
        debug_axes()
    elif "--ping" in args:
        ping_test()
    elif "--center" in args:
        send_angles_once(LEFT_CENTER, RIGHT_CENTER)
    elif "--angles" in args:
        i = args.index("--angles")
        send_angles_once(float(args[i + 1]), float(args[i + 2]))
    else:
        drive()


if __name__ == "__main__":
    main()
