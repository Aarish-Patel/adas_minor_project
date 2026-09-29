"""Sending commands to the ESP32 (WiFi UDP or USB serial) and an ESP32 emulator for testing.

Protocol (see ESP32_RC.ino): newline-separated text, e.g. "A 112.0 115.0\\nM 120\\n".
"""

import socket
import threading
import time

from adas.servo import ServoCalibration, command_text


class Esp32Actuator:
    """Turns (steer, pwm) into the ESP32's text protocol and sends it."""

    def __init__(self, link, calibration=None, wheelbase=0.20, track=0.07):
        self.link = link
        self.cal = calibration or ServoCalibration()
        self.wheelbase, self.track = wheelbase, track

    def send(self, steer, pwm):
        self.link.send(command_text(steer, pwm, self.cal, self.wheelbase, self.track))

    def stop(self):
        self.send(0.0, 0.0)


class UdpLink:
    def __init__(self, host=None, port=4210, hostname="rccar.local"):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host or self.discover(hostname)

    def discover(self, hostname):
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.settimeout(0.5)
        for _ in range(6):
            try:
                self.sock.sendto(b"PING", ("255.255.255.255", self.port))
                _, addr = self.sock.recvfrom(64)
                self.sock.settimeout(0.0)
                return addr[0]
            except (socket.timeout, OSError):
                pass
        self.sock.settimeout(0.0)
        return socket.gethostbyname(hostname)

    def send(self, text):
        try:
            self.sock.sendto(text.encode(), (self.host, self.port))
        except OSError:
            pass

    def close(self):
        self.sock.close()


class SerialLink:
    """USB serial to the ESP32. DTR/RTS are held low so opening the port does not reset it."""

    def __init__(self, port, baud=115200):
        import serial
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.05
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        time.sleep(0.5)

    def send(self, text):
        self.ser.write(text.encode())

    def close(self):
        self.ser.close()


class Esp32Emulator(threading.Thread):
    """Behaves like the ESP32 firmware over UDP: parses A/M/STOP/PING and applies the 500 ms failsafe.

    Lets the whole Pi-side chain be tested without hardware. Read .servo1, .servo2, .motor.
    """

    def __init__(self, port=0, timeout_s=0.5):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.servo1 = self.servo2 = None
        self.motor = 0
        self.timeout_s = timeout_s
        self.last_cmd = time.time()
        self.commands = 0
        self.running = True

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                if self.motor != 0 and time.time() - self.last_cmd > self.timeout_s:
                    self.motor = 0
                continue
            except OSError:
                break                                   # socket closed by stop()
            for line in data.decode(errors="ignore").splitlines():
                parts = line.strip().upper().split()
                if not parts:
                    continue
                if parts[0] == "A" and len(parts) == 3:
                    self.servo1 = max(0.0, min(180.0, float(parts[1])))
                    self.servo2 = max(0.0, min(180.0, float(parts[2])))
                    self.last_cmd = time.time()
                    self.commands += 1
                elif parts[0] == "M" and len(parts) == 2:
                    self.motor = max(-255, min(255, int(float(parts[1]))))
                    self.last_cmd = time.time()
                elif parts[0] == "STOP":
                    self.motor = 0
                    self.last_cmd = time.time()
                elif parts[0] == "PING":
                    self.sock.sendto(b"PONG", addr)

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        self.sock.close()
