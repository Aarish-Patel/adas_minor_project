import sys, time
sys.path.insert(0, "/home/pi/rc_car")
import serial
from pi.lidar_steering_diag import find_ports
_, esp_port = find_ports()
s = serial.Serial(); s.port = esp_port; s.baudrate = 115200; s.timeout = 0.05; s.dtr = False; s.rts = False
s.open(); time.sleep(1.2); s.reset_input_buffer()

def phase(name, steer, pwm, secs=2.0):
    s.write(f"A {steer} {steer}\n".encode()); buf = b""
    t0 = time.time()
    while time.time() - t0 < secs:
        s.write(f"M {-pwm}\n".encode() if pwm else b"M 0\n")
        buf += s.read(200); time.sleep(0.05)
    s.write(b"M 0\n"); time.sleep(0.3); buf += s.read(400)
    resets = buf.count(b"rst:") + buf.count(b"ROVER READY") + buf.count(b"Brownout")
    print(f"{name:34s} resets/boot banners: {resets}   raw={buf[:60]!r}", flush=True)

phase("idle, servo 90", 90, 0, 1.0)
phase("motor 140, servo 90 (straight)", 90, 140)
phase("servo 100 held, no motor", 100, 0)
phase("motor 140 + servo 100 held", 100, 140)
phase("motor 140 + servo 80 held", 80, 140)
phase("motor 100 + servo 94 held", 94, 100)
s.write(b"A 90 90\nM 0\n"); s.close()
