import glob
import time
import serial

for port in sorted(glob.glob("/dev/ttyUSB*")):
    try:
        s = serial.Serial(port, 115200, timeout=1)
        s.dtr = False
        s.rts = False
        time.sleep(0.5)
        s.write(b"M 0\n")
        s.write(b"STOP\n")
        time.sleep(0.2)
        reply = s.read(s.in_waiting or 1)
        print(f"{port}: sent stop, reply={reply!r}")
        s.close()
    except Exception as e:
        print(f"{port}: error {e}")
