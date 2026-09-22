import sys, time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"

s = serial.Serial()
s.port = PORT
s.baudrate = 115200
s.timeout = 1
s.dtr = False   # avoid resetting the ESP32 when the port opens
s.rts = False
s.open()
time.sleep(2.0)   # let the ESP32 boot / settle
s.reset_input_buffer()

def send(line):
    s.write((line + "\n").encode())
    time.sleep(0.15)
    resp = s.read(s.in_waiting or 1)
    print(f"  sent {line!r:20s} -> {resp!r}")

print(f"testing {PORT} at 115200 baud")
send("PING")
send("A 90 90")
time.sleep(0.3)
send("A 60 120")
time.sleep(0.3)
send("A 90 90")
send("M 0")
send("STOP")

# drain any extra boot/status text
time.sleep(0.3)
extra = s.read(s.in_waiting or 1)
if extra:
    print("  extra:", extra)

s.close()
