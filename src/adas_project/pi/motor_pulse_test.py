"""Short pulses at increasing PWM magnitude, both directions, to find the real dead-band.
Each pulse: 1s at that PWM, then 1s stop, so it's easy to see which ones actually move the car.
"""
import glob
import time
import serial

ESP_PORT = None
for p in sorted(glob.glob("/dev/ttyUSB*")):
    try:
        s = serial.Serial(p, 256000, timeout=1)
        s.dtr = False
        s.rts = False
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(bytes([0xA5, 0x50]))
        time.sleep(0.3)
        resp = s.read(s.in_waiting or 1)
        s.close()
        if resp[:2] != bytes([0xA5, 0x5A]):
            ESP_PORT = p
    except Exception:
        pass

print("ESP32 on", ESP_PORT)
esp = serial.Serial()
esp.port = ESP_PORT
esp.baudrate = 115200
esp.timeout = 0.2
esp.dtr = False
esp.rts = False
esp.open()
time.sleep(2.0)
esp.reset_input_buffer()

def send(line):
    esp.write((line + "\n").encode())

send("A 90 90")
send("M 0")
time.sleep(0.5)

# test both signs at increasing magnitude; positive first (known: backward), then negative
levels = [90, 110, 130, 150]
try:
    for sign, label in ((1, "POSITIVE (known: backward)"), (-1, "NEGATIVE")):
        print(f"\n=== {label} ===")
        for lvl in levels:
            pwm = sign * lvl
            print(f"  pulsing M {pwm} for 1s ...")
            send(f"M {pwm}")
            time.sleep(1.0)
            send("M 0")
            time.sleep(1.0)
finally:
    send("M 0")
    send("STOP")
    time.sleep(0.2)
    esp.close()
    print("\ndone, motor stopped")
