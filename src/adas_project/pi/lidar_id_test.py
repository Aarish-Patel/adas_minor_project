import sys, time
import serial

for port in ("/dev/ttyUSB0", "/dev/ttyUSB1"):
    print(f"=== {port} ===")
    try:
        s = serial.Serial()
        s.port = port
        s.baudrate = 256000
        s.timeout = 2
        s.dtr = False
        s.rts = False
        s.open()
        time.sleep(1.0)
        s.reset_input_buffer()
        s.write(bytes([0xA5, 0x50]))
        time.sleep(0.4)
        n = s.in_waiting
        raw = s.read(n) if n else b""
        print(f"  GET_INFO -> {n} bytes: {raw.hex(' ')}")
        if raw[:2] == bytes([0xA5, 0x5A]):
            print("  *** THIS IS THE RPLIDAR ***")
        s.close()
    except Exception as e:
        print("  error:", e)
