"""Controller only: forward the laptop controller's UDP packets to the ESP32 over USB, unchanged.
NO obstacle braking, NO ADAS - for bench tests (wheels off the ground) and hardware checks only.

    python3 pi/passthrough.py [/dev/ttyUSB0]      (stop rc-relay first; Ctrl+C stops and zeroes the motor)

Then on the laptop: python rc_controller.py   (it sends to the Pi, 192.168.1.6:4210, as usual)
PING from the controller is answered by the ESP32 itself. The ESP32's own 500 ms failsafe stops the motor if
packets stop.
"""
import glob
import socket
import sys
import time

import serial


def esp_port():
    """The ESP32 is the ttyUSB that answers PING (the other one is the LiDAR)."""
    for p in sorted(glob.glob("/dev/ttyUSB*")):
        try:
            s = serial.Serial()
            s.port, s.baudrate, s.timeout, s.dtr, s.rts = p, 115200, 0.3, False, False
            s.open()
            time.sleep(0.2)
            s.reset_input_buffer()
            s.write(b"PING\n")
            if b"PONG" in s.read(64):
                return s
            s.close()
        except Exception:
            pass
    return None


def main():
    if len(sys.argv) > 1:
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout, ser.dtr, ser.rts = sys.argv[1], 115200, 0.3, False, False
        ser.open()
    else:
        ser = esp_port()
    if ser is None:
        sys.exit("ESP32 not found on /dev/ttyUSB* (is rc-relay still running? it holds the port)")
    print(f"ESP32 on {ser.port}. CONTROLLER ONLY - no obstacle braking. Listening on UDP :4210, Ctrl+C to stop.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 4210))
    sock.settimeout(0.5)
    n = 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                continue
            text = data.decode(errors="ignore")
            if text.strip() == "PING":
                ser.reset_input_buffer()
                ser.write(b"PING\n")
                sock.sendto(b"PONG" if b"PONG" in ser.read(16) else b"NO ESP32 REPLY", addr)
                continue
            lines = [ln for ln in text.splitlines() if ln[:2] in ("A ", "M ") or ln.strip() == "STOP"]
            if lines:
                ser.write(("\n".join(lines) + "\n").encode())
                n += 1
                if n % 20 == 0:
                    print(f"\r{n} packets  last: {' | '.join(lines)}      ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        ser.write(b"M 0\nSTOP\n")
        ser.close()
        print("\nmotor stopped")


if __name__ == "__main__":
    main()
