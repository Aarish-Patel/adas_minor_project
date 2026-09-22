import socket, sys, time

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.2"
PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(1.5)

def send(line):
    sock.sendto(line.encode(), (IP, PORT))
    if line == "PING":
        try:
            data, addr = sock.recvfrom(64)
            print(f"  sent {line!r:10s} -> reply from {addr}: {data!r}")
        except socket.timeout:
            print(f"  sent {line!r:10s} -> NO REPLY (timeout)")
    else:
        print(f"  sent {line!r:10s} -> (no reply expected)")

print(f"testing UDP to {IP}:{PORT}")
send("PING")
send("A 90 90")
time.sleep(0.3)
send("A 60 120")
time.sleep(0.3)
send("A 90 90")
send("M 0")
send("STOP")
send("PING")
sock.close()
