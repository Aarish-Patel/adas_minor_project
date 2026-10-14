"""Driver packets on the relay's UDP port: latest command wins.

A remote-driving link must act on the newest command only - a stale steering/throttle command is worse than none (the
usual practice for teleoperation command streams, e.g. ROS 2 "keep last, depth 1" QoS for velocity commands). The relay
handles one packet per control tick; when ticks ran long, packets queued in the socket and were acted on one by one,
seconds late (incident 1 Oct 2026: 1943 packets dropped by the full socket, commands still arriving after the driver
let go). After each wait the relay now takes everything already queued, keeps every control command (PING, GOTO, ZONES,
...) in order, and of the driving packets ("A ..", "M ..", "T .." lines) only the newest.
"""
import socket

DRIVING_PREFIXES = ("A ", "M ", "T ")
RCVBUF_BYTES = 32768        # the socket may hold ~0.5 s of driving packets at most; the drain below keeps it near empty


def is_driving(data):
    """True for a packet made only of steering / throttle / timestamp lines."""
    lines = [ln.strip() for ln in data.decode(errors="ignore").splitlines() if ln.strip()]
    return bool(lines) and all(ln[:2] in DRIVING_PREFIXES for ln in lines)


def latest_driving(batch):
    """batch: [(data, addr), ...] in arrival order -> (packets to handle in order, number of stale driving packets dropped).
    Control commands are all kept; of the driving packets only the last one."""
    last = max((i for i, (d, _) in enumerate(batch) if is_driving(d)), default=None)
    keep = [p for i, p in enumerate(batch) if i == last or not is_driving(p[0])]
    return keep, len(batch) - len(keep)


def drain(sock, limit=512):
    """Everything already queued on the socket, without waiting."""
    out = []
    old = sock.gettimeout()
    sock.settimeout(0.0)
    try:
        while len(out) < limit:
            try:
                out.append(sock.recvfrom(4096))
            except (BlockingIOError, socket.timeout):
                break
            except ConnectionResetError:       # Windows only (laptop simulator): a reply went to a closed sender
                continue
    finally:
        sock.settimeout(old)
    return out


def small_buffer(sock, size=RCVBUF_BYTES):
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, size)
    except OSError:
        pass
