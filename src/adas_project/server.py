"""3D simulator server.  Run it and open http://localhost:8765

    python server.py [--port 8765]

Serves the browser viewer (web/) and streams the running simulation to it over a
WebSocket. The browser sends driver input (keyboard / Xbox controller) and
commands (scenario, ADAS mode, tuning) back.
"""

import argparse
import asyncio
import functools
import http.server
import json
import os
import threading
import time

import websockets
from websockets.asyncio.server import serve

from sim.session import Session
from sim import mc_viz

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
                      ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


class Runner(threading.Thread):
    """Advances the simulation in real time on its own thread."""

    def __init__(self, session, lock):
        super().__init__(daemon=True)
        self.session, self.lock = session, lock
        self.running = True

    def run(self):
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            with self.lock:
                self.session.tick(now - last)
            last = now
            time.sleep(0.012)


async def run_mc(ws, n):
    await ws.send(json.dumps({"type": "mc_status", "text": f"simulating {n} random drives (ADAS off and on) ..."}))
    cases = await asyncio.to_thread(mc_viz.run_batch, n)
    await ws.send(json.dumps({"type": "mc", "cases": cases}))


async def client(ws, session, lock):
    version = -1
    last_scan = -1
    send_full = True

    async def receive():
        nonlocal send_full
        async for raw in ws:
            if isinstance(raw, bytes):
                with lock:
                    session.cv_frame(raw)
                continue
            msg = json.loads(raw)
            with lock:
                kind = msg.get("type")
                if kind == "input":
                    session.set_input(msg.get("steer", 0.0), msg.get("pwm", 0.0))
                elif kind == "cmd":
                    cmd = msg.get("cmd")
                    if cmd == "scenario":
                        session.load(msg["id"], msg.get("seed"))
                        send_full = True
                    elif cmd == "reset":
                        session.reset()
                        send_full = True
                    elif cmd == "mode":
                        session.set_mode(msg["mode"])
                    elif cmd == "speed":
                        session.speed = max(0.1, min(8.0, float(msg["value"])))
                    elif cmd == "pause":
                        session.paused = not session.paused
                    elif cmd == "record":
                        session.toggle_record()
                    elif cmd == "teleport":
                        session.teleport(msg["x"], msg["y"], msg.get("theta", 0.0))
                    elif cmd == "park":
                        session.toggle_park()
                    elif cmd == "lane":
                        session.cycle_lane()
                    elif cmd == "acc":
                        session.toggle_acc()
                    elif cmd == "isa":
                        session.toggle_isa()
                    elif cmd == "param":
                        session.set_param(msg["id"], msg["value"])
                    elif cmd == "export":
                        await ws.send(json.dumps({"type": "export", "data": session.export_tuning()}))
                    elif cmd == "mc":
                        n = int(msg.get("n", 16))
                        asyncio.create_task(run_mc(ws, n))

    recv_task = asyncio.create_task(receive())
    try:
        while True:
            with lock:
                if send_full:
                    await ws.send(json.dumps(session.describe()))
                    await ws.send(json.dumps({"type": "tunables", "spec": session.tunable_spec()}))
                    send_full = False
                    last_scan = -1
                snap = session.snapshot(last_scan)
                last_scan = snap["scan_id"]
            await ws.send(json.dumps(snap))
            await asyncio.sleep(1 / 30)
    except websockets.ConnectionClosed:
        pass
    finally:
        recv_task.cancel()


async def main(port):
    session = Session()
    lock = threading.Lock()
    Runner(session, lock).start()

    handler = functools.partial(NoCacheHandler, directory=WEB)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    async with serve(lambda ws: client(ws, session, lock), "0.0.0.0", port + 1, max_size=2 ** 24):
        print(f"3D simulator running:  http://localhost:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    try:
        asyncio.run(main(args.port))
    except KeyboardInterrupt:
        pass
