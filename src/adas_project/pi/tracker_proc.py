"""The moving-object tracker (adas/tracking.py) in its own process (TODO AP1).

The relay is one Python process: its LiDAR thread, its control loop and its logging threads share ONE interpreter lock, so on the Pi they together
get at most one core however many it has. The tracker was ~half of the LiDAR thread's time (profile of 9 Nov, pi/sampler.py) and starved the control
loop (12 of 50 driver packets a second handled). Here the LiDAR thread only posts each scan (vehicle-frame points + the car's own motion) to a worker
process on another core and the results come back asynchronously.

The tracks are advisory (moving-object warnings, the crossing planner, the follow mode; the brake gate's static logic does not depend on them): a result
older than STALE_S is not served (no tracks, no moving contact) instead of being trusted late, and when the worker dies the caller falls back to
tracking in its own thread (`alive()` / `submit()` returning False).
"""
import multiprocessing as mp
import queue
import threading
import time

STALE_S = 0.8


def _worker(inq, outq, vp):
    from adas.tracking import Tracker, moving_object_contact
    tracker = Tracker()
    while True:
        item = inq.get()
        if item is None:
            return
        xy, now, v, omega, delta, direction = item
        tracks = tracker.update(xy, now, v, omega)
        contact = moving_object_contact(tracks, delta, direction, abs(v), vp, horizon=1.5)
        info = [{"id": tr.id, "x": round(tr.pos[0], 3), "y": round(tr.pos[1], 3), "moving": tr.moving} for tr in tracks]
        outq.put((now, info, contact, tracks))


class TrackerProcess:
    """submit() scans, results arrive on `on_result(t_scan, info, contact, tracks)` (called from a reader thread)."""

    def __init__(self, vehicle_params, on_result):
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"          # (Windows: the laptop twin)
        ctx = mp.get_context(method)
        self.inq, self.outq = ctx.Queue(maxsize=2), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(self.inq, self.outq, vehicle_params), daemon=True, name="tracker")
        self.proc.start()
        self.on_result, self.dropped, self.running = on_result, 0, True
        threading.Thread(target=self._reader, daemon=True, name="tracker-reader").start()

    def alive(self):
        return self.proc.is_alive()

    def submit(self, xy, now, v, omega, delta, direction):
        """Post one scan; False when the worker is dead or still busy with older scans (then this scan is dropped: a newer one is coming)."""
        if not self.proc.is_alive():
            return False
        try:
            self.inq.put_nowait((xy, now, v, omega, delta, direction))
            return True
        except queue.Full:
            self.dropped += 1
            return True

    def _reader(self):
        while self.running:
            try:
                res = self.outq.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return
            self.on_result(*res)

    def stop(self):
        self.running = False
        try:
            self.inq.put_nowait(None)
        except (queue.Full, ValueError, OSError):
            pass
        self.proc.join(timeout=1.0)
        if self.proc.is_alive():
            self.proc.terminate()
