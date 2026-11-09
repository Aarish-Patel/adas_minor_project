"""The relay's view of the marker camera service (pi/marker_camera.py): a background thread polls its /state and keeps the marker
observations of the last seconds as adas.markers.MarkerObs in the vehicle frame. The client object is the relay's MARKER_SOURCE; the relay
never waits for the camera (a dead service just means no observations: "marker not visible")."""
import json
import threading
import time
import urllib.request

import numpy as np

from adas.markers import MarkerObs

KEEP_S = 1.5            # observations older than this are dropped (the car is standing when the parking command is given)
STALE_S = 0.6           # a service state older than this (its own clock) is not a live camera


class MarkerClient:
    def __init__(self, url="http://127.0.0.1:8091/state", period_s=0.1, marker_id=None, start=True):
        self.url, self.period, self.marker_id = url, period_s, marker_id
        self._obs, self._lock = [], threading.Lock()
        self.last_ok, self.service_ok = 0.0, False
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=1.0) as r:
                    st = json.loads(r.read())
                self.ingest(st)
            except Exception:
                self.service_ok = False
            time.sleep(self.period)

    def ingest(self, st, now=None):
        """Take one /state document (also the unit tests' entry)."""
        now = time.time() if now is None else now
        self.service_ok = bool(st.get("ok")) and abs(now - float(st.get("t", 0.0))) < STALE_S
        if not self.service_ok:
            return
        self.last_ok = now
        new = []
        for m in st.get("markers", []):
            if self.marker_id is not None and m["id"] != self.marker_id:
                continue
            new.append((float(m["t"]), MarkerObs(id=int(m["id"]), corners=np.zeros((4, 2)), x=m["x"], y=m["y"], z=m["z"], facing=m["facing"],
                                                 dist=m["dist"], bearing=m["bearing"], t=float(m["t"]), size_px=m["size_px"])))
        with self._lock:
            self._obs = [(t, o) for t, o in self._obs + new if now - t < KEEP_S]

    def __call__(self, now=None):
        """The observations of the last KEEP_S seconds (MarkerObs, vehicle frame), oldest first."""
        now = time.time() if now is None else now
        with self._lock:
            return [o for t, o in self._obs if now - t < KEEP_S]
