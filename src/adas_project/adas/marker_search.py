"""Park at a marker from a general area (TODO AQ3): drive to the area, find the ArUco marker on the bay wall, then hand over to the marker
parking (AutoNav nose-in plan + the LiDAR wall stop).

The operator names only the general area (a pose in front of the bay, facing the wall). The car cannot turn on the spot and its camera is
fixed-focus (a marker is unreadable beyond ~1.2 m), so the search is a visual approach in short steps, each one a click-to-go goal that the
planner and the brake gate drive like any other:

    GOTO_AREA   click-to-go to the area pose (heading = towards the bay).
    SEARCH      nothing seen: a short step ahead, the heading swinging left / right so the camera sweeps the wall (never closer to a wall than
                WALL_STOP_M: then the marker is not here).
    APPROACH    a marker-like thing or the id is seen but farther than PARK_DIST_M: a step towards it (bearing and range from the camera).
    PARK        the id is seen within PARK_DIST_M: the marker parking takes over (it measures the bay from the fused observations).

Visual search by short steps towards the strongest cue is the usual behaviour of marker-docking robots (Bayesian / look-and-move servoing,
Chaumette & Hutchinson, "Visual servo control, Part I", IEEE Robotics & Automation Magazine 2006: look, move a bit, look again). Pure
decision logic: no hardware, no clock but the `now` it is given.
"""
import math

GOTO_AREA, SEARCH, APPROACH, PARK, DONE = "goto_area", "search", "approach", "park", "done"

PARK_DIST_M = 1.5           # the id seen this near: park (a 15 cm marker is ~45 px wide there, readable with the blur-tolerant decoder)
STANDOFF_M = 1.1            # an approach step aims to stand this far from what it drives towards
STEP_MIN_M, STEP_MAX_M = 0.25, 0.60
SEARCH_STEP_M = 0.35
SWEEP_DEG = (0.0, 22.0, -22.0, 0.0, -22.0, 22.0)      # heading change of the successive search steps (the camera's 70 deg view covers the wall)
WALL_STOP_M = 0.55          # no marker found with the wall this near: give up (the marker is not on this wall)
MAX_STEPS = 14
MAX_S = 150.0               # the whole search, from the command


class MarkerSearch:
    def __init__(self, area, now, target_id=None):
        """area = (x, y, heading_deg) in the car's frame at the command (the same numbers as GOTO). Returns nothing; poll `step`."""
        self.area, self.t0, self.id = area, now, target_id
        self.state, self.msg, self.steps, self.done, self.ok = GOTO_AREA, "driving to the parking area", 0, False, False
        self._started = False

    # -- helpers ------------------------------------------------------------------------------------------------
    def _finish(self, ok, msg):
        self.done, self.ok, self.msg, self.state = True, ok, msg, DONE

    def _target(self, obs):
        """The nearest observation of the marker id (vehicle frame), or None."""
        o = [m for m in obs if self.id is None or m.id == self.id]
        return min(o, key=lambda m: m.dist) if o else None

    # -- one tick -----------------------------------------------------------------------------------------------
    def step(self, now, pose, nav_active, target_obs, cands, front_free, goto, park, nav_msg=""):
        """pose: the car in the world frame. nav_active: a click-to-go is planned / driving / waiting for the throttle. target_obs: MarkerObs of
        the marker id (vehicle frame). cands: [dict x, y, dist, bearing] marker-like things (any id). front_free: LiDAR free distance straight
        ahead (m) or None. goto(x, y, heading_deg) -> bool (vehicle frame NOW); park() -> message. Returns the state message."""
        if self.done:
            return self.msg
        if now - self.t0 > MAX_S:
            self._finish(False, "marker not found in time - stopped")
            return self.msg
        if not self._started:                                       # (the area goal is given in the frame of the command, which is now)
            self._started = True
            near = self._target(target_obs)
            if near is not None:                                    # the marker is in view already: no trip to the area
                self.state, self.msg = SEARCH, "marker in view"
                return self.step(now, pose, False, target_obs, cands, front_free, goto, park, "")
            # (the area's heading is NOT asked for: a goal pose turned 90 deg sent the car round in circles in a small room - 11 Nov; driving
            # straight to the point leaves it facing the way it came, towards the bay, and the search steps turn it from there)
            if not goto(self.area[0], self.area[1], None):
                self._finish(False, "no way to the parking area")
            return self.msg
        seen = self._target(target_obs)
        if nav_active:
            if seen is not None and seen.dist <= PARK_DIST_M:          # seen on the way: park from here (the bay plan replaces the step)
                self.state = PARK
                res = park()
                self._finish(res not in ("marker not visible", "no way into the bay from here"), f"marker found {seen.dist:.2f} m ahead: {res}")
            return self.msg
        low = nav_msg.lower()
        if any(w in low for w in ("cancel", "driver steered", "handed back")):         # the operator stopped it (GOTO CANCEL, steering, brake)
            self._finish(False, "cancelled")
            return self.msg
        seen = self._target(target_obs)
        if seen is not None and seen.dist <= PARK_DIST_M:
            self.state = PARK
            res = park()
            self._finish(res not in ("marker not visible", "no way into the bay from here"), f"marker found {seen.dist:.2f} m ahead: {res}")
            return self.msg
        if self.state == GOTO_AREA:                                 # the area leg ended
            if any(w in low for w in ("stuck", "no way", "no safe", "blocked", "stopped on")):
                self._finish(False, f"could not reach the parking area ({nav_msg})")
                return self.msg
            self.state, self.msg = SEARCH, "looking for the marker"
        if self.steps >= MAX_STEPS:
            self._finish(False, "marker not found - stopped")
            return self.msg
        cue = seen if seen is not None else (min(cands, key=lambda c: c["dist"]) if cands else None)
        if cue is not None:
            d, b = (cue.dist, cue.bearing) if hasattr(cue, "dist") else (cue["dist"], cue["bearing"])
            self.state = APPROACH
            step = min(max(d - STANDOFF_M, STEP_MIN_M), STEP_MAX_M)
            x, y = step * math.cos(b), step * math.sin(b)
            self.msg = f"driving towards the marker ({d:.1f} m)"
        else:
            if front_free is not None and front_free < WALL_STOP_M:
                self._finish(False, "marker not found: the wall is here and no marker in view")
                return self.msg
            self.state = SEARCH
            b = math.radians(SWEEP_DEG[self.steps % len(SWEEP_DEG)])
            step = SEARCH_STEP_M if front_free is None else max(0.15, min(SEARCH_STEP_M, front_free - WALL_STOP_M))
            x, y = step * math.cos(b), step * math.sin(b)
            self.msg = "looking for the marker"
        self.steps += 1
        if not goto(x, y, None):
            self._finish(False, "no way forward while looking for the marker")
        return self.msg
