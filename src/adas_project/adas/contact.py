"""Virtual bumper: contact with something the LiDAR cannot see, inferred from motion.

A dark, absorbing or very low object returns no LiDAR points, so neither the brake gate nor the planner knows it is there (incident
1 Oct 2026; the twin's dark box in ROOM). The car then pushes into it - the throttle is on, nothing brakes, but the scan-matched pose does
not move. That is the signature of a bumper hit: robot vacuums use a physical bumper for exactly this and mark the hit in their map, as
navigation stacks feed bumper / range contacts into an obstacle layer (ROS Nav2 range and bumper layers). Here the "bumper" is the pose:
throttle clearly above the dead-band for WINDOW_S while the pose moved less than MIN_MOVE_M and the gate was not braking -> contact on
the side the car was driving towards. The caller marks the bumper line in the obstacle memory (adas/memory.ObstacleMemory.add_bumper):
the gate then stops pushing (no hammering of the drivetrain) and the planners route round it.
"""
import math

WINDOW_S = 0.6          # pushing this long without moving = contact
MIN_MOVE_M = 0.015      # moved less than this over the window
MIN_SPEED = 0.07        # the throttle commands at least this speed (m/s) - clearly above the dead-band (exploration creeps at ~0.10)


class ContactDetector:
    def __init__(self, pwm_min, window_s=WINDOW_S, min_move_m=MIN_MOVE_M):
        self.pwm_min, self.window_s, self.min_move_m = pwm_min, window_s, min_move_m
        self.since, self.ref, self.dir = None, None, 0
        self.contacts = 0

    def update(self, t, physical, pose, braking, stamp=None):
        """t (s), physical = throttle sent to the motor (PWM, + forward), pose = scan-matched (x, y, th), braking = the gate is acting,
        stamp = the pose's scan counter: a pose dead-reckoned between scans moves with the commanded speed even when the car is stuck,
        so only the pose of a new scan counts (the last scan pose is kept otherwise). Returns +1 / -1 on a new contact ahead / behind,
        else 0."""
        if stamp is not None:
            if stamp != getattr(self, "_stamp", None):
                self._stamp, self._scan_pose = stamp, pose
            pose = getattr(self, "_scan_pose", None)
        d = (physical > 0) - (physical < 0)
        if braking or abs(physical) < self.pwm_min or d == 0 or pose is None:
            self.since, self.ref, self.dir = None, None, 0
            return 0
        if self.since is None or d != self.dir:
            self.since, self.ref, self.dir = t, (pose[0], pose[1]), d
            return 0
        if math.hypot(pose[0] - self.ref[0], pose[1] - self.ref[1]) > self.min_move_m:
            self.since, self.ref = t, (pose[0], pose[1])        # it moves: no contact
            return 0
        if t - self.since >= self.window_s:
            self.since, self.ref = t, (pose[0], pose[1])        # (one report per window while it keeps pushing)
            self.contacts += 1
            return d
        return 0
