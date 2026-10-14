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

WINDOW_S = 0.4          # pushing this long without moving = contact (5 scans at 12 Hz; pushes are often cut after ~0.6 s)
MIN_MOVE_M = 0.015      # moved less than this over the window
LAG_S = 0.15            # motor + servo lag before a commanded speed shows in the pose
MIN_SPEED = 0.05        # the throttle commands at least this speed (m/s) - above the dead-band (~11 PWM); creeping pushes run at PWM 30-40


class ContactDetector:
    def __init__(self, pwm_min, window_s=WINDOW_S, min_move_m=MIN_MOVE_M, speed_of=None):
        """speed_of(pwm) -> m/s (the car's throttle model): a slow creep gets the time it needs to move 2 x min_move_m plus LAG_S
        before it counts as stuck, so a slow start is not taken for a contact."""
        self.pwm_min, self.window_s, self.min_move_m, self.speed_of = pwm_min, window_s, min_move_m, speed_of
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
        window = self.window_s
        if self.speed_of is not None:
            window = max(window, 2.0 * self.min_move_m / max(abs(self.speed_of(abs(physical))), 0.02) + LAG_S)
        if t - self.since >= window:
            self.since, self.ref = t, (pose[0], pose[1])        # (one report per window while it keeps pushing)
            self.contacts += 1
            return d
        return 0
