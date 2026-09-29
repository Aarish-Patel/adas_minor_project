"""Responsibility-Sensitive Safety (Shalev-Shwartz, Shammah, Shashua, "On a Formal Model of Safe and Scalable
Self-driving Cars", Mobileye, arXiv 1708.06374, 2017) - the formal safe-distance rules, as an independent check on the
ADAS and a number the driver can see (TODO N7).

Longitudinal (RSS, definition 1): the minimum gap between a car behind (speed v_r) and an object ahead (speed v_f, in the
same direction) such that the car, whatever the object does within its physical limits, can avoid a collision by
responding properly - accelerating at most a_accel for the response time rho, then braking at least b_min:

    d_min = max(0, v_r rho + 1/2 a_accel rho^2 + (v_r + rho a_accel)^2 / (2 b_min) - v_f^2 / (2 b_max))

For a stationary object v_f = 0 (a wall, a box). b_max is the hardest the object ahead could brake.
Lateral (definition 3): two objects side by side with lateral speeds v1, v2 keep at least mu plus the lateral braking
distances of both, after both accelerate laterally for rho.

Constants are this car's: rho = 0.20 s (scan + relay + motor, the same as the brake gate's REACTION_S), a_accel 0.8 m/s^2
(measured motor lag), b_min 2.5 m/s^2 (comfortable, well under the ~4 m/s^2 a throttle cut gives and the 8 m/s^2 of an
active brake), b_max 4.0 m/s^2 (a moving obstacle's hardest stop), lateral mu 0.02 m.

RSS is stricter than the gate on purpose (b_min 2.5, not the car's real 4-8): the gate is tuned to stop as late as physics
allows without contact, RSS is the yardstick for 'was the car ever at risk' - shown as a margin in the HMI and used in
the report to compare a driver alone with the ADAS.
"""
from dataclasses import dataclass


@dataclass
class RSSParams:
    rho: float = 0.20
    a_accel: float = 0.8
    b_min: float = 2.5
    b_max: float = 4.0
    mu: float = 0.02
    a_lat: float = 1.0
    b_lat: float = 2.0


def longitudinal_min_distance(v_rear, v_front=0.0, p=None):
    """Minimum safe gap (m) to an object ahead moving at v_front (m/s, same direction; 0 = stationary)."""
    p = p or RSSParams()
    v = max(0.0, v_rear)
    vf = max(0.0, v_front)
    d = v * p.rho + 0.5 * p.a_accel * p.rho ** 2 + (v + p.rho * p.a_accel) ** 2 / (2 * p.b_min) - vf ** 2 / (2 * p.b_max)
    return max(0.0, d)


def lateral_min_distance(v1, v2, p=None):
    """Minimum safe lateral gap (m) between two objects moving towards each other laterally (v1 > 0 to the right, v2 < 0
    to the left for approach), definition 3 of the RSS paper."""
    p = p or RSSParams()
    v1r = v1 + p.rho * p.a_lat
    v2r = v2 - p.rho * p.a_lat
    d = p.mu + max(0.0, (v1 + v1r) / 2 * p.rho + v1r ** 2 / (2 * p.b_lat) - ((v2 + v2r) / 2 * p.rho - v2r ** 2 / (2 * p.b_lat)))
    return max(0.0, d)


def margin(free_m, v, v_front=0.0, p=None):
    """free_m minus the RSS minimum: positive = a proper response exists, negative = the car is inside the unsafe zone."""
    return free_m - longitudinal_min_distance(v, v_front, p)
