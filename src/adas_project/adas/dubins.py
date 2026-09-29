"""Dubins paths (Dubins 1957; closed-form words from Shkel & Lumelsky 2001): the shortest forward path between two
poses for a car with a minimum turning radius, made of three pieces - turn / straight / turn (LSL, RSR, LSR, RSL)
or three turns (RLR, LRL). Used as the analytic expansion of the Hybrid A* when the goal has a heading
(adas/hybrid_astar.plan_to_point), as Dolgov et al. use Reeds-Shepp curves.
"""
import math

import numpy as np

TWO_PI = 2 * math.pi


def _mod(a):
    return a % TWO_PI


def _words(alpha, beta, d):
    """All feasible words: (name, t, p, q) in normalised units (radius 1)."""
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    out = []
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)                  # LSL
    if p2 >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        out.append(("LSL", _mod(-alpha + tmp), math.sqrt(p2), _mod(beta - tmp)))
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)                  # RSR
    if p2 >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        out.append(("RSR", _mod(alpha - tmp), math.sqrt(p2), _mod(-beta + tmp)))
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)                 # LSR
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(("LSR", _mod(-alpha + tmp), p, _mod(-_mod(beta) + tmp)))
    p2 = -2 + d * d + 2 * cab - 2 * d * (sa + sb)                 # RSL
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(("RSL", _mod(alpha - tmp), p, _mod(beta - tmp)))
    c = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0          # RLR
    if abs(c) <= 1:
        p = _mod(TWO_PI - math.acos(c))
        t = _mod(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        out.append(("RLR", t, p, _mod(alpha - beta - t + p)))
    c = (6.0 - d * d + 2 * cab + 2 * d * (-sa + sb)) / 8.0         # LRL
    if abs(c) <= 1:
        p = _mod(TWO_PI - math.acos(c))
        t = _mod(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        out.append(("LRL", t, p, _mod(_mod(beta) - alpha - t + p)))
    return out


def shortest(start, goal, radius):
    """(word, lengths in metres (3,), total length) of the shortest Dubins path, or None."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    D = math.hypot(dx, dy)
    d = D / radius
    th = math.atan2(dy, dx) if D > 1e-9 else 0.0
    alpha, beta = _mod(start[2] - th), _mod(goal[2] - th)
    words = _words(alpha, beta, d)
    if not words:
        return None
    name, t, p, q = min(words, key=lambda w: w[1] + w[2] + w[3])
    return name, np.array([t, p, q]) * radius, (t + p + q) * radius


def length(start, goal, radius):
    s = shortest(start, goal, radius)
    return math.inf if s is None else s[2]


def sample(start, goal, radius, ds=0.04):
    """Poses (N, 3) along the shortest Dubins path (excluding the start), or None."""
    s = shortest(start, goal, radius)
    if s is None:
        return None
    name, segs, _ = s
    x, y, th = start
    out = []
    for kind, L in zip(name, segs):
        n = max(1, int(math.ceil(L / ds)))
        step = L / n
        k = 0.0 if kind == "S" else (1.0 / radius if kind == "L" else -1.0 / radius)
        for _ in range(n):
            if k == 0.0:
                x += step * math.cos(th)
                y += step * math.sin(th)
            else:
                nth = th + k * step
                x += (math.sin(nth) - math.sin(th)) / k
                y -= (math.cos(nth) - math.cos(th)) / k
                th = nth
            out.append((x, y, th))
    return np.array(out) if out else None


def sample_reverse(start, goal, radius, ds=0.04):
    """The same, driven BACKWARDS (the car's heading stays its heading): the mirror problem with both headings
    turned by pi, sampled, then turned back."""
    fw = sample((start[0], start[1], start[2] + math.pi), (goal[0], goal[1], goal[2] + math.pi), radius, ds)
    if fw is None:
        return None
    fw[:, 2] -= math.pi
    return fw
