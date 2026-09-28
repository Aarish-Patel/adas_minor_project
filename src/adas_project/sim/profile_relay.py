"""Per-tick processing cost of everything the relay runs, on realistic Monte Carlo drives (TODO K2).

    python -m sim.profile_relay [seeds]           -> table: median / 95th / max ms per call, and the per-tick
                                                     total, scaled to the Pi 5 (PI_FACTOR slower than this laptop)
    python3 -m sim.profile_relay [seeds] --on-pi  -> the same measured ON the Pi (no scaling), plus the worker
                                                     process's hand-over latency and the CPU temperature

The relay's control loop runs at ~20 Hz (50 ms); everything on it must fit with room to spare on the Pi.
"""
import collections
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sim.hw_sim import PI_COMPUTE_FACTOR as PI_FACTOR  # noqa: E402  (Pi 5 vs this laptop, single thread)
BUDGET_MS = 20.0         # of the 50 ms tick, on the Pi, for all the ADAS work together

TIMES = collections.defaultdict(list)


def timed(cls, name, label):
    orig = getattr(cls, name)

    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            TIMES[label].append((time.perf_counter() - t0) * 1000)
    setattr(cls, name, wrapper)


def main(seeds=6):
    import adas.plan_service as ps
    import pi.path_gate as pg
    import pi.relay_assists as ra
    # path searches run in the planner's worker process on the car: time them separately, and take them out of
    # the control-loop cost below (in these simulations they run inline)
    for name in ("plan_line_job", "plan_point_job"):
        orig = getattr(ps, name)

        def job(*a, _orig=orig, **k):
            t0 = time.perf_counter()
            try:
                return _orig(*a, **k)
            finally:
                TIMES["planner (worker process)"].append((time.perf_counter() - t0) * 1000)
        setattr(ps, name, job)
    timed(pg.PathGate, "decide", "gate.decide (braking)")
    timed(pg.PathGate, "on_scan", "gate.on_scan (memory)")
    timed(pg.PathGate, "overlay", "gate.overlay (GUI path)")
    timed(ra.RelayAssists, "process", "assists.process (evasive/assists)")
    timed(ra.RelayIntent, "update", "intent.update (learned model)")
    timed(ra.RelaySpeed, "on_scan", "speed.on_scan (RF2O + EKF)")
    timed(ra.RelaySpeed, "command", "speed.command (EKF predict)")
    from sim.relay_mc import DEFAULT_STYLES, run
    t0 = time.perf_counter()
    ticks = 0
    for sd in range(seeds):
        for st in DEFAULT_STYLES:
            r = run((sd, "adas+intent", st))
            ticks += len(r["trace"])
    # the relay also draws the predicted path for the GUI every packet: measure it on the last gate's state
    print(f"{ticks} control ticks over {seeds * len(DEFAULT_STYLES)} drives ({time.perf_counter() - t0:.0f} s wall)")
    print(f"{'piece':36s} {'calls':>6s} {'median':>7s} {'p95':>7s} {'max':>7s}  (ms, laptop)   p95 on the Pi")
    per_tick = 0.0
    plan_total = sum(TIMES.get("planner (worker process)", []))
    for k, v in TIMES.items():
        a = np.array(v)
        if not k.startswith("planner"):
            per_tick += a.sum() / ticks
        print(f"{k:36s} {len(a):6d} {np.median(a):7.2f} {np.percentile(a, 95):7.2f} {a.max():7.1f}  "
              f"{np.percentile(a, 95) * PI_FACTOR:8.1f}")
    per_tick -= plan_total / ticks                 # the searches are inside assists.process here, not on the car
    print(f"control loop, mean per 50 ms tick (searches excluded - they run in the worker): {per_tick:.2f} ms laptop "
          f"= ~{per_tick * PI_FACTOR:.1f} ms on the Pi (budget {BUDGET_MS:.0f} ms)")
    return TIMES


def worker_handover(n=20):
    """Round trip of a small job through the planner's worker process (what a swerve waits on besides the search)."""
    from adas.plan_service import PlanService
    svc = PlanService("process").start()
    try:
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            job = svc.submit(sum, [1, 2, 3])
            while not job.ready():
                time.sleep(0.0005)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts)), float(np.max(ts))
    finally:
        svc.shutdown()


def cpu_state():
    """CPU temperature and clock from /sys (vcgencmd needs extra permissions)."""
    out = []
    for label, path, scale in (("temp", "/sys/class/thermal/thermal_zone0/temp", 1000.0),
                               ("clock MHz", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000.0)):
        try:
            out.append(f"{label} {int(open(path).read()) / scale:.1f}")
        except (OSError, ValueError):
            pass
    return ", ".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--on-pi" in sys.argv:
        PI_FACTOR = 1.0
        print("on the Pi:", cpu_state())
    main(int(args[0]) if args else 6)
    med, mx = worker_handover()
    print(f"planner worker hand-over: median {med:.1f} ms, max {mx:.1f} ms (plus the search itself)")
    if "--on-pi" in sys.argv:
        print("after:", cpu_state())
