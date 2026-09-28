"""Drivetrain stress in a relay drive log (after the rear shaft broke on 28 Sep with no crash).

    python tools/drivetrain_report.py logs/<file>_relay.jsonl.gz [...]

From every motor command the relay sent ("cmd" records, M lines, physical sign: + forward) and the speed estimate:
  reversals   the command flipped direction while the car was still moving (gearbox / shaft shock load)
  brake       reverse pulses sent to stop the car (active braking) - count and speed when fired
  jumps       throttle steps larger than 120 PWM between consecutive commands
  stalls      throttle above 40 PWM held for more than 1 s while the car stood still (pushing against something)
"""
import gzip
import json
import sys
import zlib


def records(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="ignore") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError, zlib.error):
                return                          # a log cut short (power loss, relay killed) - use what is there
            if not line:
                return
            try:
                yield json.loads(line)
            except ValueError:
                continue


def analyse(path, motor_reversed=True):
    cmds, speed = [], []                       # (t, physical), (t, v)
    for r in records(path):
        if r.get("k") == "cmd" and str(r.get("line", "")).startswith("M "):
            try:
                w = float(r["line"].split()[1])
            except (IndexError, ValueError):
                continue
            cmds.append((r["t"], -w if motor_reversed else w))
        elif r.get("k") == "drv" and r.get("v_est") is not None:
            speed.append((r["t"], float(r["v_est"])))
    out = {"file": path, "commands": len(cmds), "reversals": [], "brakes": [], "jumps": 0, "max_jump": 0,
           "stalls": [], "duration_s": round(cmds[-1][0] - cmds[0][0], 1) if cmds else 0}
    if not cmds:
        return out
    si = 0

    def v_at(t):
        nonlocal si
        while si + 1 < len(speed) and speed[si + 1][0] <= t:
            si += 1
        return speed[si][1] if speed else 0.0
    prev_t, prev_u, last_nz = cmds[0][0], cmds[0][1], cmds[0][1]
    stall_from = None
    for t, u in cmds[1:]:
        v = v_at(t)
        jump = abs(u - prev_u)
        out["max_jump"] = max(out["max_jump"], jump)
        if jump > 120:
            out["jumps"] += 1
        if u != 0 and last_nz != 0 and u * last_nz < 0 and abs(v) > 0.1:
            kind = "brake" if u * v < 0 and abs(u) >= 60 else "reversal"
            if kind == "brake":
                out["brakes"].append((round(t % 1000, 2), round(v, 2), int(u)))
            else:
                out["reversals"].append((round(t % 1000, 2), round(v, 2), int(last_nz), int(u)))
        if abs(u) > 40 and abs(v) < 0.02:
            stall_from = t if stall_from is None else stall_from
        else:
            if stall_from is not None and t - stall_from > 1.0:
                out["stalls"].append((round(stall_from % 1000, 1), round(t - stall_from, 1)))
            stall_from = None
        if u != 0:
            last_nz = u
        prev_t, prev_u = t, u
    return out


def main():
    for path in sys.argv[1:]:
        r = analyse(path)
        print(f"{r['file']}: {r['commands']} commands over {r['duration_s']} s")
        print(f"  direction reversals while moving: {len(r['reversals'])}  {r['reversals'][:6]}")
        print(f"  reverse brake pulses: {len(r['brakes'])}  (t, speed m/s, pwm) {r['brakes'][:6]}")
        print(f"  throttle jumps > 120 PWM: {r['jumps']} (largest {r['max_jump']:.0f})")
        print(f"  stalls (> 40 PWM, standing, > 1 s): {len(r['stalls'])}  (t, s) {r['stalls'][:6]}")


if __name__ == "__main__":
    main()
