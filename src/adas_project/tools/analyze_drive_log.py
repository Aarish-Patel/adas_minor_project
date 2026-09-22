"""Summarize and plot a real drive log recorded by pi/wifi_drive_safety.py.

    python tools/analyze_drive_log.py path/to/drive_20260101_120000.csv [out.png]

Download a log first, e.g.:
    scp pi@192.168.1.3:/home/pi/rc_car/pi/drive_logs/drive_*.csv .
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()})
    return rows


def summarize(rows):
    t0 = rows[0]["t"]
    duration = rows[-1]["t"] - t0
    front = [r["front_dist"] for r in rows if r["front_dist"] is not None]
    rear = [r["rear_dist"] for r in rows if r["rear_dist"] is not None]
    blocked_frac = np.mean([r["front_blocked"] or r["rear_blocked"] for r in rows])
    capped = sum(1 for r in rows if r["pwm_sent"] != r["pwm_commanded"])
    print(f"duration: {duration:.1f} s ({len(rows)} rows)")
    print(f"closest front seen: {min(front)*100:.1f} cm" if front else "no front readings")
    print(f"closest rear seen:  {min(rear)*100:.1f} cm" if rear else "no rear readings")
    print(f"time spent blocked: {blocked_frac * 100:.1f}%")
    print(f"ticks where the safety gate actually capped the throttle: {capped} ({100 * capped / len(rows):.1f}%)")
    fastest_front = max((r["front_speed"] for r in rows if r["front_speed"] is not None), default=0.0)
    fastest_rear = max((r["rear_speed"] for r in rows if r["rear_speed"] is not None), default=0.0)
    print(f"fastest closing speed seen: front {fastest_front:.2f} m/s, rear {fastest_rear:.2f} m/s")


def plot(rows, out):
    t0 = rows[0]["t"]
    t = np.array([r["t"] - t0 for r in rows])

    def series(key):
        return np.array([r[key] if r[key] is not None else np.nan for r in rows])

    front, rear = series("front_dist"), series("rear_dist")
    front_b, rear_b = series("front_blocked"), series("rear_blocked")
    pwm_cmd, pwm_sent = series("pwm_commanded"), series("pwm_sent")

    plt.rcParams.update({"figure.facecolor": "#0b1120", "axes.facecolor": "#111a2e",
                         "savefig.facecolor": "#0b1120", "axes.edgecolor": "#33415f",
                         "axes.labelcolor": "#c7d2fe", "text.color": "#e5ecff",
                         "xtick.color": "#94a3b8", "ytick.color": "#94a3b8", "axes.grid": True,
                         "grid.color": "#1e2b47"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, front, color="#2dd4bf", lw=1.2, label="front clearance (m)")
    ax1.plot(t, rear, color="#facc15", lw=1.2, label="rear clearance (m)")
    ax1.fill_between(t, 0, 2, where=(front_b > 0), color="#ef4444", alpha=0.15, step="post")
    ax1.fill_between(t, 0, 2, where=(rear_b > 0), color="#ef4444", alpha=0.15, step="post")
    ax1.set_ylim(0, min(2.0, np.nanmax(np.concatenate([front, rear])) * 1.1 if len(front) else 2))
    ax1.set_ylabel("clearance (m)")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title("Red shading = safety gate blocked", fontsize=10)

    ax2.plot(t, pwm_cmd, color="#60a5fa", lw=1.0, label="PWM commanded (you)")
    ax2.plot(t, pwm_sent, color="#34d399", lw=1.4, label="PWM sent (after safety cap)")
    ax2.set_ylabel("PWM")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0] + ".png"
    rows = load(path)
    if not rows:
        print("empty log")
        return
    summarize(rows)
    plot(rows, out)


if __name__ == "__main__":
    main()
