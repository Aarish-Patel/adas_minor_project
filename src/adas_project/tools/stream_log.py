"""Record the relay's live dashboard stream without a window (tests, debugging): one line per change.

    python tools/stream_log.py [--host 127.0.0.1] [--seconds 20]
"""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    from gui.dashboard import Link
    link = Link(a.host)
    t0, last, frames = time.time(), None, 0
    while time.time() - t0 < a.seconds:
        time.sleep(0.1)
        st, t_rx, arrivals = link.snapshot()
        if st is None:
            continue
        frames = len(arrivals)
        d = st.get("drive") or {}
        asst = st.get("assist") or {}
        g = st.get("gate") or {}
        sim = st.get("sim") or {}
        row = (d.get("pwm_in"), d.get("pwm_out"), round(float(d.get("servo") or 0)), asst.get("phase"),
               tuple(sorted((asst.get("info") or {}).items())), g.get("action"), (st.get("nav") or {}).get("state"))
        if row != last:
            pose = sim.get("pose")
            print(f"{time.time() - t0:5.1f}s pwm {row[0]}->{row[1]} servo {row[2]} phase {row[3]} gate {row[5]} "
                  f"nav {row[6]} pose {pose} crashes {sim.get('crashes')} | {dict(row[4])}")
            last = row
    print(f"stream: {frames} frames in the last window, pts in last frame {len(st['_pts']) if st else 0}")


if __name__ == "__main__":
    main()
