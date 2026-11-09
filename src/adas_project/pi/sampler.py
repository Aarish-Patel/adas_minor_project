"""A tiny sampling profiler for the relay (stdlib only): where does a Python process spend its time, per thread.

Started by the relay only when the flag file `~/rc_car/profile.flag` exists (nothing runs otherwise). A daemon thread looks at every other thread's
current stack every 10 ms (sys._current_frames) and counts the innermost project function (pi/ or adas/) and the innermost line; every 20 s it
rewrites `~/relay_profile.txt` with the top entries per thread. Cost: ~1 % of a core. Written for the 9 Nov question "why does the relay need
more than one core on an idle car?" (the Pi has no py-spy and nothing may be installed without permission).
"""
import collections
import os
import sys
import threading
import time

FLAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profile.flag")
OUT = os.path.expanduser("~/relay_profile.txt")
PERIOD_S, DUMP_S = 0.01, 20.0


def _project_frame(frame):
    """The innermost frame inside the project (pi/ or adas/), or None."""
    while frame is not None:
        fn = frame.f_code.co_filename.replace("\\", "/")
        if "/rc_car/pi/" in fn or "/rc_car/adas/" in fn or "/RC_Car/pi/" in fn or "/RC_Car/adas/" in fn:
            return frame
        frame = frame.f_back
    return None


def _label(frame):
    fn = frame.f_code.co_filename.replace("\\", "/").split("/")
    return f"{'/'.join(fn[-2:])}:{frame.f_code.co_name}:{frame.f_lineno}"


class Sampler:
    def __init__(self):
        self.funcs = collections.defaultdict(collections.Counter)      # thread name -> project function counts
        self.lines = collections.defaultdict(collections.Counter)      # thread name -> innermost-line counts
        self.chains = collections.defaultdict(collections.Counter)     # thread name -> "inner <- caller <- caller" (project frames)
        self.n = 0
        self.t0 = time.time()

    def sample(self):
        names = {t.ident: t.name for t in threading.enumerate()}
        me = threading.get_ident()
        self.n += 1
        for tid, frame in sys._current_frames().items():
            if tid == me:
                continue
            name = names.get(tid, str(tid))
            inner = frame
            self.lines[name][_label(inner)] += 1
            pf = _project_frame(frame)
            if pf is not None:
                self.funcs[name][_label(pf).rsplit(":", 1)[0]] += 1     # function, not line
                chain, f = [], pf
                while f is not None and len(chain) < 4:
                    if _project_frame(f) is f:
                        chain.append(_label(f).rsplit(":", 1)[0])
                    f = f.f_back
                self.chains[name][" <- ".join(chain)] += 1

    def report(self, top=14):
        out = [f"relay sampling profile: {self.n} samples every {PERIOD_S * 1000:.0f} ms over {time.time() - self.t0:.0f} s "
               f"(a thread's share = its samples / {self.n}; waiting threads show their blocking call)"]
        for name in sorted(self.funcs, key=lambda k: -sum(self.lines[k].values())):
            tot = sum(self.lines[name].values())
            out.append(f"\n== thread {name}: {100.0 * tot / max(self.n, 1):.0f} % of samples")
            out.append("  project functions (innermost):")
            for lab, c in self.funcs[name].most_common(top):
                out.append(f"    {100.0 * c / max(self.n, 1):5.1f} %  {lab}")
            out.append("  call chains (project frames, innermost first):")
            for lab, c in self.chains[name].most_common(8):
                out.append(f"    {100.0 * c / max(self.n, 1):5.1f} %  {lab}")
            out.append("  innermost lines:")
            for lab, c in self.lines[name].most_common(6):
                out.append(f"    {100.0 * c / max(self.n, 1):5.1f} %  {lab}")
        return "\n".join(out)


def maybe_start():
    """Start the sampler thread when the flag file exists. Returns the Sampler or None."""
    if not os.path.exists(FLAG):
        return None
    s = Sampler()

    def run():
        last = time.time()
        while True:
            time.sleep(PERIOD_S)
            s.sample()
            if time.time() - last >= DUMP_S:
                last = time.time()
                try:
                    with open(OUT, "w") as f:
                        f.write(s.report())
                except OSError:
                    pass
    threading.Thread(target=run, name="sampler", daemon=True).start()
    print("sampling profiler on (profile.flag exists): ~/relay_profile.txt every 20 s", flush=True)
    return s
