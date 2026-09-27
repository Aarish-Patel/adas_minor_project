"""Copy new drive logs from the Pi to RC_Car/logs/ (skips ones already here), optionally fit the simulator.

    python tools/pull_logs.py [--host 192.168.1.6] [--fit]

The password comes from the RC_PI_PASSWORD environment variable, or is asked for.
"""
import argparse
import getpass
import os
import stat
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOCAL = os.path.join(ROOT, "logs")
REMOTE = "/home/pi/logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.6")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--fit", action="store_true", help="fit sim/fitted_car.json to all logs afterwards")
    a = ap.parse_args()
    import paramiko
    pw = os.environ.get("RC_PI_PASSWORD") or getpass.getpass(f"{a.user}@{a.host} password: ")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(a.host, username=a.user, password=pw, timeout=10)
    sf = c.open_sftp()
    os.makedirs(LOCAL, exist_ok=True)
    new = 0
    for e in sorted(sf.listdir_attr(REMOTE), key=lambda e: e.filename):
        if not e.filename.endswith(".jsonl.gz") or stat.S_ISDIR(e.st_mode):
            continue
        dst = os.path.join(LOCAL, e.filename)
        if os.path.exists(dst) and os.path.getsize(dst) == e.st_size:
            continue
        sf.get(f"{REMOTE}/{e.filename}", dst)
        new += 1
        print(f"  {e.filename}  {e.st_size / 1e6:.1f} MB")
    sf.close()
    c.close()
    print(f"{new} new log(s) in {os.path.abspath(LOCAL)}")
    if a.fit:
        logs = sorted(os.path.join(LOCAL, f) for f in os.listdir(LOCAL) if f.endswith(".jsonl.gz"))
        subprocess.run([sys.executable, "-m", "sim.log_fit", *logs], cwd=ROOT)


if __name__ == "__main__":
    main()
