"""Publish the working tree to GitHub in the published layout, in one command.

    python tools/publish.py              # dry run: shows what would be pushed, changes nothing on GitHub
    python tools/publish.py --push -m "message"   # commit the new state and push it

Published layout: repository Aarish-Patel/adas_minor_project, branch real-car-adas, the project under src/adas_project/
(no README). The file set is `git ls-files --cached --others --exclude-standard` (so every git-ignored file is left out:
logs, *.log, model binaries, __pycache__, drive logs ...) minus the local-only paths in EXCLUDE (data/, logs/,
gui/zones.json). The set is copied into a cached clone (~/.cache/adas_publish), which is fetched first; files that
disappeared from the working tree are deleted from src/adas_project/ so the copy is exact.

Push is refused if the dry run finds something that looks like a password / token in the files to publish.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = "https://github.com/Aarish-Patel/adas_minor_project.git"
BRANCH = "real-car-adas"
DEST = "src/adas_project"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "adas_publish")
EXCLUDE = ("data/", "logs/", "gui/zones.json")
BIG_FILE_MB = 20.0
SECRET = re.compile(r"(password|passwd|token|api[_-]?key|secret)\s*[:=]\s*[\"'][^\"']{4,}[\"']", re.I)
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def run(args, cwd=None, check=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"{' '.join(args)}\n{r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def files_to_publish():
    out = run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    names = sorted({n for n in out.split("\0") if n})
    keep = [n for n in names if not n.startswith(EXCLUDE) and os.path.isfile(os.path.join(ROOT, n))]
    return keep


def scan_for_secrets(names):
    hits = []
    for n in names:
        p = os.path.join(ROOT, n)
        if os.path.getsize(p) > 2_000_000 or n.endswith((".png", ".jpg", ".gz", ".npz", ".joblib", ".pkl", ".npy")):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if SECRET.search(line) and "os.environ" not in line and "getpass" not in line:
                        hits.append(f"{n}:{i}: {line.strip()[:100]}")
        except (UnicodeDecodeError, OSError):
            continue
    return hits


def prepare_clone():
    if not os.path.isdir(os.path.join(CACHE, ".git")):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        run(["git", "clone", "--quiet", REPO, CACHE])
    run(["git", "fetch", "--quiet", "origin"], cwd=CACHE)
    run(["git", "reset", "--quiet", "--hard"], cwd=CACHE, check=False)
    run(["git", "clean", "-qfd"], cwd=CACHE, check=False)
    remote = run(["git", "branch", "-r", "--list", f"origin/{BRANCH}"], cwd=CACHE).strip()
    if remote:
        run(["git", "checkout", "--quiet", "-B", BRANCH, f"origin/{BRANCH}"], cwd=CACHE)
    else:
        print(f"note: branch {BRANCH} does not exist on GitHub yet - it will be created from the default branch")
        run(["git", "checkout", "--quiet", "-B", BRANCH], cwd=CACHE)


def sync(names):
    dest = os.path.join(CACHE, *DEST.split("/"))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    for n in names:
        target = os.path.join(dest, *n.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, n), target)
    run(["git", "add", "-A", "--", DEST], cwd=CACHE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--push", action="store_true", help="commit and push (default: dry run)")
    ap.add_argument("-m", "--message", help="commit message (required with --push)")
    a = ap.parse_args()
    if a.push and not a.message:
        sys.exit("--push needs -m 'commit message'")

    names = files_to_publish()
    size = sum(os.path.getsize(os.path.join(ROOT, n)) for n in names)
    print(f"{len(names)} files, {size / 1e6:.1f} MB from {ROOT} -> {REPO} {BRANCH}:{DEST}/")
    for n in names:
        mb = os.path.getsize(os.path.join(ROOT, n)) / 1e6
        if mb > BIG_FILE_MB:
            print(f"  WARNING large file {n} ({mb:.1f} MB)")
    dirty = run(["git", "status", "--porcelain"], cwd=ROOT).strip()
    if dirty:
        print("note: the working tree has uncommitted changes; they are published as they are on disk:")
        print("  " + "\n  ".join(dirty.splitlines()[:15]))
    hits = scan_for_secrets(names)
    if hits:
        print("Possible secrets in the files to publish:\n  " + "\n  ".join(hits))

    prepare_clone()
    sync(names)
    status = run(["git", "status", "--porcelain", "--", DEST], cwd=CACHE).splitlines()
    kinds = {"A": [], "M": [], "D": [], "R": []}
    for line in status:
        kinds.setdefault(line[0], []).append(line[3:])
    print(f"vs GitHub: {len(kinds['A'])} added, {len(kinds['M'])} modified, {len(kinds['D'])} deleted, "
          f"{len(kinds['R'])} renamed")
    for k, label in (("A", "added"), ("M", "modified"), ("D", "deleted")):
        for n in kinds[k][:40]:
            print(f"  {label:8s} {n}")
        if len(kinds[k]) > 40:
            print(f"  ... {len(kinds[k]) - 40} more {label}")
    if not status:
        print("nothing to publish - GitHub already matches")
        return
    if not a.push:
        print("dry run: nothing committed or pushed. Re-run with --push -m 'message' to publish.")
        run(["git", "reset", "--quiet"], cwd=CACHE)
        return
    if hits:
        sys.exit("refusing to push: remove or allow-list the possible secrets above first")
    run(["git", "commit", "--quiet", "-m", f"{a.message}\n\n{TRAILER}"], cwd=CACHE)
    run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=CACHE)
    print("pushed", run(["git", "rev-parse", "--short", "HEAD"], cwd=CACHE).strip())


if __name__ == "__main__":
    main()
