"""Headless screenshot helper for reviewing the viewer:  python tools/shot.py out.png [seconds] [scenario] [keys...]"""
import sys, time
from playwright.sync_api import sync_playwright

out = sys.argv[1]
wait = float(sys.argv[2]) if len(sys.argv) > 2 else 5
scenario = sys.argv[3] if len(sys.argv) > 3 else None
keys = sys.argv[4:]
W, H = 1600, 900

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True,
                          args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": W, "height": H})
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    pg.goto("http://localhost:8765/", wait_until="load")
    time.sleep(1.5)
    if scenario:
        pg.select_option("#scenario", scenario)
        time.sleep(0.8)
    for k in keys:
        pg.keyboard.down(k)
    time.sleep(wait)
    pg.screenshot(path=out)
    for k in keys:
        pg.keyboard.up(k)
    for l in logs[:25]:
        print(l)
    b.close()
print("saved", out)
