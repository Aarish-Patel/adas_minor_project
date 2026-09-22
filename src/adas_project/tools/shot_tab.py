"""python tools/shot_tab.py out.png scenario tab wait [keys...]   (tab: live|tune|cam|results)"""
import sys, time
from playwright.sync_api import sync_playwright
out, scenario, tab, wait = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
keys = sys.argv[5:]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    pg.goto("http://localhost:8765/", wait_until="load"); time.sleep(1.5)
    pg.select_option("#scenario", scenario); time.sleep(0.8)
    pg.click(f"#tabs [data-tab='{tab}']")
    for k in keys:
        pg.keyboard.press(k)
    time.sleep(wait)
    pg.screenshot(path=out)
    for l in logs[:8]: print(l)
    b.close()
print("saved", out)
