"""Teleport the car near the parking marker and capture the camera tab with OpenCV detections."""
import sys, time
from playwright.sync_api import sync_playwright
out = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    logs = []
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    pg.goto("http://localhost:8765/", wait_until="load"); time.sleep(1.5)
    pg.select_option("#scenario", "parking"); time.sleep(2.5)
    pg.click("#tabs [data-tab='cam']")
    pg.evaluate("window.__send({type:'cmd', cmd:'teleport', x:3.1, y:0.12, theta:8})")
    time.sleep(6)
    pg.evaluate("window.__send({type:'cmd', cmd:'pause'})")   # freeze so detection is stable
    time.sleep(3)
    pg.screenshot(path=out)
    txt = pg.inner_text("#camInfo")
    print(txt.replace("\n", " | ")[:400])
    for l in logs[:5]: print(l)
    b.close()
print("saved", out)
