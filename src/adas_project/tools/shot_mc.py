import sys, time
from playwright.sync_api import sync_playwright
out = sys.argv[1]; n = sys.argv[2] if len(sys.argv) > 2 else "24"; wait = float(sys.argv[3]) if len(sys.argv) > 3 else 6
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type in ("error",) else None)
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    pg.goto("http://localhost:8765/", wait_until="load"); time.sleep(1.5)
    pg.click("#btnMc"); time.sleep(0.5)
    pg.click(f"#mcLab [data-n='{n}']")
    time.sleep(wait)
    pg.screenshot(path=out)
    for l in logs[:10]: print(l)
    b.close()
print("saved", out)
