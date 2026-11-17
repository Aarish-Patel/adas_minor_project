"""Shareable demo clips: each raw dashboard recording (reports/demo_videos/<clip>.mp4, OpenCV mp4v) gets a title bar saying what the clip
shows and is re-encoded as H.264 / yuv420p / faststart MP4 with ffmpeg, which phones and WhatsApp play (mp4v does not).

    python tools/label_videos.py            -> reports/demo_videos/share/NN_<clip>.mp4
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "reports", "demo_videos")
OUT = os.path.join(SRC, "share")
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"

CLIPS = [
    ("overview", "Live map and LiDAR view", "Driving by hand in the simulated room: the 2D LiDAR builds the map, the dashboard follows the car."),
    ("goto_home", "Click-to-go and return to start", "A click on the map: Hybrid A* plans a path, the car drives it, re-plans round a new obstacle, then drives home."),
    ("emergency_brake", "Emergency brake", "Full throttle at a wall, no steering: the path-predicted brake gate stops the car short of it."),
    ("evasive_steer", "Evasive steer", "A box appears in the lane with the throttle held: the car plans a forward way round it and steers past."),
    ("speed_zone", "Speed-limit zone", "A zone drawn on the map: inside it the car is held to the zone's speed whatever the throttle."),
    ("parallel_park", "Parallel parking", "Click on the street's parking lane: the car finds the gap with its LiDAR and reverses in, straight, near the kerb."),
    ("bay_park", "Bay parking at the ArUco marker", "The camera finds the bay's marker, the LiDAR the walls: the car drives in nose first and stops at the wall."),
    ("monte_carlo", "Monte Carlo lab", "Hundreds of randomised drives replayed: the same scenes with the ADAS off, brake-only, ADAS and ADAS + intent."),
    ("ml_training", "ML training lab", "The learned driver-intent model: training batches, its accuracy and what it changes in the assist."),
    ("pedestrian", "Moving obstacle", "A walker steps across ahead: the tracker follows it, the gate warns and brakes for what is in the path."),
]


def banner(frame, title, sub, idx):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame[:, :, ::-1])
    W = img.width
    bar = Image.new("RGB", (W, 74), (17, 24, 39))
    d = ImageDraw.Draw(bar)
    d.rectangle([0, 0, 8, 74], fill=(59, 130, 246))
    d.text((24, 8), f"{idx:02d}  {title}", font=ImageFont.truetype(FONT_B, 28), fill=(241, 245, 249))
    d.text((24, 44), sub, font=ImageFont.truetype(FONT, 17), fill=(148, 163, 184))
    out = Image.new("RGB", (W, img.height + 74))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 74))
    a = np.asarray(out)[:, :, ::-1]
    return a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]


def convert(name, title, sub, idx):
    import cv2
    src = os.path.join(SRC, f"{name}.mp4")
    if not os.path.exists(src):
        print("missing", src)
        return None
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    ok, f = cap.read()
    if not ok:
        return None
    first = banner(f, title, sub, idx)
    h, w = first.shape[:2]
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{idx:02d}_{name}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-vf", "fps=24", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(first.tobytes())
    for _ in range(int(fps * 1.5)):                          # the title held a moment on the first frame
        p.stdin.write(first.tobytes())
    while True:
        ok, f = cap.read()
        if not ok:
            break
        p.stdin.write(banner(f, title, sub, idx).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", os.path.relpath(dst, ROOT))
    return dst


def main(argv):
    names = argv or [c[0] for c in CLIPS]
    for i, (name, title, sub) in enumerate(CLIPS, 1):
        if name in names:
            convert(name, title, sub, i)


if __name__ == "__main__":
    main(sys.argv[1:])
