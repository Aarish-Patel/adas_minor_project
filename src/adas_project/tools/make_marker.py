"""Printable ArUco marker for the bay wall (TODO AN4): an A4 page with the marker at its TRUE size.

    python tools/make_marker.py [--id 23] [--size-mm 80] [--out docs/home_demo]

Defaults come from pi/camera.json (marker_id, marker_size_m), so the page always matches what the car expects. Dictionary DICT_4X4_50 (the one
adas/markers.py and pi/marker_camera.py detect). The black square (the marker without its white margin) is `size` wide: that is the number
the car's distance and angle estimate depend on - print at 100 % / "actual size" (no fit to page) and check the printed square with a ruler
against the 50 mm scale bar on the page. A white margin of one marker module is kept around it: cut along the dotted line, leave the white.
Writes aruco_<id>_<size>mm.png (300 dpi) and .pdf (A4, exact scale).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
DPI = 300
A4_MM = (210.0, 297.0)


def px(mm):
    return int(round(mm / 25.4 * DPI))


def marker_image(marker_id, size_px):
    """The marker's pixel image (uint8, 0 = black): the black square with its 4x4 bits inside a black border, exactly size_px wide."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.generateImageMarker(d, int(marker_id), int(size_px))


def page(marker_id, size_mm):
    """The A4 page as an RGB PIL image."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = px(A4_MM[0]), px(A4_MM[1])
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    side = px(size_mm)
    m = Image.fromarray(marker_image(marker_id, side)).convert("RGB")
    x0, y0 = (W - side) // 2, px(60)
    img.paste(m, (x0, y0))
    margin = side // 6                                                    # white quiet zone (one module is side/6 for a 4x4 marker)
    cut = (x0 - margin, y0 - margin, x0 + side + margin, y0 + side + margin)
    for k in range(cut[0], cut[2], 24):                                   # dotted cut line round the white margin
        d.line([(k, cut[1]), (k + 10, cut[1])], fill="gray", width=2)
        d.line([(k, cut[3]), (k + 10, cut[3])], fill="gray", width=2)
    for k in range(cut[1], cut[3], 24):
        d.line([(cut[0], k), (cut[0], k + 10)], fill="gray", width=2)
        d.line([(cut[2], k), (cut[2], k + 10)], fill="gray", width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 54)
        small = ImageFont.truetype("arial.ttf", 38)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((px(20), px(15)), f"ArUco marker  id {marker_id}  -  DICT_4X4_50  -  black square {size_mm:g} mm", fill="black", font=font)
    d.text((px(20), px(28)), "Print at 100 % (actual size, no 'fit to page'). Check the scale bar below with a ruler.", fill="black", font=small)
    bar_y = y0 + side + margin + px(25)
    d.rectangle([x0, bar_y, x0 + px(50), bar_y + 14], fill="black")
    d.text((x0 + px(50) + 20, bar_y - 12), "= 50 mm", fill="black", font=small)
    notes = ["Fix it flat on the bay's back wall, centre of the black square at the height of the camera lens,",
             "on the centre line of the bay, facing the car. Matt paper or matt laminate (no gloss: reflections hide it).",
             f"Keep the black square exactly {size_mm:g} mm: pi/camera.json marker_size_m = {size_mm / 1000:g}. If you print another size, change that number."]
    for i, line in enumerate(notes):
        d.text((px(20), bar_y + px(20) + i * 50), line, fill="black", font=small)
    return img


def main():
    ap = argparse.ArgumentParser()
    from adas import camera_config
    cfg = camera_config.load()
    ap.add_argument("--id", type=int, default=int(cfg["marker_id"]))
    ap.add_argument("--size-mm", type=float, default=float(cfg["marker_size_m"]) * 1000.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "home_demo"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img = page(a.id, a.size_mm)
    base = os.path.join(a.out, f"aruco_{a.id}_{a.size_mm:g}mm")
    img.save(base + ".png", dpi=(DPI, DPI))
    from PIL import Image
    img.convert("1", dither=Image.Dither.NONE).save(base + ".pdf", resolution=float(DPI))     # 1 bit: lossless, sharp marker edges (no JPEG)
    print("wrote", base + ".png", "and .pdf")


if __name__ == "__main__":
    main()
