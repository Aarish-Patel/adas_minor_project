"""Generate the ArUco marker images used by the viewer (and for printing)."""
import os

import cv2

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "markers")
os.makedirs(OUT, exist_ok=True)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for marker_id in list(range(0, 12)) + [21, 22, 23]:
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, 400)
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)   # white quiet zone
    cv2.imwrite(os.path.join(OUT, f"marker_{marker_id}.png"), img)
print("wrote", len(os.listdir(OUT)), "markers to", OUT)
