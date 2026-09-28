"""Image quality and camera-measured vibration (TODO P20; the real car vibrates a lot - user, 28 Sep).

Per frame:
  blur        variance of the Laplacian (Pech-Pacheco et al. 2000): low = motion-blurred or defocused
  brightness  mean grey level; contrast: its standard deviation (under / over exposure, a covered lens)
  vibration   the camera is rigidly mounted on the chassis, so high-frequency image shift IS chassis vibration: the
              global translation between consecutive frames (phase correlation, Kuglin & Hines 1975; sub-pixel by
              cv2.phaseCorrelate) with the slow part removed (the car's own motion) - RMS in pixels and degrees of camera
              angle, and the dominant frequency from an FFT. A number the ADAS can use: how shaky is the car right now, so
              how much to trust the LiDAR scan-to-scan and how wide the safety margins should be.
The result is also a health signal: 'degraded' when the image cannot be trusted (blur, lens covered, shaking).
"""
import collections
import math

import numpy as np


class ImageQuality:
    def __init__(self, fx, window=32, blur_min=40.0, dark=25.0, bright=235.0, jitter_max_deg=1.2):
        import cv2
        self.cv2, self.fx = cv2, fx
        self.shifts = collections.deque(maxlen=window)          # (t, dx, dy)
        self.prev = None
        self.win = None
        self.blur_min, self.dark, self.bright, self.jitter_max = blur_min, dark, bright, jitter_max_deg

    def update(self, frame, t):
        cv2 = self.cv2
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        out = {"blur": blur, "brightness": float(gray.mean()), "contrast": float(gray.std()),
               "jitter_px": 0.0, "jitter_deg": 0.0, "vib_hz": 0.0}
        g = gray.astype(np.float32)
        if self.win is None or self.win.shape != g.shape:
            self.win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if self.prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev[0], g, self.win)
            # a real chassis shake moves the image by a few pixels between frames; a jump of a quarter of the frame is
            # a discontinuity (dropped frames, a scene cut) and would read as huge vibration
            if resp > 0.05 and abs(dx) < 0.25 * g.shape[1] and abs(dy) < 0.25 * g.shape[0]:
                self.shifts.append((t, dx, dy))
        self.prev = (g, t)
        if len(self.shifts) >= 12:
            a = np.array(self.shifts)
            ts, dx, dy = a[:, 0], a[:, 1], a[:, 2]
            k = 5                                                # remove the slow part: a 5-sample moving mean
            hp = []
            for s in (dx, dy):
                slow = np.convolve(s, np.ones(k) / k, mode="same")
                hp.append((s - slow)[k:-k])
            rms = float(np.sqrt(np.mean(hp[0] ** 2 + hp[1] ** 2)))
            out["jitter_px"] = rms
            out["jitter_deg"] = math.degrees(math.atan2(rms, self.fx))
            fs = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
            sig = hp[1] if np.std(hp[1]) >= np.std(hp[0]) else hp[0]
            spec = np.abs(np.fft.rfft(sig - sig.mean()))
            freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
            if len(spec) > 2:
                out["vib_hz"] = float(freqs[1:][int(np.argmax(spec[1:]))])
        why = []
        if blur < self.blur_min:
            why.append("blurred")
        if out["brightness"] < self.dark:
            why.append("too dark / lens covered")
        elif out["brightness"] > self.bright:
            why.append("over-exposed")
        if out["jitter_deg"] > self.jitter_max:
            why.append("shaking")
        out["degraded"], out["why"] = bool(why), why
        return out
