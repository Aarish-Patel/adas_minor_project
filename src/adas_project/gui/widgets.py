"""Reusable HMI widgets in the design system (gui/theme.py): KPI tile with delta, pipeline stepper, risk bar."""
import math
import time

from PySide6 import QtCore, QtGui, QtWidgets

from gui import theme
from gui.theme import C


class Kpi(QtWidgets.QFrame):
    """A key figure: small-caps title, a large light DIN numeral that tweens to its value, and a delta line
    (green / red by whether it is an improvement)."""

    def __init__(self, title, unit="", decimals=2, width=190):
        super().__init__()
        self.setObjectName("glass")
        self.setFixedHeight(104)
        self.setMinimumWidth(width)
        self.title, self.unit, self.dec = title, unit, decimals
        self.tween = theme.Tween(0.0, 9.0)
        self.sub, self.sub_colour = "", "dim"
        self.text_value = None                      # a fixed string instead of a number
        self._first = True

    def set(self, value, sub="", good=None):
        """value: number (tweened) or str; sub: the delta line; good: True/False/None for its colour."""
        if isinstance(value, str):
            self.text_value = value
        else:
            self.text_value = None
            if self._first:
                self.tween.snap(0.0)
            self.tween.set(value)
        self._first = False
        self.sub = sub
        self.sub_colour = "dim" if good is None else ("ok" if good else "bad")
        self.update()

    def tick(self, dt):
        if self.text_value is None and self.tween.step(dt):
            self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        p.setPen(theme.qcolor("dim"))
        p.setFont(theme.font(10, spacing=1.4))
        p.drawText(QtCore.QRectF(16, 12, self.width() - 32, 14), QtCore.Qt.AlignLeft, self.title.upper())
        p.setPen(theme.qcolor("text"))
        p.setFont(theme.light(40))
        txt = self.text_value if self.text_value is not None else f"{self.tween.value:.{self.dec}f}"
        p.drawText(QtCore.QRectF(16, 28, self.width() - 32, 48), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, txt)
        if self.unit:
            fm = QtGui.QFontMetricsF(theme.light(40))
            p.setFont(theme.font(13))
            p.setPen(theme.qcolor("dim"))
            p.drawText(QtCore.QPointF(20 + fm.horizontalAdvance(txt), 64), self.unit)
        p.setPen(theme.qcolor(self.sub_colour))
        p.setFont(theme.font(12))
        p.drawText(QtCore.QRectF(16, 76, self.width() - 32, 18), QtCore.Qt.AlignLeft, self.sub)


class Stepper(QtWidgets.QWidget):
    """The pipeline as a row of stages: done = brass check, active = pulsing brass ring, upcoming = grey."""

    def __init__(self, steps):
        super().__init__()
        self.steps, self.active, self.detail = steps, -1, {}
        self.setFixedHeight(66)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def set(self, active, detail=None):
        self.active, self.detail = active, detail or {}

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        n = len(self.steps)
        w = self.width()
        cell = w / n
        pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)
        for i, name in enumerate(self.steps):
            cx = cell * (i + 0.5)
            cy = 22
            if i < n - 1:
                done = i < self.active
                p.setPen(QtGui.QPen(theme.qcolor("accent") if done else theme.qcolor("hair"), 2))
                p.drawLine(QtCore.QPointF(cx + 15, cy), QtCore.QPointF(cx + cell - 15, cy))
            if i < self.active:
                p.setBrush(theme.qcolor("accent"))
                p.setPen(QtCore.Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(cx, cy), 11, 11)
                p.setPen(QtGui.QPen(theme.qcolor("bg"), 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(cx - 4.5, cy), QtCore.QPointF(cx - 1, cy + 3.5),
                                                QtCore.QPointF(cx + 5, cy - 3.5)]))
            elif i == self.active:
                halo = theme.qcolor("accent", int(40 + 60 * pulse))
                p.setBrush(halo)
                p.setPen(QtCore.Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(cx, cy), 15 + 2 * pulse, 15 + 2 * pulse)
                p.setBrush(theme.qcolor("bg1"))
                p.setPen(QtGui.QPen(theme.qcolor("accent"), 2))
                p.drawEllipse(QtCore.QPointF(cx, cy), 10, 10)
            else:
                p.setBrush(theme.qcolor("bg1"))
                p.setPen(QtGui.QPen(theme.qcolor("hair2"), 1.5))
                p.drawEllipse(QtCore.QPointF(cx, cy), 10, 10)
            p.setPen(theme.qcolor("text") if i <= self.active else theme.qcolor("dim"))
            p.setFont(theme.font(11, spacing=0.8))
            p.drawText(QtCore.QRectF(cx - cell / 2, 38, cell, 14), QtCore.Qt.AlignCenter, name.upper())
            d = self.detail.get(i)
            if d:
                p.setPen(theme.qcolor("accent"))
                p.setFont(theme.font(10))
                p.drawText(QtCore.QRectF(cx - cell / 2, 51, cell, 13), QtCore.Qt.AlignCenter, d)


class RiskBar(QtWidgets.QWidget):
    """Thin horizontal bar, teal-free: grey track, fill from white through amber to red as the value rises."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(6)
        self.v = 0.0

    def set(self, v):
        self.v = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(theme.qcolor("hair"))
        p.drawRoundedRect(QtCore.QRectF(0, 0, w, 6), 3, 3)
        col = "info" if self.v < 0.35 else "warn" if self.v < 0.65 else "bad"
        p.setBrush(theme.qcolor(col))
        p.drawRoundedRect(QtCore.QRectF(0, 0, max(6.0, w * self.v), 6), 3, 3)


def risk_rgb(r):
    """3D car colour for a crash risk 0..1: pearl white -> amber -> red (never teal/blue)."""
    r = max(0.0, min(1.0, r))
    if r < 0.5:
        k = r / 0.5
        return (0.90 + 0.05 * k, 0.91 - 0.15 * k, 0.92 - 0.45 * k, 1.0)
    k = (r - 0.5) / 0.5
    return (0.95 - 0.03 * k, 0.76 - 0.44 * k, 0.47 - 0.16 * k, 1.0)
