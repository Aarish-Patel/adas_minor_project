"""Drawer controls in the layout production cockpits use (Tesla / XPeng / NIO / Huawei HarmonyOS settings): icon tiles in a
grid that light up when on, a segmented control for exclusive modes, and small-caps section headings - instead of lists of
wide buttons. Active state is a thin luminous border plus a faint tint (glass on near-black), never a flat filled button."""
from PySide6 import QtCore, QtGui, QtWidgets

from gui import theme
from gui.theme import C


class FeatureTile(QtWidgets.QFrame):
    """A setting as a tile: large icon, name and ON/OFF. Click toggles. `hint` goes to the drawer's detail strip on hover.
    `danger` lights it red (ADAS override)."""

    toggled = QtCore.Signal(bool)
    hovered = QtCore.Signal(str)

    def __init__(self, glyph, title, hint="", danger=False):
        super().__init__()
        self.on, self.danger, self.hint = False, danger, hint
        self.setObjectName("tile")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setMinimumHeight(92)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setToolTip(hint)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        self.icon = QtWidgets.QLabel(glyph)
        f = QtGui.QFont("Segoe UI Symbol")
        f.setPixelSize(26)
        self.icon.setFont(f)
        top.addWidget(self.icon)
        top.addStretch(1)
        self.state = QtWidgets.QLabel("OFF")
        self.state.setFont(theme.semibold(10, spacing=1.4))
        top.addWidget(self.state)
        lay.addLayout(top)
        lay.addStretch(1)
        self.name = QtWidgets.QLabel(title)
        self.name.setFont(theme.semibold(13))
        self.name.setWordWrap(True)
        lay.addWidget(self.name)
        self._style()

    def _style(self):
        lit = C["bad"] if self.danger else C["accent"]
        if self.on:
            tint = theme.css_rgba("bad" if self.danger else "accent", 0.10)
            self.setStyleSheet(f"QFrame#tile {{ background: {tint}; border: 1px solid {lit}; border-radius: 14px; }} "
                               f"QLabel {{ background: transparent; }}")
        else:
            self.setStyleSheet(f"QFrame#tile {{ background: {C['bg1']}; border: 1px solid {C['hair']}; border-radius: 14px; }} "
                               f"QFrame#tile:hover {{ border-color: {C['hair2']}; background: {C['surface']}; }} "
                               f"QLabel {{ background: transparent; }}")
        self.icon.setStyleSheet(f"color: {lit if self.on else C['dim']}; font-family: 'Segoe UI Symbol'; font-size: 28px;")
        self.name.setStyleSheet(f"color: {C['text'] if self.on else C['text2']};")
        self.state.setStyleSheet(f"color: {lit if self.on else C['faint']};")
        self.state.setText("ON" if self.on else "OFF")

    def setChecked(self, on):
        on = bool(on)
        if on != self.on:
            self.on = on
            self._style()

    def isChecked(self):
        return self.on

    def enterEvent(self, ev):
        self.hovered.emit(self.hint)
        super().enterEvent(ev)

    def mousePressEvent(self, ev):
        self.setChecked(not self.on)
        self.toggled.emit(self.on)


class Segmented(QtWidgets.QFrame):
    """Mutually exclusive choices (drive mode) on a pill track; the selected one is lit."""

    changed = QtCore.Signal(str)

    def __init__(self, options):
        super().__init__()
        self.setObjectName("seg")
        self.setStyleSheet(f"QFrame#seg {{ background: {C['bg1']}; border: 1px solid {C['hair']}; border-radius: 14px; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.btns = {}
        grp = QtWidgets.QButtonGroup(self)
        for key, text in options:
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setFont(theme.semibold(12, spacing=1.4))
            b.setMinimumHeight(40)
            b.setStyleSheet(f"QPushButton {{ background: transparent; border: 1px solid transparent; border-radius: 10px; "
                            f"padding: 0 4px; color: {C['dim']}; }} "
                            f"QPushButton:checked {{ background: {theme.css_rgba('accent', 0.12)}; "
                            f"border-color: {C['accent_dim']}; color: {C['text']}; }} "
                            f"QPushButton:hover:!checked {{ color: {C['text']}; }}")
            grp.addButton(b)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            lay.addWidget(b, 1)
            self.btns[key] = b

    def select(self, key):
        b = self.btns.get(key)
        if b and not b.isChecked():
            b.setChecked(True)


def section_label(text):
    """Small-caps section heading used in every drawer."""
    lb = QtWidgets.QLabel(text.upper())
    lb.setFont(theme.semibold(11, spacing=1.6))
    lb.setStyleSheet(f"color: {C['dim']}; padding: 12px 2px 2px 2px; background: transparent;")
    return lb
