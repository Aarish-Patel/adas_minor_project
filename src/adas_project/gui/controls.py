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

    clicked = QtCore.Signal()

    def __init__(self, glyph, title, hint="", danger=False, state=True, momentary=False, compact=False):
        """state=False hides ON/OFF (tool selectors); momentary=True is a push tile (an action), it never stays lit;
        compact=True is the smaller tile used for tool rows and action grids."""
        super().__init__()
        self.on, self.danger, self.hint = False, danger, hint
        self.show_state, self.momentary, self.compact = state, momentary, compact
        self.setObjectName("tile")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setMinimumHeight(78 if compact else 92)
        if compact:
            self.setMaximumHeight(78)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setToolTip(hint)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        self.icon = QtWidgets.QLabel(glyph)
        f = QtGui.QFont("Segoe UI Symbol")
        f.setPixelSize(22 if compact else 26)
        self.icon.setFont(f)
        top.addWidget(self.icon)
        top.addStretch(1)
        self.state = QtWidgets.QLabel("OFF")
        self.state.setFont(theme.semibold(10, spacing=1.4))
        top.addWidget(self.state)
        self.state.setVisible(state)
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
        self.icon.setStyleSheet(f"color: {lit if self.on else C['dim']}; font-family: 'Segoe UI Symbol'; font-size: {22 if self.compact else 28}px;")
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
        if self.momentary:
            self.clicked.emit()
            return
        self.setChecked(not self.on)
        self.toggled.emit(self.on)
        self.clicked.emit()


class TileGroup(QtCore.QObject):
    """Makes tiles mutually exclusive (tool selectors): one is always lit."""

    def __init__(self, tiles):
        super().__init__()
        self.tiles = tiles
        for t in tiles:
            t.toggled.connect(lambda on, t=t: self._changed(t, on))

    def _changed(self, tile, on):
        if on:
            for o in self.tiles:
                if o is not tile:
                    o.setChecked(False)
        elif not any(o.isChecked() for o in self.tiles):
            tile.setChecked(True)                        # the lit one cannot be switched off by clicking it again


class Card(QtWidgets.QFrame):
    """A rounded panel (surface colour, hairline border) for grouped content: status, results, the map."""

    def __init__(self, padding=14, tone="bg1"):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ background: {C[tone]}; border: 1px solid {C['hair']}; border-radius: 14px; }} "
                           f"QLabel {{ background: transparent; }}")
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(6)


class ZoneRow(QtWidgets.QFrame):
    """One speed-limit zone as a row: colour dot, shape and limit, the scaled car speed, and a delete button."""

    delete = QtCore.Signal()

    def __init__(self, colour, title, sub):
        super().__init__()
        self.setObjectName("zrow")
        self.setStyleSheet(f"QFrame#zrow {{ background: {C['bg1']}; border: 1px solid {C['hair']}; border-radius: 10px; }} "
                           f"QLabel {{ background: transparent; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 8, 8)
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet(f"color: {colour}; font-size: 16px;")
        lay.addWidget(dot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(0)
        t = QtWidgets.QLabel(title)
        t.setFont(theme.semibold(13))
        u = QtWidgets.QLabel(sub)
        u.setFont(theme.font(11))
        u.setStyleSheet(f"color: {C['dim']};")
        col.addWidget(t)
        col.addWidget(u)
        lay.addLayout(col, 1)
        x = QtWidgets.QPushButton("✕")
        x.setFixedSize(28, 28)
        x.setStyleSheet(f"QPushButton {{ background: transparent; border: none; color: {C['dim']}; padding: 0; }} "
                        f"QPushButton:hover {{ color: {C['bad']}; }}")
        x.clicked.connect(self.delete.emit)
        lay.addWidget(x)


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
