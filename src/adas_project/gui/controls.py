"""Drawer controls in the layout production cockpits use (Tesla / XPeng / NIO / Huawei HarmonyOS settings): icon tiles in a
grid that light up when on, a segmented control for exclusive modes, and small-caps section headings - instead of lists of
wide buttons. Active state is a thin luminous border plus a faint tint (glass on near-black), never a flat filled button."""
from PySide6 import QtCore, QtGui, QtWidgets

from gui import theme
from gui.theme import C


_GC = []


def gui_thread_gc():
    """Collect Python garbage only on the GUI thread (pyqtgraph's GarbageCollector: automatic collection off, a 1 s timer collects).
    Automatic collection runs in whichever thread happens to allocate - a worker importing PyTorch collected widgets of a closed
    window there, and Qt crashed (access violation). Call once after the QApplication exists; later calls do nothing."""
    if not _GC:
        from pyqtgraph.util.garbage_collector import GarbageCollector
        _GC.append(GarbageCollector(interval=1.0))
    return _GC[0]


class ThreadSignal(QtCore.QObject):
    """A signal a worker thread may emit at any time. The worker emits on this small bridge, which the thread keeps alive, and Qt
    queues the call to `slot` on the GUI thread - or drops it when the receiving widget has been destroyed meanwhile (window closed,
    or rebuilt on a theme switch). Emitting on the destroyed widget's own signal crashed the process (Windows access violation)."""

    fired = QtCore.Signal(object)

    def __init__(self, slot):
        super().__init__()
        self.fired.connect(slot)

    def emit(self, value=None):
        self.fired.emit(value)


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


class Chip(QtWidgets.QPushButton):
    """A checkable pill for filters and selections in the labs (a coloured dot + a label): outlined when off, blue border and a
    faint tint when on - the compact sibling of FeatureTile for rows of options."""

    def __init__(self, text, colour=None):
        super().__init__(text)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        dot = colour or C["accent"]
        self.setText(f"\u25cf  {text}")
        self.setStyleSheet(
            f"QPushButton {{ background: {C['bg1']}; border: 1px solid {C['hair']}; border-radius: 15px; padding: 6px 14px; "
            f"color: {C['text2']}; font-size: 12px; text-transform: none; letter-spacing: 0.4px; }} "
            f"QPushButton:hover {{ border-color: {C['hair2']}; color: {C['text']}; }} "
            f"QPushButton:checked {{ background: {theme.css_rgba('accent', 0.10)}; border: 1px solid {C['accent']}; color: {C['text']}; }}")
        self._dot = dot


def style_primary(btn):
    """Main action: blue outline and text, a faint tint on hover - never a flat blue fill."""
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet(
        f"QPushButton {{ background: transparent; border: 1px solid {C['accent']}; border-radius: 12px; padding: 9px 20px; "
        f"color: {C['accent']}; font-weight: 600; letter-spacing: 0.8px; }} "
        f"QPushButton:hover {{ background: {theme.css_rgba('accent', 0.14)}; }} "
        f"QPushButton:disabled {{ border-color: {C['hair']}; color: {C['faint']}; }}")


def style_danger(btn):
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setStyleSheet(
        f"QPushButton {{ background: transparent; border: 1px solid {C['hair2']}; border-radius: 12px; padding: 9px 20px; "
        f"color: {C['text2']}; letter-spacing: 0.8px; }} "
        f"QPushButton:hover {{ border-color: {C['bad']}; color: {C['bad']}; background: {theme.css_rgba('bad', 0.10)}; }}")


class LegendPill(QtWidgets.QFrame):
    """A legend as one rounded pill: a coloured dot and a small-caps name per entry. Sits on a card or over a 3D scene, so it
    never covers the data the way a plot's own legend does. items: [(label, colour hex)]."""

    def __init__(self, items=(), parent=None, over_scene=False):
        super().__init__(parent)
        self.setObjectName("legend")
        bg = theme.css_rgba("surface", 0.92 if over_scene else 1.0)
        self.setStyleSheet(f"QFrame#legend {{ background: {bg}; border: 1px solid {C['hair']}; border-radius: 14px; }} "
                           f"QLabel {{ background: transparent; }}")
        self._lay = QtWidgets.QHBoxLayout(self)
        self._lay.setContentsMargins(12, 4, 12, 4)
        self._lay.setSpacing(5)
        self.set(items)

    def set(self, items):
        while self._lay.count():
            w = self._lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for label, colour in items:
            dot = QtWidgets.QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background: {colour}; border-radius: 5px;")
            self._lay.addWidget(dot)
            lab = QtWidgets.QLabel(label)
            lab.setFont(theme.font(12))
            lab.setStyleSheet(f"color: {C['text2']}; margin-right: 8px;")
            self._lay.addWidget(lab)
        self.setVisible(bool(items))
        self.adjustSize()


class EmptyHint(QtWidgets.QFrame):
    """Designed empty state laid over a widget (a chart with no data yet): an outlined glyph disc, a headline and one line of
    what to do. Shown / hidden with set_empty(); keeps itself centred on the parent."""

    def __init__(self, parent, glyph, title, text):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setAlignment(QtCore.Qt.AlignCenter)
        lay.setSpacing(4)
        disc = QtWidgets.QLabel(glyph)
        disc.setFixedSize(44, 44)
        disc.setAlignment(QtCore.Qt.AlignCenter)
        f = QtGui.QFont("Segoe UI Symbol")
        f.setPixelSize(20)
        disc.setFont(f)
        disc.setStyleSheet(f"color: {C['dim']}; border: 1px solid {C['hair2']}; border-radius: 22px;")
        lay.addWidget(disc, 0, QtCore.Qt.AlignHCenter)
        t = QtWidgets.QLabel(title)
        t.setFont(theme.semibold(13))
        t.setStyleSheet(f"color: {C['text2']};")
        t.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(t)
        s = QtWidgets.QLabel(text)
        s.setStyleSheet(f"color: {C['dim']}; font-size: 12px;")
        s.setAlignment(QtCore.Qt.AlignCenter)
        s.setWordWrap(True)
        s.setMaximumWidth(340)
        lay.addWidget(s, 0, QtCore.Qt.AlignHCenter)
        parent.installEventFilter(self)
        self._centre()

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self._centre()
        return False

    def _centre(self):
        p = self.parentWidget()
        self.adjustSize()
        self.move((p.width() - self.width()) // 2, (p.height() - self.height()) // 2)
        self.raise_()

    def set_empty(self, empty):
        self.setVisible(empty)
        if empty:
            self._centre()


class ChartCard(Card):
    """A chart on a card: title (type scale 13) and a legend pill above the plot, the plot itself without its own title or
    legend, so nothing overlaps the curves. `plot` is a pyqtgraph PlotWidget from gui.lab_tabs.plot()."""

    def __init__(self, title, plot, legend=()):
        super().__init__(padding=12)
        self.plot = plot
        self.title = QtWidgets.QLabel(title)
        self.title.setFont(theme.semibold(13))
        self.title.setStyleSheet(f"color: {C['text']};")
        self.title.setWordWrap(True)
        self.body.addWidget(self.title)
        self.legend = LegendPill(legend)
        self.body.addWidget(self.legend, 0, QtCore.Qt.AlignLeft)
        self.body.addWidget(plot, 1)
        plot.card = self
