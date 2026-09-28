"""Design system of the RC-ADAS HMI - one place for colour, type, motion and Qt styling.

Direction: premium EV cockpit (Polestar / Lucid / Porsche Taycan school), not a "dashboard template":
  - graphite neutrals with a slight warm cast instead of navy - screens in cars are near-black, never blue-black
  - ONE restrained accent (brass / champagne gold) for what the system is doing for you; everything else is
    white at different opacities
  - status colours are automotive-standard and reserved: green = ready/ok, amber = caution, red = act now
  - type: Bahnschrift (DIN 1451 style, the face of German instrument clusters) for numerals and labels,
    Segoe UI Variable for running text; tabular numerals so digits never jitter
  - motion: 120-220 ms ease-out on state changes, values tween instead of jumping, one slow pulse on the ring
"""
from PySide6 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------- colour
C = {
    # surfaces (graphite, warm)
    "bg": "#0C0D0F", "bg1": "#111316", "surface": "#16181C", "raised": "#1D2025", "hair": "#2A2D33",
    "hair2": "#3A3E46",
    # content
    "text": "#F1F3F5", "text2": "#B4BAC3", "dim": "#7B818B", "faint": "#4B5058",
    # brand accent: brass
    "accent": "#D9B45A", "accent_dim": "#8C7439", "accent_bg": "rgba(217,180,90,28)",
    # status (automotive convention)
    "ok": "#43C97F", "warn": "#F2A93B", "bad": "#EF5350", "info": "#DCE3EA",
    # data series on dark
    "s1": "#E8ECEF", "s2": "#D9B45A", "s3": "#7BA7C7", "s4": "#9AA0A8", "s5": "#43C97F", "s6": "#EF5350",
}
# the same in 0-1 floats for OpenGL
def rgb(name, a=1.0):
    h = C[name].lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def qcolor(name, alpha=255):
    q = QtGui.QColor(C[name])
    q.setAlpha(alpha)
    return q


def css_rgba(name, a):
    h = C[name].lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ---------------------------------------------------------------- type
DISPLAY = "Bahnschrift"          # numerals, labels, headings
TEXT = "Segoe UI Variable Text"  # running text, fall back to Segoe UI / system


def font(size, weight=QtGui.QFont.Normal, family=DISPLAY, tabular=True, spacing=0.0):
    f = QtGui.QFont(family, -1)
    f.setPixelSize(size)
    f.setWeight(weight)
    if tabular:
        f.setFeature(QtGui.QFont.Tag("tnum"), 1)
    if spacing:
        f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, spacing)
    f.setHintingPreference(QtGui.QFont.PreferFullHinting)
    return f


def light(size, **kw):
    return font(size, QtGui.QFont.Light, **kw)


def semibold(size, **kw):
    return font(size, QtGui.QFont.DemiBold, **kw)


# ---------------------------------------------------------------- motion
class Tween(QtCore.QObject):
    """A value that eases toward its target (critically damped-ish exponential), stepped by one shared clock.
    tween.value is what to draw; set(target) whenever the data changes. Keeps needles and numerals fluid at the
    ~30 Hz the data arrives at."""

    def __init__(self, value=0.0, rate=14.0):
        super().__init__()
        self.value, self.target, self.rate = float(value), float(value), rate

    def set(self, target):
        self.target = float(target)

    def snap(self, v):
        self.value = self.target = float(v)

    def step(self, dt):
        k = 1.0 - pow(2.718281828, -self.rate * dt)
        self.value += (self.target - self.value) * k
        return abs(self.target - self.value) > 1e-3


class Fade(QtCore.QObject):
    """0..1 opacity that fades in / out over ~180 ms; for banners and pills."""

    def __init__(self, seconds=0.18):
        super().__init__()
        self.a, self.on, self.sec = 0.0, False, seconds

    def step(self, dt):
        d = dt / self.sec
        self.a = min(1.0, self.a + d) if self.on else max(0.0, self.a - d)
        return 0.0 < self.a < 1.0


# ---------------------------------------------------------------- Qt stylesheet
STYLE = f"""
* {{ font-family: '{TEXT}', 'Segoe UI', sans-serif; font-size: 13px; color: {C['text']}; }}
QMainWindow, QWidget#root {{ background: {C['bg']}; }}
QLabel {{ background: transparent; }}
QFrame#glass {{ background: {css_rgba('surface', 0.86)}; border: 1px solid {C['hair']}; border-radius: 14px; }}
QFrame#top {{ background: {C['bg']}; border-bottom: 1px solid {C['hair']}; }}
QFrame#bar {{ background: {C['bg1']}; border-top: 1px solid {C['hair']}; }}
QPushButton {{ background: {C['raised']}; border: 1px solid {C['hair']}; border-radius: 9px; padding: 8px 16px;
               font-family: '{DISPLAY}'; letter-spacing: 0.4px; }}
QPushButton:hover {{ border-color: {C['hair2']}; background: #23272D; }}
QPushButton:pressed {{ background: {C['bg1']}; }}
QPushButton:checked {{ background: {css_rgba('accent', 0.14)}; border-color: {C['accent_dim']}; color: {C['accent']}; }}
QPushButton:disabled {{ color: {C['faint']}; }}
QPushButton#app {{ background: transparent; border: none; border-radius: 9px; padding: 8px 14px; color: {C['dim']};
                   font-size: 12px; letter-spacing: 0.8px; text-transform: uppercase; }}
QPushButton#app:hover {{ color: {C['text']}; background: {C['raised']}; }}
QPushButton#app:checked {{ color: {C['accent']}; background: transparent; border-bottom: 2px solid {C['accent']};
                           border-radius: 0; }}
QPushButton#danger {{ border-color: {C['bad']}; color: {C['bad']}; font-weight: 600; }}
QDockWidget {{ color: {C['text']}; font-family: '{DISPLAY}'; }}
QDockWidget::title {{ background: {C['bg1']}; padding: 10px 14px; border-bottom: 1px solid {C['hair']}; }}
QListWidget, QTableWidget, QPlainTextEdit, QTextEdit {{ background: {C['bg1']}; border: 1px solid {C['hair']};
                   border-radius: 8px; gridline-color: {C['hair']}; selection-background-color: {C['raised']}; }}
QHeaderView::section {{ background: {C['bg1']}; color: {C['dim']}; border: none; border-bottom: 1px solid {C['hair']};
                   padding: 6px; font-family: '{DISPLAY}'; letter-spacing: 0.6px; }}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{ background: {C['bg1']}; border: 1px solid {C['hair']};
                   border-radius: 7px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {C['surface']}; border: 1px solid {C['hair']}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {C['hair2']};
                        background: {C['bg1']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QProgressBar {{ background: {C['bg1']}; border: none; border-radius: 3px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}
QTabWidget::pane {{ border: 1px solid {C['hair']}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 8px 16px; color: {C['dim']}; border-bottom: 2px solid transparent;
                font-family: '{DISPLAY}'; letter-spacing: 0.6px; }}
QTabBar::tab:selected {{ color: {C['text']}; border-bottom: 2px solid {C['accent']}; }}
QSplitter::handle {{ background: {C['hair']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C['hair2']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QStatusBar {{ background: {C['bg1']}; color: {C['dim']}; }}
QToolTip {{ background: {C['raised']}; color: {C['text']}; border: 1px solid {C['hair2']}; padding: 6px; }}
"""


def chip_css(colour_name, filled=False):
    """Status pill: outlined, tinted on state change, uppercase display face."""
    c = C[colour_name]
    bg = css_rgba(colour_name, 0.16 if filled else 0.10)
    return (f"background: {bg}; border: 1px solid {css_rgba(colour_name, 0.55)}; color: {c}; border-radius: 11px; "
            f"padding: 3px 12px; font-family: '{DISPLAY}'; font-size: 11px; letter-spacing: 1.2px; font-weight: 600;")


def style_plot(pw, title=None):
    """A pyqtgraph plot in the design system (quiet grid, display-face labels)."""
    import pyqtgraph as pg
    pw.setBackground(C["bg1"])
    pi = pw.getPlotItem() if hasattr(pw, "getPlotItem") else pw
    for ax in ("left", "bottom"):
        a = pi.getAxis(ax)
        a.setPen(pg.mkPen(C["hair2"]))
        a.setTextPen(pg.mkPen(C["dim"]))
        a.setStyle(tickFont=font(11))
    pi.showGrid(x=True, y=True, alpha=0.10)
    if title is not None:
        pi.setTitle(title, color=C["text2"], size="10pt")
    return pw
