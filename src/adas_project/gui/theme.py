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
# Two palettes, switchable at run time (set_mode), taken from the Huawei ADS screens the user pointed to (29 Sep): a slate
# blue-grey dark mode (not black) and a pale blue-grey light mode with white cards, one luminous blue for the path ribbon and
# active lines, Tesla-style greys. Blue is used for lines, text and glow - never as a flat button fill.
PALETTES = {
    "dark": {
        "bg": "#232D3D", "bg1": "#1B2431", "surface": "#2B3648", "raised": "#34415A", "hair": "#3A475C", "hair2": "#4B5B75",
        "text": "#F3F6FA", "text2": "#C3CCDA", "dim": "#8D99AC", "faint": "#5E6B80",
        "accent": "#3D8BFF", "accent_dim": "#2A62B8", "accent_bg": "rgba(61,139,255,28)",
        "ok": "#4CC38A", "warn": "#F0A030", "bad": "#F0533F", "info": "#DCE3EA",
        "s1": "#E8ECEF", "s2": "#3D8BFF", "s3": "#7BA7C7", "s4": "#9AA0A8", "s5": "#4CC38A", "s6": "#F0533F",
        "scene_bg": "#26324A", "grid": (150, 175, 215, 46), "obstacle": (0.66, 0.72, 0.82, 0.85), "wall": (0.55, 0.62, 0.74, 0.95),
        "hover": "#3B4964",
    },
    "light": {
        "bg": "#E9EEF4", "bg1": "#F6F8FB", "surface": "#FFFFFF", "raised": "#EEF2F7", "hair": "#D5DCE6", "hair2": "#BAC5D4",
        "text": "#1B2430", "text2": "#3C4A5E", "dim": "#6B7A90", "faint": "#9AA7B9",
        "accent": "#1E6BFF", "accent_dim": "#7FA8F0", "accent_bg": "rgba(30,107,255,28)",
        "ok": "#2FA35A", "warn": "#E67E22", "bad": "#D93A2B", "info": "#31455E",
        "s1": "#1B2430", "s2": "#1E6BFF", "s3": "#4F7FA8", "s4": "#6B7A90", "s5": "#2FA35A", "s6": "#D93A2B",
        "scene_bg": "#DDE5EF", "grid": (90, 110, 140, 60), "obstacle": (0.52, 0.58, 0.68, 0.85), "wall": (0.60, 0.66, 0.75, 0.95),
        "hover": "#E2E8F0",
    },
}
MODE = "dark"
C = dict(PALETTES["dark"])
_hooks = []


def on_change(fn):
    """Register a function to run after every palette switch (module-level colour tables refresh themselves)."""
    _hooks.append(fn)


def set_mode(mode):
    """Switch the palette in place (widgets built afterwards pick it up; the main window is rebuilt on a switch)."""
    global MODE, STYLE
    MODE = mode if mode in PALETTES else "dark"
    C.clear()
    C.update(PALETTES[MODE])
    STYLE = build_style()
    for fn in _hooks:
        fn()


def load_mode():
    """The last chosen mode (QSettings), dark by default."""
    try:
        m = QtCore.QSettings("RC-ADAS", "gui").value("theme", "dark")
    except Exception:
        m = "dark"
    set_mode(str(m))
    return MODE


def save_mode():
    try:
        QtCore.QSettings("RC-ADAS", "gui").setValue("theme", MODE)
    except Exception:
        pass


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
def build_style():
    return f"""
* {{ font-family: '{TEXT}', 'Segoe UI', sans-serif; font-size: 13px; color: {C['text']}; }}
QMainWindow, QWidget#root {{ background: {C['bg']}; }}
QLabel {{ background: transparent; }}
QFrame#glass {{ background: {css_rgba('surface', 0.86)}; border: 1px solid {C['hair']}; border-radius: 14px; }}
QFrame#top {{ background: {C['bg']}; border-bottom: 1px solid {C['hair']}; }}
QFrame#bar {{ background: {C['bg1']}; border-top: 1px solid {C['hair']}; }}
QPushButton {{ background: {C['raised']}; border: 1px solid {C['hair']}; border-radius: 9px; padding: 8px 16px;
               font-family: '{DISPLAY}'; letter-spacing: 0.4px; }}
QPushButton:hover {{ border-color: {C['hair2']}; background: {C['hover']}; }}
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
QDockWidget {{ background: {C['bg1']}; }}
QDockWidget QScrollArea, QDockWidget QScrollArea > QWidget > QWidget {{ background: {C['bg1']}; }}
QDockWidget::title {{ background: {C['bg1']}; padding: 10px 14px; border-bottom: 1px solid {C['hair']}; }}
QListWidget, QTableWidget, QPlainTextEdit, QTextEdit {{ background: {C['bg1']}; border: 1px solid {C['hair']};
                   border-radius: 8px; gridline-color: {C['hair']}; selection-background-color: {C['raised']}; }}
QTableCornerButton::section {{ background: {C['bg1']}; border: none; border-bottom: 1px solid {C['hair']}; }}
QHeaderView::section {{ background: {C['bg1']}; color: {C['dim']}; border: none; border-bottom: 1px solid {C['hair']};
                   padding: 6px; font-family: '{DISPLAY}'; letter-spacing: 0.6px; }}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{ background: {C['bg1']}; border: 1px solid {C['hair']};
                   border-radius: 7px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {C['surface']}; border: 1px solid {C['hair']}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {C['hair2']};
                        background: {C['bg1']}; }}
QCheckBox::indicator:checked {{ background: {css_rgba('accent', 0.30)}; border: 2px solid {C['accent']}; }}
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


STYLE = build_style()


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
