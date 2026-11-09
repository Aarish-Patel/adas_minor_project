"""Screenshot of the dashboard's Camera drawer (TODO AN3): start `python pi/marker_camera.py --sim` first (port 8091), then

    python tools/camera_drawer_shot.py [host] [out.png]

opens the real dashboard window, the Camera drawer, waits for the stream and saves the window (reports/camera_drawer.png by default).
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "reports", "camera_drawer.png")
    sys.argv = sys.argv[:1]
    import gui.ev as EV
    from PySide6 import QtCore, QtWidgets
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gui import theme
    from gui.controls import gui_thread_gc
    from gui.dashboard import Link
    gui_thread_gc()
    theme.load_mode()
    app.setStyleSheet(theme.STYLE)
    win = EV.EVWindow(Link(host))
    win.resize(1700, 1000)
    win.show()
    win.toggle("rear", True)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        app.processEvents()
        time.sleep(0.02)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    win.grab().save(out)
    print("saved", out, "| camera drawer facing:", win.panels["rear"][1].facing, "| status:", win.panels["rear"][1].status.text().replace("\n", " / "))
    os._exit(0)


if __name__ == "__main__":
    main()
