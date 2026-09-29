import sys, time
sys.path.insert(0, "/home/pi/rc_car")
from pi.lidar_steering_diag import Rig
from pi.test_gui import TestGui
rig = Rig(); gui = TestGui(rig)
gui.set("GUI check (no motion)", "just showing the LiDAR view", 0.0, activity="IDLE", servo=90.0)
gui.log("scan fresh: %s" % rig.is_fresh())
time.sleep(240); rig.close()
