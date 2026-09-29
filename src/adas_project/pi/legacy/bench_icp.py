import sys, time
sys.path.insert(0, "/home/pi/rc_car")
import numpy as np
from pi.scanmatch import icp, polar_to_xy
rng = np.random.default_rng(0)
pts = [(a, 1.5 + 0.4*np.sin(np.radians(a*3)) + rng.normal(0,0.01)) for a in np.arange(-180,180,1.3)]
A = polar_to_xy(pts); B = polar_to_xy([(a+1.0, d) for a, d in pts])
t0 = time.time()
for _ in range(10): r = icp(A, B, (0, 0, 0))
print("ICP per call: %.1f ms  rot=%.2f deg" % ((time.time()-t0)/10*1000, np.degrees(r[2])))
