import glob
from rplidar import RPLidar

for p in sorted(glob.glob("/dev/ttyUSB*")):
    try:
        lidar = RPLidar(p, baudrate=256000, timeout=2)
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        print(f"{p}: stopped")
    except Exception as e:
        print(f"{p}: {e}")
