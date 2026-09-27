#!/bin/sh
# Builds rc_lidar_stream against Slamtec's official SDK (cloned and built once into ~/rplidar_sdk).
set -e
SDK="${RPLIDAR_SDK:-$HOME/rplidar_sdk}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SDK/output/Linux/Release/libsl_lidar_sdk.a" ]; then
    [ -d "$SDK" ] || git clone --depth 1 https://github.com/Slamtec/rplidar_sdk "$SDK"
    make -C "$SDK" -j4
fi
g++ -O2 -std=c++11 -I"$SDK/sdk/include" -I"$SDK/sdk/src" "$HERE/rc_lidar_stream.cpp" \
    "$SDK/output/Linux/Release/libsl_lidar_sdk.a" -lpthread -o "$HERE/rc_lidar_stream"
echo "built $HERE/rc_lidar_stream"
