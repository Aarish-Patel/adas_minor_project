// Streams RPLIDAR scans in the sensor's high-density ("typical") mode, one full rotation per line on stdout:
//
//   <unix_time_s> <n> a0 d0 q0 a1 d1 q1 ...
//
// a = angle_z_q14 (degrees = a * 90 / 16384), d = dist_mm_q2 (mm = d / 4), q = quality. Raw integers keep
// the Python side cheap. The first line is "MODE <name> <us_per_sample> <max_distance_m>".
// Built against Slamtec's rplidar_sdk (see pi/lidar_stream/build.sh). Stops cleanly on SIGINT/SIGTERM.
//
//   rc_lidar_stream /dev/ttyUSB1 [baud=256000]

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#include "sl_lidar.h"
#include "sl_lidar_driver.h"

using namespace sl;

static volatile bool g_run = true;
static void on_signal(int) { g_run = false; }

static double now_s() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <serial port> [baud]\n", argv[0]);
        return 2;
    }
    const char* port = argv[1];
    sl_u32 baud = argc > 2 ? (sl_u32)atoi(argv[2]) : 256000;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, on_signal);

    ILidarDriver* drv = *createLidarDriver();
    if (!drv) { fprintf(stderr, "no driver\n"); return 1; }
    IChannel* ch = *createSerialPortChannel(port, baud);
    if (!ch || SL_IS_FAIL(drv->connect(ch))) { fprintf(stderr, "cannot connect to %s\n", port); delete drv; return 1; }

    sl_lidar_response_device_health_t health;
    if (SL_IS_OK(drv->getHealth(health)) && health.status == SL_LIDAR_STATUS_ERROR) {
        fprintf(stderr, "lidar reports an error (code %d)\n", health.error_code);
        delete drv;
        return 1;
    }

    drv->setMotorSpeed();
    LidarScanMode mode;
    if (SL_IS_FAIL(drv->startScan(false, true, 0, &mode))) {
        fprintf(stderr, "startScan failed\n");
        drv->setMotorSpeed(0);
        delete drv;
        return 1;
    }
    printf("MODE %s %.2f %.1f\n", mode.scan_mode, mode.us_per_sample, mode.max_distance);
    fflush(stdout);

    static sl_lidar_response_measurement_node_hq_t nodes[8192];
    while (g_run) {
        size_t count = sizeof(nodes) / sizeof(nodes[0]);
        sl_result r = drv->grabScanDataHq(nodes, count, 2000);
        if (SL_IS_FAIL(r)) {
            if (r == SL_RESULT_OPERATION_TIMEOUT) continue;
            fprintf(stderr, "grab failed 0x%x\n", (unsigned)r);
            break;
        }
        drv->ascendScanData(nodes, count);
        printf("%.4f %zu", now_s(), count);
        for (size_t i = 0; i < count; ++i)
            printf(" %u %u %u", nodes[i].angle_z_q14, nodes[i].dist_mm_q2, nodes[i].quality >> 2);
        printf("\n");
        if (fflush(stdout) != 0) break;       // reader went away
    }
    drv->stop();
    drv->setMotorSpeed(0);
    delete drv;
    return 0;
}
