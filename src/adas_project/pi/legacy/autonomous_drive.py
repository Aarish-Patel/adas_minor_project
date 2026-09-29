"""Drive the car around on its own, using the REAL measured turn-radius arc-sweep from
path_predict.py (not a fixed cone) to decide when to stop - "use the predicted path from
the turn radius and speed calibrations to decide stop", per the explicit request.

Layered safety, all independent of the arc prediction (defense in depth - a bug in the new
arc math can't remove the older, separately-tested checks):
  1. Rig's own hard floor (HARD_FLOOR_M, full 360, geometry-aware escape if violated)
  2. front()/rear() narrow-cone clearance (same as the calibration diagnostic used all night)
  3. NEW: arc-sweep predicted stop distance at the CURRENT steering offset, using the car's
     real measured turn radius and a speed-scaled required stopping distance

Talks to the hardware directly (stops rc-relay first, same pattern as every other pi/*.py
script tonight) - the relay's own cone-based safety gate doesn't know about steering-aware
arcs, so this needs its own full control rather than layering on top of it via UDP.

Explores by creeping forward, alternating gentle steering offsets, and reversing/
redirecting via the same geometry-aware escape logic used all night whenever it gets close,
whether that's from the hard floor, the cones, OR the new arc prediction.
"""
import json
import sys
import time

sys.path.insert(0, "/home/pi/rc_car")
from pi.lidar_steering_diag import (Rig, floor_violated,  # noqa: E402
                                     reposition_away_from_nearest, reset_forward_drift,
                                     CLEARANCE_MIN_M, HARD_FLOOR_M, TICK_S)
from pi.path_predict import predicted_stop_distance, VP  # noqa: E402

# The room tonight is consistently tight (front/rear both hover ~0.44-0.51m near the car's
# current spot, just under the diagnostic's CLEARANCE_MIN_M=0.55 used for calibration
# passes). Repositioning can't clear that bar within a reasonable attempt budget here, so
# this script accepts a looser START position - the in-flight stop checks during actual
# driving (hard floor, front cone, and the new arc prediction) stay at their full strict
# thresholds regardless of how generous or tight the starting position was.
START_CLEARANCE_MIN_M = 0.40


def ensure_start_clearance(rig, log, purpose_bearing=0):
    reposition_away_from_nearest(rig, log, purpose_bearing=purpose_bearing)
    d, _ = rig.floor()
    front_c, rear_c = rig.front(), rig.rear()
    return (d is None or d >= HARD_FLOOR_M) and \
           (front_c is None or front_c >= START_CLEARANCE_MIN_M) and \
           (rear_c is None or rear_c >= START_CLEARANCE_MIN_M)

REPORT_PATH = "/home/pi/rc_car/pi/autonomous_drive_report.json"

DRIVE_PWM = 130            # modest, constant cruising speed
STEER_PATTERN = [0, 20, 0, -20, 0, 35, 0, -35]   # gentle explore pattern, cycles
SEGMENT_MAX_S = 3.0        # cap on any single forward segment, regardless of clearance
CHECK_EVERY_S = 0.12
BASE_MARGIN_M = 0.12       # same shape as the relay's required_margin(), standalone here
REACTION_TIME_S = 0.3
ASSUMED_DECEL = 1.0
SPEED_EST_PER_PWM = 0.0018   # matches Rig's own estimate, from tonight's fitted speed_model
TOTAL_RUN_S = 90           # hard cap for this first supervised test


def required_stop_distance(speed_m_s):
    v = speed_m_s
    return BASE_MARGIN_M + v * REACTION_TIME_S + (v * v) / (2.0 * ASSUMED_DECEL)


def main():
    log = []
    rig = Rig()
    t_start = time.time()
    result = {"started": t_start, "segments": []}

    try:
        if not ensure_start_clearance(rig, log):
            result["aborted"] = "insufficient clearance at start"
            return

        pattern_i = 0
        while time.time() - t_start < TOTAL_RUN_S:
            offset = STEER_PATTERN[pattern_i % len(STEER_PATTERN)]
            pattern_i += 1

            if not ensure_start_clearance(rig, log, purpose_bearing=offset):
                log.append({"event": "give_up_no_clearance", "t": time.time() - t_start})
                break

            rig.steer(90 + offset)
            time.sleep(0.15)

            seg_start = time.time()
            stop_reason = None
            est_speed = 0.0
            while time.time() - seg_start < SEGMENT_MAX_S:
                bad, fd, fa = floor_violated(rig)
                if bad:
                    stop_reason = "hard_floor"
                    break
                front_c = rig.front()
                if front_c is not None and front_c < CLEARANCE_MIN_M:
                    stop_reason = "front_cone"
                    break

                pts = rig.points()
                dist_to_contact = predicted_stop_distance(pts, offset, +1, horizon=2.0)
                est_speed = min(est_speed + SPEED_EST_PER_PWM * TICK_S * 4, DRIVE_PWM * SPEED_EST_PER_PWM)
                needed = required_stop_distance(est_speed)
                if dist_to_contact < needed:
                    stop_reason = "predicted_path"
                    break

                rig.motor_forward(DRIVE_PWM)
                time.sleep(TICK_S)

            rig.stop()
            seg_elapsed = time.time() - seg_start
            result["segments"].append({
                "t": seg_start - t_start, "offset": offset, "duration_s": seg_elapsed,
                "stop_reason": stop_reason or "segment_time_cap",
                "front_at_stop": rig.front(), "rear_at_stop": rig.rear(),
            })
            log.append({"event": "segment_end", "offset": offset, "stop_reason": stop_reason,
                        "duration_s": seg_elapsed})

            reset_forward_drift(rig, seg_elapsed * 0.6, forward_pwm=DRIVE_PWM)

        result["finished"] = time.time()
        result["total_segments"] = len(result["segments"])
        from collections import Counter
        result["stop_reason_counts"] = dict(Counter(s["stop_reason"] for s in result["segments"]))

    finally:
        rig.stop()
        rig.steer(90)
        time.sleep(0.2)
        rig.close()
        result["log"] = log
        with open(REPORT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2, default=str))


if __name__ == "__main__":
    main()
