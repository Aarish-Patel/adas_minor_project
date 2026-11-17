# Whole-project pass: plan (17 Nov 2026)

Order asked for by the user: first make the working functions better (go-to, evasive steer, brake gate, parallel parking, bay
parking), then new things. Every step follows the same loop:

1. **Measure** the function on the twin's relay code (the car's own code, actuation noise on, several seeds) with a script that
   prints numbers in the user's terms.
2. **Find the cause** of the worst result from a trace of one failing run (not by guessing).
3. **Fix** it in the shared code (`adas/`, `pi/`), using a published method where one exists.
4. **Re-measure** on the same script, then run the regression (unit tests, relay scenarios, click-to-go, parking, evasive).
5. **Check on the dashboard** with screenshots, commit, publish to GitHub.
6. **Car:** deploy only with the user's OK; the user drives the test; the drive log is read back.

Nothing below is deployed to the Pi yet. Twin numbers are not car numbers.

## 1. Brake gate

| | |
|---|---|
| Goal | stop about 5 cm from an obstacle from any speed, as one smooth slow-down, never touching |
| Script | `python -m sim.stopping_eval` (27 approaches, throttle held, 9 throttles x 3 seeds) |
| Before | final gap 4.1-6.9 cm, 0 contacts, but 1.11 brake pulses and 2.48 stops per approach (brake at ~33 cm, speeds up again, brakes at ~12 cm) |
| Cause | the throttle cap was the throttle for the allowed speed *now*; the command acts a scan + a command delay later |
| Done | cap leads the limit by 0.25 s (`CAP_LEAD_S`): 0.00 pulses, 1.11 stops, gap 4.8-7.8 cm, 0 contacts |
| Next | (a) same test with a late scan (+0.1 / +0.25 s) and with 30 % command loss; (b) reversing toward a wall; (c) on the car: 3 approaches at slow / medium / full throttle, read the gap and the pulses from the drive log |

## 2. Go-to (click-to-go) and return to start

| | |
|---|---|
| Goal | arrive on the clicked point (under 3 cm), no needless stops, a path from any start pose including beside a wall |
| Script | `python -m sim.autonav_eval suite 2` (13 scenarios x 2 seeds) |
| Before | 25/26, mean arrival error 5.5 cm (it stopped as soon as it was within 8 cm) |
| Done | stops on the goal (`_close_enough`, steering held over the last 12 cm): mean error 1.8 cm; escape slack 5 mm -> 2 cm (a start beside a noisy wall had no first move on some scans) |
| Open | `corridor_ahead` seed 1 ends 14 cm off: the scan-matched pose drifts along a featureless corridor |
| Next | (a) degeneracy-aware scan matching: find the direction the scan does not constrain and use the throttle model along it (Zhang, Kaess & Singh, ICRA 2016); (b) return-to-start accuracy after a loop (script: drive a loop, HOME, error at the start); (c) starts 3-8 cm from a wall in every direction (new cases in the suite) |

## 3. Evasive steer (forward-only, as the user set on 10 Nov)

| | |
|---|---|
| Goal | drive round an obstacle when there is room, stop when there is not, never reverse, never touch |
| Script | `python -m sim.evasive_eval 2` (box 1.0-3.0 m ahead, 3 throttles, 3 lateral offsets, 2 seeds = 90 runs) |
| Now | 88/90 pass, 0 contacts, no reversing; the other 2 stop in front of a box 1 m ahead |
| Done | measured only; the optional back-off was repaired (it ended at once after the 3 cm standoff change) |
| Next | (a) add cases: box near a wall (only one side free), two boxes, a narrow gap; (b) time to pass (6-9 s now) - check whether the 0.40 m/s cap or the planning delay costs it; (c) the head-on walker scenario (the only failing relay scenario): decide the right behaviour - stop and hold - and judge the car on what it does, not on a walker that walks into a stopped car |

## 4. Parallel parking

| | |
|---|---|
| Goal | park in any gap the car fits: inside the lane, straight (under 5 deg), near the kerb, no contact, without giving up |
| Scripts | `python -m sim.parking_validation suite` (streets) and `... home 3` (walled bays like the user's room: 24 cases x 3 seeds) |
| Before | street cases: 5 of 62 fail (5-6 cm along, one 0.5 cm onto the kerb); home bays 14/72 |
| Causes found | "no safe path to that point": (1) pull-up pose 10 cm from the corner of the wall at the lane's end, (2) no first move from a start beside a noisy wall |
| Done | pull-up poses with room for the approach, escape slack: home bays 36/72 (the user's 0.6 m deep bay 21/24), 0 collisions; gate crawl limit 2.5 cm inside the gap (to be measured) |
| Open | 0.4-0.5 m deep lanes: the car ends skewed 20-50 deg in the gap and gives up after 6 re-plans |
| Next | (a) re-measure with the 2.5 cm crawl limit; (b) trace one skewed run: is it the tracker (steering lag in the S), the gate holding a stroke short, or the shuffle search? fix that; (c) centre along the gap from the measured neighbours for the 5-6 cm street misses; (d) kerb clip (`par_street01b`): kerb clearance on the last stroke; (e) dashboard check with screenshots on STREET and on a drawn lane |

## 5. Bay parking (marker + LiDAR)

| | |
|---|---|
| Goal | from a general area: find the bay, line up on its axis, drive in head-on, stop 3 cm from the back wall, straight (under 3 deg), centred |
| Scripts | `python -m sim.parking_validation suite` (front_* cases), `python tools/front_bay_validation.py`, dashboard clip |
| Now | heading from the back wall (0.4-0.9 deg in three cases); validation still judged the old 13 cm stop |
| Done | truth moved to the 3 cm stop |
| Next | (a) re-measure all front cases against the 3 cm truth; (b) the final stop: wall measured on the last leg, crawl limit 1.8 cm - check the stop is 2-4 cm at every approach angle; (c) PARK_AREA search and PARK_BAY (drawn bay): twin scenarios that start turned away / off to the side, judged on the end pose; (d) blurred, far and partly hidden marker cases; (e) on the car: measure hfov with the calibration tool (asked for separately), then 5 runs from different starts |

## 6. Cross-cutting, after the five functions

1. **Driver controller** (done, to try with the pad): a trigger counts only after it was seen released; a lost pad commands zero.
2. **ESP32 firmware** (edited, not compiled, not flashed): reset reason in the boot banner, damaged command lines ignored,
   PWM to zero around a direction change, radio off under serial control. To do: compile in the Arduino IDE, flash, bench test
   (PING, A, M, garbage line, failsafe).
3. **Localisation** (item 2a) - it also sets how well a drawn lane or bay stays in place.
4. **Dashboard:** bottom bar labels that fit (done), fitted solid walls instead of ragged quads (done), moving objects as
   figures (done) - to be reviewed on screenshots in both themes; then path ribbon fade, ground shading, the car model.
5. **Intent model:** paired Monte Carlo with the retuned gate (ADAS vs ADAS + intent, untouched seeds); retrain on the laptop
   GPU only if the benefit is gone.
6. **Demo clips** re-recorded for what changed (parallel park without the loop, bay search, pedestrian with the figure), labelled, H.264.
7. **Reports:** update the numbers that changed (stopping figure, parking results).

## 7. Order of work from here

1. Finish the running regression; commit and publish what is green (items 1-3 done, 4-5 partly).
2. Parallel parking 4a-4d.
3. Bay parking 5a-5d.
4. Go-to 2a-2c (localisation), evasive 3a-3c.
5. Section 6.
6. Car session with the user: deploy, then one function at a time in this order - brake, go-to, evasive, parallel, bay.
