# Results summary (all numbers from simulation configured with the real car's measurements)

- Slide figure 01: **Emergency braking: gap left in front of the wall vs speed**  (`01_demo_braking.png`)
- Slide figure 02: **Roll-out after cutting the throttle: simulator vs the real car**  (`02_demo_rollout.png`)
- Slide figure 03: **Obstacle bypass: paths around a block, real-car profile**  (`03_bypass_paths.png`)
- Slide figure 04: **Follow-the-leader: gap to the leader**  (`04_demo_follow.png`)
- Slide figure 05: **Braking distance (simulator, default car)**  (`05_stopping.png`)
- Slide figure 06: **Warnings: hazards caught vs false alarms, physics vs intent-aware**  (`06_warning_curves.png`)
- Slide figure 07: **Driver-intent model**  (`07_intent_model.png`)

## Acceptance tests

- **PASS** Emergency braking at 4 speeds (also with the car 25% faster than calibrated): no contact in 8 runs, smallest gap 10.0 cm, top speed tested 1.12 m/s
- **PASS** Throttle-cut roll-out: simulator vs the real car: real car rolled 1-7 cm at 0.25-0.36 m/s; simulator 3-5 cm in that range (limit 12 cm)
- **PASS** Failsafes: controller lost, Pi dies, LiDAR unplugged, link delay: LiDAR unplugged at speed: stopped 93 cm short; watchdog stops the car; Link delay rises to 150 ms: stopped 10 cm short; Controller disconnects at full speed: stopped 93 cm short; Pi program dies (ESP32 failsafe): motor -200 -> 0 within 0.8 s of the last command
- **PASS** Intent-aware warning catches more hazards than physics-only at <= 4 false alarms/min: <= 2 false alarms/min: physics 0%, blend 0%, adaptive 44%; <= 4 false alarms/min: physics 0%, blend 0%, adaptive 57%; <= 8 false alarms/min: physics 93%, blend 75%, adaptive 67%
- **PASS** Follow-the-leader: holds a gap, and does not hit a leader that stops dead: steady gap 61 cm (+/-1.3); leader stops: closest approach 20 cm, no contact
- **PASS** Obstacle bypass: rejoins the line, or refuses a gap that is too narrow: 60/60 correct, 0 contacts, median final lateral error 0.6 cm (95th 1.6), median clearance 6.7 cm

## Other evaluations

- Auto-parking (simulator, camera markers): 32/60 correct, median lateral error 0.6 cm
- Fault injection: 9/9 passed
- Warnings: hazards caught at <= 4 false alarms/min: physics n/a, blend n/a, adaptive 57%. hazards caught at <= 8 false alarms/min: physics 93%, blend 75%, adaptive 67%.

## Not yet measured on the real car

- Stopping distance above 0.36 m/s, turn gain at the new servo centre, full-lock radius, and every simulated result above until it is repeated on the car. See STATUS.md.
