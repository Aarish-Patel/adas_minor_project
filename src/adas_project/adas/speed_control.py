"""Closed-loop speed control for the autonomous legs (TODO N1): feedforward from the fitted throttle->speed model plus a PI
correction on the measured speed (the relay's EKF / scan-matched speed).

Until now every autonomous manoeuvre (click-to-go, evasive steer, reverse creep, speed-zone caps) sent the *open-loop* throttle
`pwm_for_speed(v_target)` - right on the day the model was fitted, wrong on a sagging battery, a carpet, a hill or a slightly
different gearbox: the car ran 20 % slow with a 0.8 battery factor, so plans that assumed 0.5 m/s were driven at 0.4. The controller
is the standard automotive cruise-control structure (feedforward + PI with anti-windup, e.g. Astrom & Murray, Feedback Systems,
2008, ch. 10-11; conditional integration for anti-windup, Astrom & Hagglund 1995):

    u = pwm_ff(v*) + kp (v* - v) + I ,   dI = ki (v* - v) dt  only while u is not saturated, |I| <= i_max

Gains are conservative (the speed measurement is ~4 cm/s noisy at 10 Hz and the motor lags ~0.1 s): kp 90 PWM per m/s, ki 140.
The integrator is cleared when the target is zero or changes direction, and it never acts through the driver's throttle - only
the throttle the ADAS itself commands is corrected.
"""
import math


class SpeedController:
    def __init__(self, model, kp=90.0, ki=140.0, i_max=45.0, max_pwm=255.0):
        self.model, self.kp, self.ki, self.i_max, self.max_pwm = model, kp, ki, i_max, max_pwm
        self.i = 0.0
        self.sign = 0.0

    def reset(self):
        self.i, self.sign = 0.0, 0.0

    def pwm(self, v_target, v_measured, dt):
        """Throttle (+ forward) for a signed target speed (m/s), given the signed measured speed."""
        if abs(v_target) < 1e-3:
            self.reset()
            return 0.0
        sign = math.copysign(1.0, v_target)
        if sign != self.sign:                        # new direction: start from the feedforward alone
            self.i, self.sign = 0.0, sign
        tgt = abs(v_target)
        meas = v_measured * sign
        err = tgt - meas
        ff = self.model.pwm_for_speed(tgt)
        u_unsat = ff + self.kp * err + self.i
        u = min(self.max_pwm, max(0.0, u_unsat))
        if u == u_unsat or (u_unsat > u and err < 0) or (u_unsat < u and err > 0):     # conditional integration
            self.i = max(-self.i_max, min(self.i_max, self.i + self.ki * err * dt))
        return sign * u
