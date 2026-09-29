"""Conformal risk control for the driver-intent warning threshold (TODO R5, P19).

The intent model outputs a risk in [0, 1] and the ADAS warns / intervenes when it crosses a threshold. Until now that
threshold was a hand-picked 0.5. Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control",
arXiv 2208.02814, ICLR 2024) picks it from calibration drives so that a *stated* risk - e.g. "at most 10 % of the drives
that end in a crash are warned less than 1 s before contact" - holds for a new drive drawn from the same distribution,
without assuming anything about the model (distribution-free).

    loss_i(lam) in [0, B], monotone non-increasing in lam-> "worse as the threshold gets stricter"
    lam_hat = inf { lam : n/(n+1) * mean_i loss_i(lam) + B/(n+1) <= alpha }
    =>  E[ loss_test(lam_hat) ] <= alpha        (exchangeable calibration and test drives)

Here loss_i(lam) is 1 when drive i is a crash that was warned late (or never) with threshold lam, else 0: a larger lam
warns later, so the loss grows with lam and the rule picks the LARGEST lam that still meets alpha (fewest false alarms).
The guarantee needs exchangeability: a shifted test set (unseen car parameters) can break it, and
sim/conformal_intent.py measures by how much.
"""
import numpy as np


def conformal_risk_threshold(losses, lambdas, alpha, B=1.0):
    """losses: array (n_drives, n_lambdas), each in [0, B], non-decreasing in lambda along axis 1;
    lambdas: increasing thresholds. Returns the largest lambda with n/(n+1) * mean_loss + B/(n+1) <= alpha, or None when even
    the smallest lambda cannot meet alpha with n drives (need n >= B/alpha - 1)."""
    L = np.asarray(losses, float)
    n = len(L)
    adj = n / (n + 1.0) * L.mean(axis=0) + B / (n + 1.0)
    ok = np.where(adj <= alpha)[0]
    if len(ok) == 0:
        return None
    return float(np.asarray(lambdas)[ok.max()])


def late_warning_loss(risk_runs, contact_idx, lambdas, dt=0.05, lead_s=1.0):
    """For each crash drive: 1 if the risk did not stay >= lambda for the last `lead_s` seconds before contact, else 0.
    risk_runs: list of risk arrays (one per drive, time-ordered, ending at the contact); returns (n, len(lambdas))."""
    need = int(round(lead_s / dt))
    out = np.zeros((len(risk_runs), len(lambdas)))
    for i, (r, c) in enumerate(zip(risk_runs, contact_idx)):
        seg = np.asarray(r)[max(0, c - need):c + 1] if c is not None else np.asarray(r)[-need - 1:]
        lo = seg.min() if len(seg) else 0.0
        # warned early enough <=> risk stayed >= lambda over the whole window <=> lambda <= min(seg); also drives that
        # start closer to contact than lead_s cannot be warned that early - they count as late
        early_enough = len(r) > need
        out[i] = np.where((np.asarray(lambdas) <= lo) & early_enough, 0.0, 1.0)
    return out
