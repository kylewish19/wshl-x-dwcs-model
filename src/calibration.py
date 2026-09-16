import numpy as np
from scipy.optimize import minimize

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_anchor_calibrator(raw_prob, y, lam=4.0):
    """Fit p_cal = sigmoid(slope * logit(p_raw) + intercept).

    L2 penalty anchors slope=1 and intercept=0 so a small weekly DWCS
    sample cannot wildly retune the probability scale.
    """
    x = _logit(raw_prob)
    y = np.asarray(y, dtype=float)

    def loss(theta):
        slope, intercept = theta
        p = np.clip(_sigmoid(slope * x + intercept), 1e-8, 1 - 1e-8)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        penalty = lam * ((slope - 1.0) ** 2 + intercept ** 2)
        return nll + penalty

    res = minimize(loss, [1.0, 0.0], method="BFGS")
    return {
        "slope": float(res.x[0]),
        "intercept": float(res.x[1]),
        "success": bool(res.success),
    }

def calibrate(raw_prob, slope, intercept):
    return _sigmoid(slope * _logit(raw_prob) + intercept)
