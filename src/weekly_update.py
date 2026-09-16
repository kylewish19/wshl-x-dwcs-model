from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import beta

from calibration import fit_anchor_calibrator

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = {
    "bouts": 356,
    "ko_tko": 151,
    "submission": 67,
    "decision_or_other": 138,
    "inside_7m30": 146,
    "not_inside_7m30": 210,
}

def refit_calibration(ledger):
    fit = fit_anchor_calibrator(
        ledger.locked_probability.values,
        ledger.pick_correct.values,
        lam=4.0,
    )
    return fit

def update_method_prior(ledger):
    classes = ["KO/TKO", "SUB", "DEC"]
    alpha = np.array([
        HISTORICAL["ko_tko"],
        HISTORICAL["submission"],
        HISTORICAL["decision_or_other"],
    ], dtype=float) + 0.5

    for i, method in enumerate(classes):
        alpha[i] += int((ledger.actual_method == method).sum())

    p = alpha / alpha.sum()
    return dict(zip(classes, p.tolist()))

def elapsed_seconds(round_number, clock):
    if pd.isna(clock) or str(clock).strip() == "":
        return None
    mm, ss = map(int, str(clock).split(":"))
    return (int(round_number) - 1) * 300 + mm * 60 + ss

def update_ou15_prior(ledger):
    a = HISTORICAL["inside_7m30"] + 0.5
    b = HISTORICAL["not_inside_7m30"] + 0.5

    known = []
    for _, row in ledger.iterrows():
        sec = elapsed_seconds(row.actual_round, row.actual_time)
        if sec is not None:
            known.append(int(sec < 450))

    a += sum(known)
    b += len(known) - sum(known)

    return {
        "UNDER_1_5": a / (a + b),
        "OVER_1_5": b / (a + b),
        "under_95_ci": [
            float(beta.ppf(0.025, a, b)),
            float(beta.ppf(0.975, a, b)),
        ],
    }

if __name__ == "__main__":
    ledger = pd.read_csv(ROOT / "data" / "dwcs_weeks2_6_training_ledger.csv")
    print("Calibration:", refit_calibration(ledger))
    print("Method prior:", update_method_prior(ledger))
    print("O/U 1.5 prior:", update_ou15_prior(ledger))
