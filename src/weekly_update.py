from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import beta

from calibration import fit_anchor_calibrator

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = json.loads(
    (ROOT / "data" / "historical_aggregate_counts.json").read_text()
)

def refit_calibration(ledger):
    return fit_anchor_calibrator(
        ledger.locked_probability.values,
        ledger.pick_correct.values,
        lam=4.0,
    )

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

def under_1_5_label(round_number, clock):
    """Return 1 for Under 1.5, 0 for Over 1.5, None if R2 clock is unknown."""
    r = int(round_number)
    if r == 1:
        return 1
    if r >= 3:
        return 0
    if pd.isna(clock) or str(clock).strip() == "":
        return None
    mm, ss = map(int, str(clock).split(":"))
    elapsed = 300 + mm * 60 + ss
    return int(elapsed < 450)

def update_ou15_prior(ledger):
    a = HISTORICAL["inside_7m30"] + 0.5
    b = HISTORICAL["not_inside_7m30"] + 0.5

    labels = []
    for _, row in ledger.iterrows():
        label = under_1_5_label(row.actual_round, row.actual_time)
        if label is not None:
            labels.append(label)

    a += sum(labels)
    b += len(labels) - sum(labels)

    return {
        "UNDER_1_5": a / (a + b),
        "OVER_1_5": b / (a + b),
        "known_labeled_rows": len(labels),
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
