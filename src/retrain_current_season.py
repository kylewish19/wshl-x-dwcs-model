from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_2017_2025.csv"
PREFIGHT = ROOT / "data" / "current" / "current_season_prefight_features.csv"
RESULTS = ROOT / "data" / "current" / "current_season_results.csv"
REPORTS = ROOT / "reports"
WEEKLY = REPORTS / "weekly"
MODELS = ROOT / "models"
WEEKLY.mkdir(parents=True, exist_ok=True)

NON_FEATURE = {
    "event_date", "event_id", "bout_key", "bout_id", "week",
    "fighter_a_id", "fighter_b_id", "fighter_a", "fighter_b",
    "winner_a", "winner_name", "method", "finish_round",
    "duration_minutes", "notes"
}

# Tape-derived Week 6+ features do not exist historically and are intentionally
# filled with zero for old fights rather than retroactively invented.
TAPE_FEATURES = [
    "recovery_after_hurt_diff",
    "scramble_conversion_diff",
    "counter_grappling_transition_diff",
    "multi_route_finishing_diff",
    "early_finish_hazard_diff",
]

def make_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.35,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
        )),
    ])

def standardized_coefficients(model, features):
    return pd.DataFrame({
        "feature": features,
        "coefficient_standardized": model.named_steps["model"].coef_[0],
    }).sort_values(
        "coefficient_standardized",
        key=lambda s: s.abs(),
        ascending=False,
    )

def main():
    hist = pd.read_csv(HIST)
    prefight = pd.read_csv(PREFIGHT)
    results = pd.read_csv(RESULTS)

    for f in TAPE_FEATURES:
        if f not in hist.columns:
            hist[f] = 0.0

    if prefight.empty or results.empty:
        raise SystemExit("No completed current-season feature/result rows yet.")

    completed = prefight.merge(
        results,
        on=["bout_id", "week"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    completed = completed[completed["winner_a"].notna()].copy()
    if completed.empty:
        raise SystemExit("No completed labeled current-season rows yet.")

    historical_features = [
        c for c in hist.columns
        if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(hist[c])
    ]
    extra = [f for f in TAPE_FEATURES if f not in historical_features]
    features = historical_features + extra

    for f in features:
        if f not in hist.columns:
            hist[f] = 0.0
        if f not in completed.columns:
            completed[f] = 0.0

    train_hist = hist[features + ["winner_a"]].copy()
    train_current = completed[features + ["winner_a"]].copy()
    train = pd.concat([train_hist, train_current], ignore_index=True)

    model = make_model()
    model.fit(train[features], train["winner_a"].astype(int))
    coef = standardized_coefficients(model, features)

    last_week = int(completed["week"].max())
    snapshot = WEEKLY / f"week_{last_week}_coefficients.csv"

    previous_files = sorted(
        WEEKLY.glob("week_*_coefficients.csv"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    previous = None
    previous_label = "historical_baseline"
    previous_files = [p for p in previous_files if p != snapshot]
    if previous_files:
        previous = pd.read_csv(previous_files[-1])
        previous_label = previous_files[-1].stem
    else:
        previous = pd.read_csv(REPORTS / "winner_logistic_coefficients.csv")

    delta = coef.merge(
        previous[["feature", "coefficient_standardized"]],
        on="feature",
        how="left",
        suffixes=("_new", "_old"),
    )
    delta["delta"] = (
        delta["coefficient_standardized_new"]
        - delta["coefficient_standardized_old"].fillna(0)
    )
    delta["abs_delta"] = delta["delta"].abs()
    delta = delta.sort_values("abs_delta", ascending=False)

    coef.to_csv(snapshot, index=False)
    coef.to_csv(REPORTS / "current_winner_coefficients.csv", index=False)
    delta.to_csv(REPORTS / "current_coefficient_delta.csv", index=False)
    joblib.dump(model, MODELS / "dwcs_w_logistic_current.joblib")

    summary = {
        "model": "DWCS-W current retrain",
        "trained_through_week": last_week,
        "historical_rows": int(len(train_hist)),
        "current_labeled_rows": int(len(train_current)),
        "total_training_rows": int(len(train)),
        "previous_coefficient_reference": previous_label,
        "feature_count": len(features),
        "note": (
            "Current-season rows are only added after their prefight features "
            "were frozen and their result was later joined by bout_id."
        ),
    }
    (REPORTS / "current_retrain_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
