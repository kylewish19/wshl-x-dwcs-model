from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from build_historical_matrix import (
    NON_FEATURE,
    feature_columns,
    make_champion,
    make_challenger,
    score,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_2017_2025.csv"
CURRENT = ROOT / "data" / "current_season" / "prefight_feature_rows.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def coefficient_frame(model, features):
    clf = model.named_steps["model"]
    return pd.DataFrame({
        "feature": features,
        "coefficient_standardized": clf.coef_[0],
    })


def load_current():
    if not CURRENT.exists():
        return pd.DataFrame()
    df = pd.read_csv(CURRENT)
    required = {"event_date", "event_id", "fighter_a", "fighter_b", "winner_a"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Current-season file missing columns: {sorted(missing)}")
    return df


def main():
    hist = pd.read_csv(HIST)
    current = load_current()

    if current.empty:
        raise SystemExit(
            "No labeled current-season feature rows found. Add completed bouts to "
            "data/current_season/prefight_feature_rows.csv before retraining."
        )

    combined = pd.concat([hist, current], ignore_index=True, sort=False)
    features = feature_columns(hist)

    # Keep the production feature set stable unless a versioned schema change is intentional.
    for f in features:
        if f not in combined.columns:
            combined[f] = np.nan

    # Historical champion before current-season labels.
    before = make_champion()
    before.fit(hist[features], hist.winner_a.astype(int))
    before_coef = coefficient_frame(before, features).rename(
        columns={"coefficient_standardized": "coefficient_before"}
    )

    # Retrain using all labels available through the latest completed event.
    after = make_champion()
    challenger = make_challenger()
    after.fit(combined[features], combined.winner_a.astype(int))
    challenger.fit(combined[features], combined.winner_a.astype(int))

    after_coef = coefficient_frame(after, features).rename(
        columns={"coefficient_standardized": "coefficient_after"}
    )

    delta = before_coef.merge(after_coef, on="feature", how="outer")
    delta["delta"] = delta.coefficient_after - delta.coefficient_before
    delta["abs_delta"] = delta.delta.abs()
    delta = delta.sort_values("abs_delta", ascending=False)

    latest_week = int(current.get("week", pd.Series([0])).max()) if "week" in current else 0
    suffix = f"week{latest_week}" if latest_week else "current"
    delta.to_csv(REPORTS / f"winner_coefficient_delta_{suffix}.csv", index=False)

    joblib.dump(after, MODELS / "dwcs_winner_logistic_current.joblib")
    joblib.dump(challenger, MODELS / "dwcs_winner_hgb_current.joblib")

    summary = {
        "historical_rows": int(len(hist)),
        "current_season_rows": int(len(current)),
        "combined_rows": int(len(combined)),
        "latest_event_date": str(combined.event_date.max()),
        "feature_count": len(features),
        "largest_absolute_coefficient_changes": delta.head(10).to_dict("records"),
        "note": "These coefficient changes come from actual refitting after adding new labeled DWCS fights; they are not manual edits."
    }
    (REPORTS / f"winner_retrain_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
