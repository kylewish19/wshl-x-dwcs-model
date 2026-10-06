from pathlib import Path

import prepare_week8_ext_features as base

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"

base.MATCHUPS = CURRENT / "week9_matchups.csv"
base.BASE_FEATURES = CURRENT / "week9_model_features.csv"
base.POSTCUT = CURRENT / "week9_postcutoff_fights.csv"
base.OUT = CURRENT / "week9_secondary_model_features.csv"

if __name__ == "__main__":
    base.main()
