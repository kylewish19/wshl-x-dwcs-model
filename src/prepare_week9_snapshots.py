from pathlib import Path
import sys

import prepare_week8_snapshots as base

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
REPORTS = ROOT / "reports"

base.MATCHUPS = CURRENT / "week9_matchups.csv"
base.RECORDS = CURRENT / "week9_current_records.csv"
base.POSTCUT = CURRENT / "week9_postcutoff_fights.csv"
base.OUT = CURRENT / "week9_fighter_snapshots.csv"
base.OUT_REPORT = REPORTS / "week9_snapshot_coverage.json"

if __name__ == "__main__":
    base.main()
