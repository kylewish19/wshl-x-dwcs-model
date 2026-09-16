import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NON_FEATURE = {
    "event_date", "event_id", "fighter_a", "fighter_b", "winner_a",
    "method", "round", "time", "source", "split"
}

def feature_columns(df):
    return [
        c for c in df.columns
        if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])
    ]

def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.c_[1-p, p], labels=[0, 1])),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return out

def walk_forward_splits(df, min_train_events=20):
    """Test each event using only rows from earlier event dates."""
    d = df.sort_values(["event_date", "event_id"]).copy()
    events = d[["event_date", "event_id"]].drop_duplicates().reset_index(drop=True)
    for i in range(min_train_events, len(events)):
        test_event = events.iloc[i].event_id
        cutoff = events.iloc[i].event_date
        train = d[d.event_date < cutoff]
        test = d[d.event_id == test_event]
        if len(train) and len(test):
            yield train, test

def fit_champion(train, features):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.35,
            penalty="l2",
            solver="lbfgs",
            max_iter=4000,
        )),
    ])
    model.fit(train[features].fillna(0), train.winner_a.astype(int))
    return model

def fit_challenger(train, features):
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_depth=3,
        max_iter=250,
        l2_regularization=1.0,
        min_samples_leaf=12,
        random_state=42,
    )
    model.fit(train[features].fillna(0), train.winner_a.astype(int))
    return model

def audit(df):
    features = feature_columns(df)
    logistic_y, logistic_p = [], []
    challenger_y, challenger_p = [], []
    folds = []

    for fold, (train, test) in enumerate(walk_forward_splits(df), 1):
        champion = fit_champion(train, features)
        challenger = fit_challenger(train, features)

        p1 = champion.predict_proba(test[features].fillna(0))[:, 1]
        p2 = challenger.predict_proba(test[features].fillna(0))[:, 1]
        y = test.winner_a.astype(int).values

        logistic_y.extend(y.tolist())
        logistic_p.extend(p1.tolist())
        challenger_y.extend(y.tolist())
        challenger_p.extend(p2.tolist())

        folds.append({
            "fold": fold,
            "event_id": str(test.event_id.iloc[0]),
            "event_date": str(test.event_date.iloc[0]),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "champion_brier": metrics(y, p1)["brier"],
            "challenger_brier": metrics(y, p2)["brier"],
        })

    return pd.DataFrame(folds), {
        "features": features,
        "champion": metrics(logistic_y, logistic_p),
        "challenger": metrics(challenger_y, challenger_p),
    }

def standardized_coefficients(champion, features):
    clf = champion.named_steps["clf"]
    return pd.DataFrame({
        "feature": features,
        "coefficient_standardized": clf.coef_[0],
    }).sort_values(
        "coefficient_standardized",
        key=lambda s: s.abs(),
        ascending=False,
    )
