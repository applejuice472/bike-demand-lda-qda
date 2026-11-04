"""
Bike demand classification with LDA and QDA, evaluated with nested cross-validation.

Predicts whether Capital Bikeshare (Washington D.C.) needs to increase its bike
stock in a given hour (high vs low demand) from temporal and weather features.

Key points:
  * The data is imbalanced (about 82% low demand, 18% high demand), so F1 for the
    high demand class is the main metric, not accuracy.
  * Hyperparameters are tuned in an inner CV loop and performance is estimated in
    an outer CV loop, so tuning never sees the data it is evaluated on.
  * Scaling is fitted inside each training fold (via a Pipeline) to avoid leakage.
  * Naive baselines are included so the models have something honest to beat.

Usage:
    python lda_qda_nested_cv.py --data data/training_data_VT2026.csv
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "increase_stock"
CATEGORICAL = ["hour_of_day", "month", "day_of_week"]
NUMERICAL = [
    "temp", "dew", "humidity", "precip", "snow",
    "snowdepth", "windspeed", "cloudcover", "visibility",
]
SEED_OUTER, SEED_INNER = 42, 123


def load_data(path):
    """Load the CSV, encode the target as 1 = high demand, and one-hot encode categoricals."""
    df = pd.read_csv(path)
    y = df[TARGET].map({"high_bike_demand": 1, "low_bike_demand": 0})
    X = df.drop(columns=TARGET)

    # drop_first=True avoids perfectly collinear dummy columns. Without it the
    # covariance matrices become singular and QDA fails to fit.
    X = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True, dtype=float)
    return X, y


def make_pipeline(model, columns):
    """Scale only the continuous features; binary and dummy columns pass through."""
    numeric = [c for c in NUMERICAL if c in columns]
    prep = ColumnTransformer(
        [("scale", StandardScaler(), numeric)], remainder="passthrough"
    )
    return Pipeline([("prep", prep), ("model", model)])


def scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def nested_cv(name, model, param_grid, X, y):
    """Inner CV picks hyperparameters, outer CV gives an unbiased performance estimate."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_OUTER)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_INNER)

    fold_scores, chosen = [], []
    for train_idx, test_idx in outer.split(X, y):
        search = GridSearchCV(
            make_pipeline(model, X.columns), param_grid,
            cv=inner, scoring="f1", n_jobs=-1,
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = search.predict(X.iloc[test_idx])
        fold_scores.append(scores(y.iloc[test_idx], y_pred))
        chosen.append(str(search.best_params_))

    summary = {k: np.mean([s[k] for s in fold_scores]) for k in fold_scores[0]}
    std_f1 = np.std([s["f1"] for s in fold_scores])
    print(f"\n{name} (nested CV, mean over 5 outer folds)")
    print(f"  F1 {summary['f1']:.4f} ± {std_f1:.4f} | accuracy {summary['accuracy']:.4f}"
          f" | precision {summary['precision']:.4f} | recall {summary['recall']:.4f}")
    print(f"  Most chosen params: {Counter(chosen).most_common(1)[0][0]}")
    return summary


def baselines(y):
    """Trivial predictors that any useful model has to beat."""
    rng = np.random.default_rng(SEED_OUTER)
    p_high = y.mean()
    preds = {
        "Always low demand": np.zeros(len(y)),
        "Always high demand": np.ones(len(y)),
        "Random 50/50": rng.integers(0, 2, len(y)),
        "Stratified random": rng.choice([0, 1], size=len(y), p=[1 - p_high, p_high]),
    }
    return {name: scores(y, p) for name, p in preds.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", default="data/training_data_VT2026.csv")
    args = parser.parse_args()

    X, y = load_data(args.data)
    print(f"{X.shape[0]} samples, {X.shape[1]} features after encoding, "
          f"{y.mean():.0%} high demand")

    # Shrinkage only works with lsqr/eigen, and eigen needs it to handle the
    # near-singular covariance from the dummy columns.
    shrinkage = ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]
    lda_grid = [
        {"model__solver": ["svd"]},
        {"model__solver": ["lsqr"], "model__shrinkage": [None] + shrinkage},
        {"model__solver": ["eigen"], "model__shrinkage": shrinkage},
    ]
    # QDA estimates one covariance matrix per class from few minority samples,
    # so it needs regularisation to be stable at all.
    qda_grid = {"model__reg_param": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]}

    results = baselines(y)
    results["QDA (nested CV)"] = nested_cv("QDA", QuadraticDiscriminantAnalysis(), qda_grid, X, y)
    results["LDA (nested CV)"] = nested_cv("LDA", LinearDiscriminantAnalysis(), lda_grid, X, y)

    table = pd.DataFrame(results).T[["accuracy", "f1", "precision", "recall"]]
    print("\nComparison against naive baselines")
    print(table.round(4).to_string())

    best_baseline = max(v["f1"] for k, v in results.items() if "CV" not in k)
    lda_f1 = results["LDA (nested CV)"]["f1"]
    print(f"\nLDA improves F1 over the best baseline by "
          f"{(lda_f1 - best_baseline) / best_baseline:.0%}")


if __name__ == "__main__":
    main()
