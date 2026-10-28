"""Train and evaluate baseline fraud classifiers on the saved Phase 2C split.

Models:
  - DummyClassifier(strategy="most_frequent"): always predicts legitimate (Class 0).
    A reference point that shows what "doing nothing" scores.
  - Logistic Regression in a Pipeline (StandardScaler -> LogisticRegression), trained
    on the natural (imbalanced) training distribution, default 0.5 threshold.

The split is NOT regenerated: train/test rows are rebuilt from the original CSV using
data/processed/split_assignments.csv, and the run stops if the split fails verification.
The test set is used only for final evaluation (no tuning, no resampling).

Usage:
    python scripts/train_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from inspect_dataset import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    TARGET,
    DatasetValidationError,
    load_dataset,
    sha256_of,
    validate_dataset,
)

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "baseline"
RANDOM_STATE = 42
POSITIVE_CLASS = 1  # fraud
THRESHOLD = 0.5


def load_split(df: pd.DataFrame, data_path: Path, split_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Rebuild train/test from the saved assignments, verifying them first."""
    assignments_path = split_dir / "split_assignments.csv"
    metadata_path = split_dir / "split_metadata.json"
    for p in (assignments_path, metadata_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found. Run `python scripts/prepare_data.py` first.")

    metadata = json.loads(metadata_path.read_text())
    assignments = pd.read_csv(assignments_path)

    if sha256_of(data_path) != metadata["source_sha256"]:
        raise DatasetValidationError("Raw CSV does not match the file the split was created from (SHA-256 differs).")
    features = metadata["feature_columns"]
    if TARGET in features:
        raise DatasetValidationError(f"Target '{TARGET}' is listed as an input feature.")
    if features != [c for c in df.columns if c != TARGET]:
        raise DatasetValidationError("Saved feature columns do not match the CSV columns.")
    # row_id is the 0-based row position in the raw CSV; every row must be assigned exactly once.
    if not (len(assignments) == len(df) and assignments["row_id"].is_unique
            and assignments["row_id"].between(0, len(df) - 1).all()):
        raise DatasetValidationError("Split assignments do not cover every source row exactly once.")
    if not set(assignments["split"]) <= {"train", "test"}:
        raise DatasetValidationError(f"Unexpected split labels: {sorted(set(assignments['split']))}")

    split = assignments.set_index("row_id")["split"].reindex(df.index)
    is_train = split == "train"
    X, y = df[features], df[TARGET]
    X_train, y_train, X_test, y_test = X[is_train], y[is_train], X[~is_train], y[~is_train]

    overlap = X_train.drop_duplicates().merge(X_test.drop_duplicates(), how="inner")
    if len(overlap):
        raise DatasetValidationError(f"{len(overlap)} identical feature rows appear in both train and test.")
    for name, expected_y in (("train", y_train), ("test", y_test)):
        meta = metadata["splits"][name]
        if (len(expected_y), int(expected_y.sum())) != (meta["rows"], meta["fraud"]):
            raise DatasetValidationError(f"{name} row/fraud counts differ from split metadata.")

    return X_train, y_train, X_test, y_test, features


def build_models() -> dict[str, object]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            # Scaler is part of the pipeline so its mean/std are learned from training data only.
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }


def fraud_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(Class = 1), locating the column by label rather than assuming position."""
    classes = list(model.classes_)
    if POSITIVE_CLASS not in classes:
        raise ValueError(f"Model was not trained with positive class {POSITIVE_CLASS}; classes={classes}")
    return model.predict_proba(X)[:, classes.index(POSITIVE_CLASS)]


def evaluate(y_true: pd.Series, proba: np.ndarray) -> dict:
    y_pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # zero_division=np.nan: report "undefined" instead of silently substituting 0.
    precision = precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=np.nan)
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=np.nan)
    return {
        "threshold": THRESHOLD,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": None if np.isnan(precision) else float(precision),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS)),
        "f1": None if np.isnan(f1) else float(f1),
        "average_precision": float(average_precision_score(y_true, proba, pos_label=POSITIVE_CLASS)),
        "n_distinct_scores": int(np.unique(proba).size),
        "accuracy_secondary": float(accuracy_score(y_true, y_pred)),
    }


def fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "undefined"
    return f"{value:.2%}" if pct else f"{value:.4f}"


def format_report(results: dict, y_test: pd.Series, notes: list[str]) -> str:
    lines = [
        "BASELINE EVALUATION - untouched test set",
        "=" * 41,
        f"Test rows: {len(y_test):,} | fraud: {int(y_test.sum())} ({y_test.mean():.4%}) | threshold: {THRESHOLD}",
        f"PR-AUC of a random/constant scorer = fraud prevalence = {y_test.mean():.4f}",
        "",
    ]
    for name, m in results.items():
        cm = m["confusion_matrix"]
        lines += [
            f"[{name}]",
            "  Confusion matrix (rows = actual, cols = predicted)",
            f"                  pred legit   pred fraud",
            f"    actual legit  {cm['tn']:>10,}   {cm['fp']:>10,}   <- FP: legit customers flagged",
            f"    actual fraud  {cm['fn']:>10,}   {cm['tp']:>10,}   <- FN: fraud missed",
            f"  Precision:          {fmt(m['precision'])}",
            f"  Recall:             {fmt(m['recall'])}",
            f"  F1:                 {fmt(m['f1'])}",
            f"  Average precision:  {fmt(m['average_precision'])}  (distinct scores: {m['n_distinct_scores']})",
            f"  Accuracy (2nd):     {fmt(m['accuracy_secondary'], pct=True)}",
            "",
        ]
    lines += ["Notes", *[f"  - {n}" for n in notes]]
    return "\n".join(lines)


def plot_pr_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series, results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        PrecisionRecallDisplay.from_predictions(
            y_test, fraud_probability(model, X_test), pos_label=POSITIVE_CLASS, ax=ax, name=name,
        )
    lr = results["logistic_regression"]
    ax.scatter(lr["recall"], lr["precision"], color="black", zorder=3, label=f"LR @ threshold {THRESHOLD}")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        df = load_dataset(args.data_path)
        validate_dataset(df)
        X_train, y_train, X_test, y_test, features = load_split(df, args.data_path, args.split_dir)
    except (FileNotFoundError, DatasetValidationError, KeyError) as exc:
        print(f"ERROR: split verification failed, not training: {exc}", file=sys.stderr)
        return 1

    models = build_models()
    notes: list[str] = []
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            notes.append(f"{name}: solver did not fully converge (ConvergenceWarning).")

    results = {name: evaluate(y_test, fraud_probability(model, X_test)) for name, model in models.items()}
    if results["dummy_most_frequent"]["precision"] is None:
        notes.append("dummy_most_frequent: predicts no fraud, so precision = TP/(TP+FP) = 0/0 is undefined. "
                     "F1 = 2TP/(2TP+FP+FN) = 0/99 is still defined and equals 0.")
    if results["dummy_most_frequent"]["n_distinct_scores"] == 1:
        notes.append("dummy_most_frequent: gives every row the same score, so its PR-AUC only equals "
                     "the fraud prevalence; it cannot rank transactions at all.")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(results, y_test, notes)
    (args.report_dir / "evaluation.txt").write_text(report + "\n")
    (args.report_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_pr_curves(models, X_test, y_test, results, args.report_dir / "pr_curve.png")

    joblib.dump(
        {
            "model": models["logistic_regression"],
            "feature_columns": features,
            "positive_class": POSITIVE_CLASS,
            "threshold": THRESHOLD,
            "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
            "test_metrics": results["logistic_regression"],
        },
        args.model_dir / "baseline_logreg.joblib",
    )

    print(report)
    print(f"\nModel saved to:   {args.model_dir / 'baseline_logreg.joblib'}")
    print(f"Reports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
