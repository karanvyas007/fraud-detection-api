"""Select a provisional decision threshold using training data only.

Out-of-fold (OOF) fraud probabilities are produced for every training row with
group-aware StratifiedGroupKFold on the Phase 2C duplicate groups, so identical
feature rows always fall in the same fold. In each fold the full
StandardScaler -> LogisticRegression pipeline is cloned and fitted on the other
folds only; the held-out fold is only ever scored.

Test labels are never read: test rows are dropped right after verifying that the
split assignments cover the CSV, and only test *features* are used for the
train/test overlap check. The Phase 3A model and reports are not touched.

Provisional selection criterion (no business costs are defined yet):
    the threshold that maximises F1 on OOF probabilities, searched on a 0.01 grid.
F1 weights precision and recall equally; it is a neutral placeholder, not "optimal".

Usage:
    python scripts/tune_threshold.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from inspect_dataset import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    TARGET,
    DatasetValidationError,
    load_dataset,
    sha256_of,
    validate_dataset,
)
from train_baseline import DEFAULT_SPLIT_DIR, POSITIVE_CLASS, RANDOM_STATE, build_models, fraud_probability

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "threshold_tuning"
N_FOLDS = 5
BASELINE_THRESHOLD = 0.5
SEARCH_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
REPORT_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_training_rows(df: pd.DataFrame, data_path: Path, split_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return X_train, y_train, groups_train. Test labels are never extracted."""
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
    if TARGET in features or features != [c for c in df.columns if c != TARGET]:
        raise DatasetValidationError("Saved feature columns are inconsistent with the CSV or include the target.")
    if not (len(assignments) == len(df) and assignments["row_id"].is_unique
            and assignments["row_id"].between(0, len(df) - 1).all()):
        raise DatasetValidationError("Split assignments do not cover every source row exactly once.")

    assignments = assignments.set_index("row_id").reindex(df.index)
    is_train = assignments["split"] == "train"
    X_train, X_test_features = df.loc[is_train, features], df.loc[~is_train, features]
    overlap = X_train.drop_duplicates().merge(X_test_features.drop_duplicates(), how="inner")
    if len(overlap):
        raise DatasetValidationError(f"{len(overlap)} identical feature rows appear in both train and test.")

    y_train = df.loc[is_train, TARGET]
    if (len(y_train), int(y_train.sum())) != (metadata["splits"]["train"]["rows"], metadata["splits"]["train"]["fraud"]):
        raise DatasetValidationError("Training row/fraud counts differ from split metadata.")
    return X_train, y_train, assignments.loc[is_train, "group_id"], features


def make_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X, y, groups))


def verify_folds(folds: list[tuple[np.ndarray, np.ndarray]], y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Each row validated exactly once, no group split across fit/validation, fraud in every fold."""
    n = len(y)
    times_validated = np.zeros(n, dtype=int)
    rows = []
    g = groups.to_numpy()
    for i, (fit_idx, val_idx) in enumerate(folds):
        if np.intersect1d(fit_idx, val_idx).size or fit_idx.size + val_idx.size != n:
            raise AssertionError(f"Fold {i}: fit and validation indices overlap or do not cover all rows.")
        if np.intersect1d(g[fit_idx], g[val_idx]).size:
            raise AssertionError(f"Fold {i}: a duplicate group appears in both fit and validation rows.")
        y_val = y.iloc[val_idx]
        if y_val.sum() == 0:
            raise AssertionError(f"Fold {i}: validation fold has no fraud cases.")
        times_validated[val_idx] += 1
        rows.append({"fold": i, "val_rows": len(val_idx), "val_fraud": int(y_val.sum()),
                     "val_fraud_%": round(y_val.mean() * 100, 4)})
    if not (times_validated == 1).all():
        raise AssertionError("Every training row must be in exactly one validation fold.")
    return pd.DataFrame(rows).set_index("fold")


def out_of_fold_probabilities(X: pd.DataFrame, y: pd.Series, folds) -> tuple[np.ndarray, np.ndarray]:
    template = build_models()["logistic_regression"]
    oof = np.full(len(y), np.nan)
    fold_of_row = np.full(len(y), -1)
    for i, (fit_idx, val_idx) in enumerate(folds):
        model = clone(template)  # fresh, unfitted scaler + model for every fold
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        proba = fraud_probability(model, X.iloc[val_idx])
        # Positive-class check: thresholding P(fraud) at 0.5 must reproduce the model's own predictions.
        if not np.array_equal((proba >= 0.5).astype(int), model.predict(X.iloc[val_idx])):
            raise AssertionError(f"Fold {i}: extracted probabilities do not correspond to Class {POSITIVE_CLASS}.")
        oof[val_idx] = proba
        fold_of_row[val_idx] = i
    if np.isnan(oof).any() or not ((oof >= 0) & (oof <= 1)).all():
        raise AssertionError("OOF probabilities are missing or outside [0, 1].")
    return oof, fold_of_row


def threshold_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "flagged": int(tp + fp)}


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Max F1; ties broken toward the higher threshold (fewer customers flagged)."""
    best = table["f1"].max()
    return table[np.isclose(table["f1"], best)].sort_values("threshold").iloc[-1]


def per_fold_best(y: np.ndarray, oof: np.ndarray, fold_of_row: np.ndarray) -> pd.DataFrame:
    """Stability check: how much does the max-F1 threshold move between folds?"""
    rows = []
    for i in np.unique(fold_of_row):
        m = fold_of_row == i
        table = pd.DataFrame([threshold_metrics(y[m], oof[m], t) for t in SEARCH_GRID])
        best = select_threshold(table)
        rows.append({"fold": int(i), "best_threshold": best["threshold"], "f1": best["f1"],
                     "average_precision": average_precision_score(y[m], oof[m])})
    return pd.DataFrame(rows).set_index("fold")


def plot_thresholds(table: pd.DataFrame, selected: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for col, color in [("precision", "#4C72B0"), ("recall", "#C44E52"), ("f1", "#55A868")]:
        ax.plot(table["threshold"], table[col], label=col, color=color)
    ax.axvline(BASELINE_THRESHOLD, color="grey", ls="--", label=f"baseline {BASELINE_THRESHOLD}")
    ax.axvline(selected, color="black", ls=":", label=f"provisional {selected:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("OOF precision / recall / F1 vs threshold (training data)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(table["recall"], table["precision"], color="#4C72B0")
    for t, marker in [(BASELINE_THRESHOLD, "o"), (selected, "s")]:
        row = table.loc[np.isclose(table["threshold"], t)].iloc[0]
        ax.scatter(row["recall"], row["precision"], marker=marker, color="black", zorder=3,
                   label=f"t={t:.2f}: P={row['precision']:.2f}, R={row['recall']:.2f}, FP={int(row['fp'])}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OOF precision-recall trade-off (0.01-0.99 grid)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_report(fold_summary, oof_ap, table, selected, baseline, stability, oof_hash) -> str:
    def row(r):
        p = "undefined" if np.isnan(r["precision"]) else f"{r['precision']:.3f}"
        return (f"  {r['threshold']:>5.2f}  {p:>9}  {r['recall']:>6.3f}  {r['f1']:>6.3f}  "
                f"{int(r['tp']):>4}  {int(r['fp']):>5}  {int(r['fn']):>4}  {int(r['flagged']):>7}")

    header = "  thresh  precision  recall      f1    TP     FP    FN  flagged"
    report_rows = table[table["threshold"].isin(REPORT_THRESHOLDS) | np.isclose(table["threshold"], selected["threshold"])]
    lines = [
        "THRESHOLD ANALYSIS - training data only (out-of-fold probabilities)",
        "=" * 66,
        f"Model: StandardScaler -> LogisticRegression, refit in each of {N_FOLDS} StratifiedGroupKFold folds "
        f"(random_state={RANDOM_STATE})",
        f"Training rows: {int(fold_summary['val_rows'].sum()):,} | fraud: {int(fold_summary['val_fraud'].sum())}",
        "Test labels were not read.",
        "",
        "Validation folds (each training row validated exactly once; no duplicate group split)",
        fold_summary.to_string(),
        "",
        f"OOF average precision (PR-AUC): {oof_ap:.4f}",
        f"OOF probabilities SHA-256 (two runs identical): {oof_hash}",
        "",
        "Candidate thresholds",
        header,
        *[row(r) for _, r in report_rows.iterrows()],
        "",
        "Selection criterion: maximise OOF F1 on a 0.01 grid (ties -> higher threshold).",
        "F1 treats a missed fraud and a false alarm as equally costly. That is a placeholder",
        "objective, not a business decision; with real costs the threshold should be re-derived.",
        "",
        "Baseline vs provisional",
        header,
        row(baseline),
        row(selected),
        f"  Change: TP {int(selected['tp'] - baseline['tp']):+d}, FP {int(selected['fp'] - baseline['fp']):+d}, "
        f"FN {int(selected['fn'] - baseline['fn']):+d}",
        "",
        "Stability: max-F1 threshold chosen within each fold separately",
        stability.to_string(float_format=lambda x: f"{x:.3f}"),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        df = load_dataset(args.data_path)
        validate_dataset(df)
        X, y, groups, _ = load_training_rows(df, args.data_path, args.split_dir)
    except (FileNotFoundError, DatasetValidationError, KeyError) as exc:
        print(f"ERROR: split verification failed, not tuning: {exc}", file=sys.stderr)
        return 1

    folds = make_folds(X, y, groups)
    fold_summary = verify_folds(folds, y, groups)
    oof, fold_of_row = out_of_fold_probabilities(X, y, folds)

    # Reproducibility: rebuilding folds and refitting must give bit-identical OOF probabilities.
    oof_repeat, _ = out_of_fold_probabilities(X, y, make_folds(X, y, groups))
    if not np.array_equal(oof, oof_repeat):
        raise AssertionError("OOF probabilities differ between two runs with the same random_state.")
    oof_hash = hashlib.sha256(oof.tobytes()).hexdigest()

    y_arr = y.to_numpy()
    table = pd.DataFrame([threshold_metrics(y_arr, oof, t) for t in SEARCH_GRID])
    selected = select_threshold(table)
    baseline = table.loc[np.isclose(table["threshold"], BASELINE_THRESHOLD)].iloc[0]
    stability = per_fold_best(y_arr, oof, fold_of_row)
    oof_ap = average_precision_score(y_arr, oof, pos_label=POSITIVE_CLASS)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(fold_summary, oof_ap, table, selected, baseline, stability, oof_hash)
    (args.report_dir / "threshold_report.txt").write_text(report + "\n")
    table.to_csv(args.report_dir / "threshold_table.csv", index=False, float_format="%.6f")
    pd.DataFrame({"row_id": X.index, "group_id": groups.to_numpy(), "fold": fold_of_row,
                  TARGET: y_arr, "oof_fraud_proba": oof}).to_csv(args.report_dir / "oof_predictions.csv", index=False)
    plot_thresholds(table, float(selected["threshold"]), args.report_dir / "threshold_plot.png")
    (args.report_dir / "selected_threshold.json").write_text(json.dumps({
        "provisional_threshold": float(selected["threshold"]),
        "baseline_threshold": BASELINE_THRESHOLD,
        "criterion": "max out-of-fold F1 on training data, 0.01 grid, ties -> higher threshold",
        "cv": f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE}) on duplicate groups",
        "oof_average_precision": oof_ap,
        "oof_metrics_at_provisional": {k: float(v) for k, v in selected.items()},
        "oof_metrics_at_baseline": {k: float(v) for k, v in baseline.items()},
        "oof_sha256": oof_hash,
        "test_labels_used": False,
    }, indent=2) + "\n")

    print(report)
    print(f"\nReports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
