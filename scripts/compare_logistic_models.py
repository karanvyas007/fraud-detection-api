"""Compare Logistic Regression configurations with training-only, group-aware CV.

Configurations: class_weight in {None, "balanced"} x C in {0.1, 1.0, 10.0}.
(None, 1.0) is exactly the Phase 3A/3C baseline; every other configuration is
a clone of that pipeline with only C and/or class_weight changed.

Each configuration gets out-of-fold (OOF) probabilities from the same Phase 3B
StratifiedGroupKFold folds (duplicate groups kept together). In each fold a fresh
StandardScaler -> LogisticRegression pipeline is fitted on the fitting rows only.

Metrics are reported in two clearly separated families:
  - Ranking (threshold-free): average precision (PR-AUC), plus Brier score and mean
    predicted probability as calibration indicators.
  - Threshold-dependent: precision, recall, F1, FP, FN at
      * 0.50 (default rule, same for all),
      * 0.09 (Phase 3B threshold, tuned for the baseline only, shown for reference),
      * each configuration's own exact max-OOF-F1 threshold (optimistic because it
        is chosen and scored on the same OOF predictions),
      * an equal alert budget: every model flags as many rows as the baseline at 0.09.

Test labels are never read. Earlier artifacts are checksummed before and after.

Usage:
    python scripts/compare_logistic_models.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, brier_score_loss, precision_recall_curve

from inspect_dataset import DEFAULT_DATA_PATH, PROJECT_ROOT, DatasetValidationError, load_dataset, sha256_of, validate_dataset
from train_baseline import DEFAULT_SPLIT_DIR, POSITIVE_CLASS, RANDOM_STATE, build_models, fraud_probability
from tune_threshold import N_FOLDS, load_training_rows, make_folds, threshold_metrics, verify_folds

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "model_comparison"
PHASE_3B_RECORD = PROJECT_ROOT / "reports" / "threshold_tuning" / "selected_threshold.json"
BASELINE_NAME = "C=1.0, unweighted"
FIXED_THRESHOLDS = {"t=0.50": 0.50, "t=0.09 (3B)": 0.09}
BUDGET_REFERENCE_THRESHOLD = 0.09  # alert budget = rows the baseline flags at this threshold
CONFIGS = {
    f"C={c}, {'balanced' if cw else 'unweighted'}": {"model__C": c, "model__class_weight": cw}
    for cw in (None, "balanced")
    for c in (0.1, 1.0, 10.0)
}
PROTECTED_DIRS = ["data/raw", "data/processed", "models", "reports/eda", "reports/baseline",
                  "reports/threshold_tuning", "reports/final_evaluation"]


def protected_checksums() -> dict[str, str]:
    paths = sorted(p for d in PROTECTED_DIRS for p in (PROJECT_ROOT / d).glob("*")
                   if p.is_file() and p.name != ".DS_Store")
    return {str(p.relative_to(PROJECT_ROOT)): sha256_of(p) for p in paths}


def build_pipelines() -> dict:
    template = build_models()["logistic_regression"]
    pipelines = {name: clone(template).set_params(**params) for name, params in CONFIGS.items()}
    # The baseline entry must be the unmodified Phase 3A configuration.
    # Compare each step's own hyperparameters (get_params on the pipeline also holds step objects).
    step_params = lambda p: {name: step.get_params() for name, step in p.steps}
    if step_params(pipelines[BASELINE_NAME]) != step_params(template):
        raise AssertionError("Baseline configuration differs from the Phase 3A pipeline.")
    return pipelines


def oof_for_config(pipeline, X: pd.DataFrame, y: pd.Series, folds) -> tuple[np.ndarray, int]:
    oof = np.full(len(y), np.nan)
    convergence_warnings = 0
    for i, (fit_idx, val_idx) in enumerate(folds):
        model = clone(pipeline)  # fresh scaler + model; fitted on fitting rows only
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        proba = fraud_probability(model, X.iloc[val_idx])
        # Positive-class check: P(fraud) >= 0.5 must reproduce the model's own class predictions.
        if not np.array_equal((proba >= 0.5).astype(int), model.predict(X.iloc[val_idx])):
            raise AssertionError(f"Fold {i}: probabilities do not correspond to Class {POSITIVE_CLASS}.")
        oof[val_idx] = proba
    if np.isnan(oof).any() or not ((oof >= 0) & (oof <= 1)).all():
        raise AssertionError("OOF probabilities are missing or outside [0, 1].")
    return oof, convergence_warnings


def compute_all_oof(pipelines: dict, X, y, folds) -> tuple[dict, dict]:
    oof, conv = {}, {}
    for name, pipeline in pipelines.items():
        oof[name], conv[name] = oof_for_config(pipeline, X, y, folds)
    return oof, conv


def exact_max_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Max-F1 over every distinct score (no grid, so no edge effects near 0 or 1)."""
    precision, recall, thresholds = precision_recall_curve(y, proba, pos_label=POSITIVE_CLASS)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    best = np.flatnonzero(np.isclose(f1, f1.max()))
    return float(thresholds[best[-1]])  # ties -> higher threshold (fewer alerts)


def budget_threshold(proba: np.ndarray, budget: int) -> float:
    """Score of the budget-th highest row; flags ~budget rows (more only if scores tie)."""
    return float(np.sort(proba)[::-1][budget - 1])


def summarise(y: np.ndarray, oof: dict, fold_of_row: np.ndarray, conv: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, fold_rows, threshold_rows = [], [], []
    budget = threshold_metrics(y, oof[BASELINE_NAME], BUDGET_REFERENCE_THRESHOLD)["flagged"]
    for name, proba in oof.items():
        policies = {
            **FIXED_THRESHOLDS,
            "own max-F1 (exact)": exact_max_f1_threshold(y, proba),
            f"alert budget {budget}": budget_threshold(proba, budget),
        }
        for policy, t in policies.items():
            threshold_rows.append({"config": name, "policy": policy, **threshold_metrics(y, proba, t)})

        fold_ap = []
        for f in np.unique(fold_of_row):
            m = fold_of_row == f
            ap = average_precision_score(y[m], proba[m])
            fold_ap.append(ap)
            fold_rows.append({"config": name, "fold": int(f), "average_precision": ap,
                              "f1_at_own_threshold": threshold_metrics(y[m], proba[m], policies["own max-F1 (exact)"])["f1"]})
        summary_rows.append({
            "config": name,
            "oof_average_precision": average_precision_score(y, proba),
            "fold_ap_mean": np.mean(fold_ap),
            "fold_ap_std": np.std(fold_ap, ddof=1),
            "fold_ap_min": np.min(fold_ap),
            "fold_ap_max": np.max(fold_ap),
            "brier": brier_score_loss(y, proba),
            "mean_predicted_proba": proba.mean(),
            # Scores that round to exactly 1.0 tie at the top of the ranking and lower AP.
            "n_proba_equal_1": int((proba == 1.0).sum()),
            "own_threshold": policies["own max-F1 (exact)"],
            "convergence_warnings": conv[name],
        })

    summary = pd.DataFrame(summary_rows).set_index("config")
    folds = pd.DataFrame(fold_rows)
    # Paired comparison with the baseline on the same folds.
    base = folds[folds["config"] == BASELINE_NAME].set_index("fold")["average_precision"]
    diffs = folds.assign(ap_diff_vs_baseline=folds["average_precision"] - folds["fold"].map(base))
    paired = diffs.groupby("config")["ap_diff_vs_baseline"].agg(
        mean="mean", std=lambda s: s.std(ddof=1), folds_better=lambda s: int((s > 0).sum()))
    summary = summary.join(paired.add_prefix("ap_diff_"))
    return summary, diffs, pd.DataFrame(threshold_rows)


def format_report(summary, folds, thresholds, fold_summary, y, hashes, base_matches_3b, notes) -> str:
    pd.set_option("display.width", 200)
    ranking = summary[["oof_average_precision", "fold_ap_mean", "fold_ap_std", "fold_ap_min", "fold_ap_max",
                       "ap_diff_mean", "ap_diff_std", "ap_diff_folds_better", "brier", "mean_predicted_proba",
                       "n_proba_equal_1"]]
    fold_table = folds.pivot(index="config", columns="fold", values="average_precision")
    thr = thresholds.assign(precision=thresholds["precision"].round(3), recall=thresholds["recall"].round(3),
                            f1=thresholds["f1"].round(3))
    thr = thr[["config", "policy", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "flagged"]]
    lines = [
        "LOGISTIC REGRESSION COMPARISON - training data only, out-of-fold (OOF)",
        "=" * 70,
        f"Folds: StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE}) "
        "on duplicate groups (same folds as Phase 3B)",
        f"Training rows: {len(y):,} | fraud: {int(y.sum())} | observed fraud rate: {y.mean():.5f}",
        "Test labels were not read.",
        f"Baseline OOF identical to Phase 3B: {base_matches_3b}",
        "",
        "Validation folds",
        fold_summary.to_string(),
        "",
        "1) RANKING METRICS (threshold-free)",
        "   ap_diff_* = paired per-fold AP difference vs the baseline on identical folds",
        "   brier / mean_predicted_proba = calibration indicators (lower Brier is better;",
        f"   a well-calibrated model's mean probability is close to {y.mean():.5f})",
        ranking.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "Average precision per fold",
        fold_table.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "2) THRESHOLD-DEPENDENT METRICS (OOF, all 227,845 training rows)",
        "   'own max-F1 (exact)': each model's F1-maximising threshold over all distinct scores.",
        "     Tuned and scored on the same OOF predictions, so optimistic; same policy for all.",
        "   'alert budget N': each model flags its top-N scores, N = rows the baseline flags at",
        "     0.09. Compares models at equal review workload, independent of probability scale.",
        thr.to_string(index=False),
        "",
        "Notes",
        *[f"  - {n}" for n in notes],
        "",
        "OOF probability SHA-256 (identical across two independent runs)",
        *[f"  {k}: {v}" for k, v in hashes.items()],
    ]
    return "\n".join(lines)


def plot_comparison(y, oof, folds, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, proba in oof.items():
        PrecisionRecallDisplay.from_predictions(
            y, proba, pos_label=POSITIVE_CLASS, ax=axes[0], name=name,
            curve_kwargs={"linestyle": "--" if "balanced" in name else "-"},
        )
    axes[0].set_title("OOF precision-recall curves (training data)")
    axes[0].legend(fontsize=7, loc="lower left")

    names = list(oof)
    for f in sorted(folds["fold"].unique()):
        vals = folds[folds["fold"] == f].set_index("config").loc[names, "average_precision"]
        axes[1].plot(range(len(names)), vals, marker="o", alpha=0.6, label=f"fold {f}")
    axes[1].set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("Per-fold average precision (lines join the same fold)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected_before = protected_checksums()
    try:
        df = load_dataset(args.data_path)
        validate_dataset(df)
        X, y, groups, _ = load_training_rows(df, args.data_path, args.split_dir)
    except (FileNotFoundError, DatasetValidationError, KeyError) as exc:
        print(f"ERROR: split verification failed, not comparing: {exc}", file=sys.stderr)
        return 1

    folds = make_folds(X, y, groups)
    fold_summary = verify_folds(folds, y, groups)  # coverage, group separation, fraud in every fold
    fold_of_row = np.empty(len(y), dtype=int)
    for i, (_, val_idx) in enumerate(folds):
        fold_of_row[val_idx] = i

    pipelines = build_pipelines()
    oof, conv = compute_all_oof(pipelines, X, y, folds)

    # Reproducibility: rebuild folds and refit every configuration from scratch.
    oof_repeat, _ = compute_all_oof(build_pipelines(), X, y, make_folds(X, y, groups))
    for name in oof:
        if not np.array_equal(oof[name], oof_repeat[name]):
            raise AssertionError(f"{name}: OOF probabilities differ between two identical runs.")
    hashes = {name: hashlib.sha256(p.tobytes()).hexdigest() for name, p in oof.items()}
    base_matches_3b = None
    if PHASE_3B_RECORD.is_file():
        base_matches_3b = hashes[BASELINE_NAME] == json.loads(PHASE_3B_RECORD.read_text()).get("oof_sha256")

    y_arr = y.to_numpy()
    summary, fold_metrics, thresholds = summarise(y_arr, oof, fold_of_row, conv)

    notes = [
        "class_weight='balanced' weights each fraud ~290x a legitimate row during fitting. This changes the "
        "learned coefficients and pushes probabilities upward (see mean_predicted_proba), so its scores are "
        "not calibrated fraud probabilities and 0.50 / 0.09 do not mean the same thing for it.",
        "Weighted models saturate: n_proba_equal_1 rows get a probability of exactly 1.0 and tie at the top "
        "of the ranking, which lowers their AP. Ranking by decision_function would break those ties "
        "(a fold-0 spot check moved balanced AP from 0.687 to 0.698, still below unweighted 0.749).",
        "The 0.09 threshold was tuned for the unweighted C=1.0 baseline only; it is shown for reference, "
        "not as a fair operating point for other configurations.",
        "Compare configurations primarily on ranking metrics and on threshold metrics under each one's own "
        "threshold. Treat differences smaller than the fold-to-fold spread (ap_diff_std) as inconclusive.",
    ]
    if any(conv.values()):
        notes.append(f"Convergence warnings per configuration: {conv}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(summary, fold_metrics, thresholds, fold_summary, y_arr, hashes, base_matches_3b, notes)
    (args.report_dir / "comparison_report.txt").write_text(report + "\n")
    summary.to_csv(args.report_dir / "summary.csv", float_format="%.6f")
    fold_metrics.to_csv(args.report_dir / "fold_metrics.csv", index=False, float_format="%.6f")
    thresholds.to_csv(args.report_dir / "threshold_metrics.csv", index=False, float_format="%.6f")
    plot_comparison(y_arr, oof, fold_metrics, args.report_dir / "comparison_plot.png")
    (args.report_dir / "comparison_config.json").write_text(json.dumps({
        "configs": {k: {p.split("__")[1]: v for p, v in params.items()} for k, params in CONFIGS.items()},
        "baseline": BASELINE_NAME,
        "cv": f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE})",
        "fixed_thresholds": FIXED_THRESHOLDS,
        "own_threshold_policy": "exact max OOF F1 over all distinct scores, ties -> higher threshold",
        "alert_budget_policy": f"top-N scores, N = baseline flagged at {BUDGET_REFERENCE_THRESHOLD}",
        "oof_sha256": hashes,
        "baseline_oof_matches_phase_3b": base_matches_3b,
        "test_labels_used": False,
    }, indent=2) + "\n")

    if protected_checksums() != protected_before:
        print("ERROR: an earlier artifact changed during the comparison.", file=sys.stderr)
        return 1

    print(report)
    print(f"\nProtected files unchanged ({len(protected_before)} checked). Reports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
