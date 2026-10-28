"""Fit the final baseline pipeline on all training rows and evaluate it once on the test set.

The two decision thresholds are fixed *before* any test label is read:
  - 0.50: default baseline
  - 0.09: provisional, selected in Phase 3B from training-only out-of-fold F1
The script refuses to run if reports/threshold_tuning/selected_threshold.json disagrees.
Test results are reported, never used to change the model or thresholds.

Earlier artifacts (raw CSV, split files, Phase 3A model and reports, Phase 3B reports)
are checksummed before and after the run to prove they were not modified.

Usage:
    python scripts/train_final.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

from inspect_dataset import DEFAULT_DATA_PATH, PROJECT_ROOT, DatasetValidationError, load_dataset, sha256_of, validate_dataset
from train_baseline import DEFAULT_MODEL_DIR, DEFAULT_SPLIT_DIR, POSITIVE_CLASS, RANDOM_STATE, build_models, fraud_probability, load_split
from tune_threshold import threshold_metrics

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "final_evaluation"
THRESHOLD_FILE = PROJECT_ROOT / "reports" / "threshold_tuning" / "selected_threshold.json"
FIXED_THRESHOLDS = {"baseline": 0.50, "provisional": 0.09}
FINAL_MODEL_NAME = "final_logreg.joblib"

PROTECTED_FILES = [
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_DIR / "split_assignments.csv",
    DEFAULT_SPLIT_DIR / "split_metadata.json",
    DEFAULT_MODEL_DIR / "baseline_logreg.joblib",
    *sorted((PROJECT_ROOT / "reports" / "baseline").glob("*")),
    *sorted((PROJECT_ROOT / "reports" / "eda").glob("*")),
    *sorted((PROJECT_ROOT / "reports" / "threshold_tuning").glob("*")),
]


def checksums(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(PROJECT_ROOT)): sha256_of(p) for p in paths if p.is_file()}


def verify_fixed_thresholds() -> None:
    """Thresholds must match the Phase 3B record before test labels are touched."""
    if not THRESHOLD_FILE.is_file():
        raise FileNotFoundError(f"{THRESHOLD_FILE} not found. Run `python scripts/tune_threshold.py` first.")
    record = json.loads(THRESHOLD_FILE.read_text())
    if record.get("test_labels_used") is not False:
        raise DatasetValidationError("Threshold record does not confirm it was chosen without test labels.")
    recorded = {"baseline": record["baseline_threshold"], "provisional": record["provisional_threshold"]}
    if any(not np.isclose(recorded[k], v) for k, v in FIXED_THRESHOLDS.items()):
        raise DatasetValidationError(f"Thresholds {recorded} differ from the fixed values {FIXED_THRESHOLDS}.")


def evaluate_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    m = threshold_metrics(y_true, proba, threshold)
    m["precision"] = None if np.isnan(m["precision"]) else float(m["precision"])
    m["accuracy_secondary"] = (m["tp"] + m["tn"]) / len(y_true)
    m["average_precision"] = float(average_precision_score(y_true, proba, pos_label=POSITIVE_CLASS))
    return m


def evaluate_all(models: dict, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[dict, dict]:
    probas = {name: fraud_probability(model, X_test) for name, model in models.items()}
    results = {
        name: {label: evaluate_at(y_test, proba, t) for label, t in FIXED_THRESHOLDS.items()}
        for name, proba in probas.items()
    }
    return results, probas


def same_fitted_pipeline(a, b) -> bool:
    sa, sb = a.named_steps, b.named_steps
    return (np.array_equal(sa["scaler"].mean_, sb["scaler"].mean_)
            and np.array_equal(sa["scaler"].scale_, sb["scaler"].scale_)
            and np.array_equal(sa["model"].coef_, sb["model"].coef_)
            and np.array_equal(sa["model"].intercept_, sb["model"].intercept_))


def format_report(results: dict, y_test: np.ndarray, notes: list[str]) -> str:
    def fmt(v):
        return "undefined" if v is None else f"{v:.4f}"

    lines = [
        "FINAL BASELINE EVALUATION - held-out test set (single evaluation)",
        "=" * 64,
        f"Test rows: {len(y_test):,} | fraud: {int(y_test.sum())} ({y_test.mean():.4%})",
        f"Fixed thresholds (set before reading test labels): {FIXED_THRESHOLDS}",
        "",
        f"{'model':<22}{'thresh':>7}{'TN':>8}{'FP':>6}{'FN':>5}{'TP':>5}{'flagged':>9}"
        f"{'precision':>11}{'recall':>8}{'F1':>8}{'PR-AUC':>8}{'accuracy':>10}",
    ]
    for name, by_threshold in results.items():
        for m in by_threshold.values():
            lines.append(
                f"{name:<22}{m['threshold']:>7.2f}{m['tn']:>8,}{m['fp']:>6}{m['fn']:>5}{m['tp']:>5}{m['flagged']:>9}"
                f"{fmt(m['precision']):>11}{m['recall']:>8.4f}{m['f1']:>8.4f}{m['average_precision']:>8.4f}"
                f"{m['accuracy_secondary']:>10.4%}"
            )
    lines += ["", "Notes", *[f"  - {n}" for n in notes]]
    return "\n".join(lines)


def plot_pr(y_test: np.ndarray, probas: dict, results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        PrecisionRecallDisplay.from_predictions(y_test, proba, pos_label=POSITIVE_CLASS, ax=ax, name=name)
    for label, marker in [("baseline", "o"), ("provisional", "s")]:
        m = results["final_logreg"][label]
        ax.scatter(m["recall"], m["precision"], marker=marker, color="black", zorder=3,
                   label=f"t={m['threshold']:.2f}: P={m['precision']:.2f}, R={m['recall']:.2f}, FP={m['fp']}")
    ax.set_title("Final model - precision-recall (test set)")
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
    protected_before = checksums(PROTECTED_FILES)
    try:
        verify_fixed_thresholds()  # before any test label is read
        df = load_dataset(args.data_path)
        validate_dataset(df)
        X_train, y_train, X_test, y_test, features = load_split(df, args.data_path, args.split_dir)
    except (FileNotFoundError, DatasetValidationError, KeyError) as exc:
        print(f"ERROR: pre-flight check failed, not training: {exc}", file=sys.stderr)
        return 1

    # Fresh, unfitted copies of the Phase 3A configuration, fitted on all training rows only.
    templates = build_models()
    models = {
        "dummy_most_frequent": clone(templates["dummy_most_frequent"]).fit(X_train, y_train),
        "final_logreg": clone(templates["logistic_regression"]).fit(X_train, y_train),
    }
    final = models["final_logreg"]
    if list(final.feature_names_in_) != features:
        raise AssertionError("Fitted feature order differs from the split metadata feature order.")

    y_test_arr = y_test.to_numpy()
    results, probas = evaluate_all(models, X_test, y_test_arr)

    notes = [
        "dummy_most_frequent never predicts fraud: precision = 0/0 is undefined; its PR-AUC equals the "
        "fraud prevalence because every row gets the same score.",
        "PR-AUC is threshold-independent, so it is identical for both thresholds of the same model.",
    ]
    baseline_path = args.model_dir / "baseline_logreg.joblib"
    if baseline_path.is_file():
        identical = same_fitted_pipeline(final, joblib.load(baseline_path)["model"])
        notes.append(
            "final_logreg is bit-identical to the Phase 3A baseline (same config, data and deterministic solver), "
            "so its 0.50 test result was already reported in Phase 3A; only the 0.09 result is new."
            if identical else
            "final_logreg differs from the Phase 3A baseline parameters; investigate before comparing reports."
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / FINAL_MODEL_NAME
    if model_path.resolve() == baseline_path.resolve():
        raise AssertionError("Refusing to overwrite the Phase 3A baseline model.")

    metadata = {
        "model_type": "Pipeline(StandardScaler -> LogisticRegression)",
        "training_config": {"logistic_regression": final.named_steps["model"].get_params(), "random_state": RANDOM_STATE,
                            "trained_on": "all training rows", "resampling": None, "class_weight": None},
        "feature_columns": features,
        "positive_class": POSITIVE_CLASS,
        "thresholds": FIXED_THRESHOLDS,
        "threshold_source": str(THRESHOLD_FILE.relative_to(PROJECT_ROOT)),
        "train_rows": int(len(y_train)),
        "train_fraud": int(y_train.sum()),
        "checksums": checksums([args.data_path, args.split_dir / "split_assignments.csv",
                                args.split_dir / "split_metadata.json", THRESHOLD_FILE]),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    joblib.dump({"model": final, **metadata}, model_path)

    # Reload check: the saved artifact must reproduce identical probabilities and metrics.
    reloaded = joblib.load(model_path)
    reloaded_proba = fraud_probability(reloaded["model"], X_test[reloaded["feature_columns"]])
    if not np.array_equal(reloaded_proba, probas["final_logreg"]):
        raise AssertionError("Reloaded model produces different test probabilities.")
    reloaded_metrics = {label: evaluate_at(y_test_arr, reloaded_proba, t) for label, t in reloaded["thresholds"].items()}
    if reloaded_metrics != results["final_logreg"]:
        raise AssertionError("Reloaded model produces different test metrics.")
    notes.append("Reload check passed: saved model reproduces identical test probabilities and metrics.")

    report = format_report(results, y_test_arr, notes)
    (args.report_dir / "test_evaluation.txt").write_text(report + "\n")
    (args.report_dir / "test_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.report_dir / "model_metadata.json").write_text(json.dumps(
        {**metadata, "model_file": str(model_path.relative_to(PROJECT_ROOT)), "model_sha256": sha256_of(model_path),
         "test_proba_sha256": hashlib.sha256(probas["final_logreg"].tobytes()).hexdigest()},
        indent=2, default=str) + "\n")
    plot_pr(y_test_arr, probas, results, args.report_dir / "pr_curve.png")

    protected_after = checksums(PROTECTED_FILES)
    if protected_after != protected_before:
        changed = [k for k in protected_before if protected_before[k] != protected_after.get(k)]
        print(f"ERROR: protected files changed: {changed}", file=sys.stderr)
        return 1

    print(report)
    print(f"\nProtected files unchanged ({len(protected_before)} checked).")
    print(f"Model saved to:   {model_path}")
    print(f"Reports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
