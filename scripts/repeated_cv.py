"""Repeated group-aware CV: unweighted vs class_weight={0: 1, 1: 10} Logistic Regression.

Only the fold assignment changes between repeats: StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=seed) for each seed in SEEDS (fixed in advance, not chosen after looking at
results). Both models are scored on exactly the same folds within a seed; in every fold a fresh
StandardScaler -> LogisticRegression is fitted on the fitting rows only. The model's own
random_state stays at 42 (the lbfgs solver is deterministic and does not use it).

Per seed:
  - pooled OOF average precision on decision scores (and the mean of per-fold AP),
  - an equal alert budget of 397 alerts per pass over the training set, split across folds
    in proportion to fold size; within each fold rows are ranked by decision score and ties
    are broken by ascending row_id, so exactly the budget is flagged,
  - challenger minus baseline differences in frauds caught and false alarms.

Seeds reuse the same 227,845 rows, so repeats are strongly correlated. The summary reports
counts, ranges and spread only - no significance test that treats seeds or folds as independent.

Test labels are never read. Earlier files are checksummed before and after.

Usage:
    python scripts/repeated_cv.py
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from inspect_dataset import DEFAULT_DATA_PATH, PROJECT_ROOT, DatasetValidationError, load_dataset, sha256_of, validate_dataset
from train_baseline import DEFAULT_SPLIT_DIR, build_models
from tune_threshold import N_FOLDS, load_training_rows, verify_folds
from weighting_experiment import TOTAL_ALERT_BUDGET, allocate_budget, oof_scores, top_k

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "repeated_cv"
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # 42 = the seed used in Phases 3B-4B
CONFIGS = {"baseline": None, "w=10": {0: 1, 1: 10}}
PHASE_4B_CONFIG = PROJECT_ROOT / "reports" / "weighting_experiment" / "experiment_config.json"
PHASE_4B_KEYS = {"baseline": "unweighted decision", "w=10": "w=10 decision"}
PROTECTED_DIRS = ["data/raw", "data/processed", "models", "scripts", "reports/eda", "reports/baseline",
                  "reports/threshold_tuning", "reports/final_evaluation", "reports/model_comparison",
                  "reports/weighting_experiment"]


def protected_checksums() -> dict[str, str]:
    paths = sorted(p for d in PROTECTED_DIRS for p in (PROJECT_ROOT / d).glob("*")
                   if p.is_file() and p.name != ".DS_Store" and p.name != Path(__file__).name)
    return {str(p.relative_to(PROJECT_ROOT)): sha256_of(p) for p in paths}


def build_pipelines() -> dict:
    template = build_models()["logistic_regression"]
    return {name: clone(template).set_params(model__class_weight=cw) for name, cw in CONFIGS.items()}


def folds_for_seed(X, y, groups, seed: int):
    return list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups))


def partition_fingerprint(folds) -> str:
    """Hash of the set of validation folds, independent of fold order."""
    canonical = sorted((np.sort(v) for _, v in folds), key=lambda a: a[0])
    return hashlib.sha256(b"|".join(a.tobytes() for a in canonical)).hexdigest()


def run_seed(X, y, groups, seed: int) -> tuple[list, dict]:
    folds = folds_for_seed(X, y, groups, seed)
    verify_folds(folds, y, groups)  # coverage, group separation, fraud in every fold
    scores = {}
    for name, pipeline in build_pipelines().items():
        _, decision, conv = oof_scores(pipeline, X, y, folds)  # also checks positive-class handling
        if conv:
            raise AssertionError(f"seed {seed}, {name}: {conv} convergence warnings.")
        scores[name] = decision
    return folds, scores


def evaluate_seed(seed: int, folds, scores: dict, y: np.ndarray, row_ids: np.ndarray) -> tuple[dict, list, list]:
    budgets = allocate_budget([len(v) for _, v in folds], TOTAL_ALERT_BUDGET)
    alerts, fold_rows, tie_rows = {}, [], []
    for name, d in scores.items():
        mask_all = np.zeros(len(y), dtype=bool)
        for f, (_, val_idx) in enumerate(folds):
            mask, tie = top_k(d[val_idx], row_ids[val_idx], budgets[f])
            mask_all[val_idx[mask]] = True
            yf = y[val_idx]
            fold_rows.append({"seed": seed, "config": name, "fold": f, "ap": average_precision_score(yf, d[val_idx]),
                              "alerts": int(mask.sum()), "tp": int(yf[mask].sum())})
            tie_rows.append({"seed": seed, "config": name, "fold": f, **tie})
        alerts[name] = mask_all

    is_fraud = y == 1
    row = {"seed": seed, "fold_budgets": budgets}
    for name, d in scores.items():
        tp = int((alerts[name] & is_fraud).sum())
        n_alerts = int(alerts[name].sum())
        fold_ap = [r["ap"] for r in fold_rows if r["config"] == name]
        row |= {f"{name}_ap_pooled": average_precision_score(y, d), f"{name}_ap_fold_mean": float(np.mean(fold_ap)),
                f"{name}_alerts": n_alerts, f"{name}_tp": tp, f"{name}_fp": n_alerts - tp,
                f"{name}_fn": int(is_fraud.sum()) - tp, f"{name}_recall": tp / is_fraud.sum(),
                f"{name}_precision": tp / n_alerts}
    row |= {
        "ap_pooled_diff": row["w=10_ap_pooled"] - row["baseline_ap_pooled"],
        "ap_fold_mean_diff": row["w=10_ap_fold_mean"] - row["baseline_ap_fold_mean"],
        "tp_diff": row["w=10_tp"] - row["baseline_tp"],
        "fp_diff": row["w=10_fp"] - row["baseline_fp"],
        "frauds_only_w10": int((alerts["w=10"] & ~alerts["baseline"] & is_fraud).sum()),
        "frauds_only_baseline": int((alerts["baseline"] & ~alerts["w=10"] & is_fraud).sum()),
    }
    return row, fold_rows, tie_rows


def describe(s: pd.Series) -> str:
    return (f"mean {s.mean():+.4g}, std {s.std(ddof=1):.3g}, min {s.min():+.4g}, max {s.max():+.4g} | "
            f"better {int((s > 0).sum())}, equal {int((s == 0).sum())}, worse {int((s < 0).sum())}")


def format_report(per_seed: pd.DataFrame, folds: pd.DataFrame, ties: pd.DataFrame, n_rows: int, n_fraud: int,
                  distinct_partitions: int, phase_4b_match, conv_note: str) -> str:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    ap_cols = ["seed", "baseline_ap_pooled", "w=10_ap_pooled", "ap_pooled_diff", "baseline_ap_fold_mean",
               "w=10_ap_fold_mean", "ap_fold_mean_diff"]
    budget_cols = ["seed", "baseline_alerts", "baseline_tp", "baseline_fp", "baseline_fn", "baseline_recall",
                   "baseline_precision", "w=10_alerts", "w=10_tp", "w=10_fp", "w=10_fn", "w=10_recall",
                   "w=10_precision", "tp_diff", "fp_diff", "frauds_only_w10", "frauds_only_baseline"]
    pivot = folds.pivot_table(index=["seed", "fold"], columns="config", values=["ap", "tp"])
    fold_ap_diff = pivot[("ap", "w=10")] - pivot[("ap", "baseline")]
    fold_tp_diff = pivot[("tp", "w=10")] - pivot[("tp", "baseline")]
    cutoff_ties = ties[ties["tied_at_cutoff"] > 1]
    lines = [
        "REPEATED GROUP-AWARE CV - baseline (unweighted) vs w=10 ({0: 1, 1: 10}), C=1.0, training data only",
        "=" * 98,
        f"Training rows: {n_rows:,} | fraud: {n_fraud} | Test labels were not read.",
        f"Seeds: {SEEDS} -> {distinct_partitions} distinct fold partitions; {N_FOLDS} folds each",
        f"Seed 42 reproduces Phase 4B decision scores: {phase_4b_match}",
        conv_note,
        f"Alert budget: {TOTAL_ALERT_BUDGET} per pass over the training set, split across folds by fold size;",
        "  rank by decision score within each fold, ties broken by ascending row_id (exactly the budget is flagged).",
        "",
        "1) RANKING - average precision on decision scores",
        per_seed[ap_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "2) EQUAL ALERT BUDGET (pooled over the 5 folds of each seed)",
        per_seed[budget_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "   At a fixed budget, every extra fraud caught is one fewer false alarm (fp_diff = -tp_diff).",
        "",
        "3) STABILITY ACROSS SEEDS (w=10 minus baseline; one value per seed)",
        f"   pooled AP diff:     {describe(per_seed['ap_pooled_diff'])}",
        f"   fold-mean AP diff:  {describe(per_seed['ap_fold_mean_diff'])}",
        f"   frauds caught diff: {describe(per_seed['tp_diff'])}",
        f"   false alarms diff:  {describe(per_seed['fp_diff'])}",
        "",
        f"   Per seed x fold ({len(fold_ap_diff)} folds; NOT independent - same rows reappear across seeds)",
        f"   fold AP diff:       {describe(fold_ap_diff)}",
        f"   fold TP diff:       {describe(fold_tp_diff)}",
        "",
        "4) TIES AT THE BUDGET CUTOFF",
        (f"   {len(cutoff_ties)} of {len(ties)} fold cutoffs fell inside a tied score group:\n"
         + cutoff_ties.to_string(index=False)) if len(cutoff_ties)
        else f"   None: all {len(ties)} fold cutoffs fell on a unique decision score; the row_id rule was never needed.",
        "",
        "LIMITATIONS",
        "  - Every seed re-partitions the same 227,845 rows and 393 frauds. Seeds are correlated views of one",
        "    dataset, not independent replications, so no p-value is computed from them.",
        "  - Stability across seeds shows the result does not hinge on one lucky partition; it does not show",
        "    the effect generalises to new data or future fraud patterns.",
        "  - Each fold model trains on 80% of the training rows; a model trained on all rows may differ slightly.",
        "  - The 397-alert budget is one operating point; the comparison can change at other budgets.",
        "  - Duplicate groups and random (not time-ordered) splits mean this does not test temporal drift.",
    ]
    return "\n".join(lines)


def plot(per_seed: pd.DataFrame, folds: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = range(len(per_seed))
    labels = per_seed["seed"].astype(str)
    axes[0].bar(x, per_seed["ap_pooled_diff"], color="#4C72B0")
    axes[0].set_title("Pooled AP difference (w=10 - baseline)")
    axes[1].bar(x, per_seed["tp_diff"], color="#55A868")
    axes[1].set_title(f"Extra frauds caught at {TOTAL_ALERT_BUDGET} alerts")
    for ax in axes[:2]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(list(x), labels)
        ax.set_xlabel("fold seed")

    pivot = folds.pivot_table(index=["seed", "fold"], columns="config", values="tp")
    diffs = (pivot["w=10"] - pivot["baseline"]).astype(int)
    counts = diffs.value_counts().sort_index()
    axes[2].bar(counts.index, counts.values, color="#C44E52")
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_title(f"Per-fold TP difference ({len(diffs)} seed x fold, not independent)")
    axes[2].set_xlabel("w=10 TP - baseline TP")
    axes[2].set_ylabel("folds")
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
    if len(set(SEEDS)) < 5:
        raise AssertionError("At least five distinct seeds are required.")
    protected_before = protected_checksums()
    try:
        df = load_dataset(args.data_path)
        validate_dataset(df)
        X, y, groups, _ = load_training_rows(df, args.data_path, args.split_dir)
    except (FileNotFoundError, DatasetValidationError, KeyError) as exc:
        print(f"ERROR: split verification failed, not running: {exc}", file=sys.stderr)
        return 1

    y_arr, row_ids = y.to_numpy(), X.index.to_numpy()
    seed_rows, fold_rows, tie_rows, hashes, partitions = [], [], [], {}, set()
    for seed in SEEDS:
        folds, scores = run_seed(X, y, groups, seed)
        # Reproducibility: rebuild the folds and refit both models; scores must be bit-identical.
        _, repeat = run_seed(X, y, groups, seed)
        for name in scores:
            if not np.array_equal(scores[name], repeat[name]):
                raise AssertionError(f"seed {seed}, {name}: decision scores differ between identical runs.")
            hashes[f"seed {seed} {name}"] = hashlib.sha256(scores[name].tobytes()).hexdigest()
        partitions.add(partition_fingerprint(folds))
        row, fr, tr = evaluate_seed(seed, folds, scores, y_arr, row_ids)
        seed_rows.append(row)
        fold_rows += fr
        tie_rows += tr
        print(f"seed {seed}: done (AP diff {row['ap_pooled_diff']:+.4f}, TP diff {row['tp_diff']:+d})", file=sys.stderr)

    phase_4b_match = None
    if PHASE_4B_CONFIG.is_file():
        ref = json.loads(PHASE_4B_CONFIG.read_text())["oof_sha256"]
        phase_4b_match = all(ref[PHASE_4B_KEYS[n]] == hashes[f"seed 42 {n}"] for n in CONFIGS)
        if not phase_4b_match:
            raise AssertionError("Seed 42 does not reproduce the Phase 4B decision scores.")

    per_seed = pd.DataFrame(seed_rows)
    folds_df, ties_df = pd.DataFrame(fold_rows), pd.DataFrame(tie_rows)
    if not (per_seed[["baseline_alerts", "w=10_alerts"]] == TOTAL_ALERT_BUDGET).all().all():
        raise AssertionError("A seed did not flag exactly the alert budget.")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(per_seed, folds_df, ties_df, len(y_arr), int(y_arr.sum()), len(partitions),
                           phase_4b_match, "Convergence warnings: none (any warning stops the run).")
    (args.report_dir / "repeated_cv_report.txt").write_text(report + "\n")
    per_seed.to_csv(args.report_dir / "per_seed.csv", index=False, float_format="%.6f")
    folds_df.to_csv(args.report_dir / "per_fold.csv", index=False, float_format="%.6f")
    ties_df.to_csv(args.report_dir / "cutoff_ties.csv", index=False, float_format="%.6f")
    plot(per_seed, folds_df, args.report_dir / "repeated_cv_plot.png")
    (args.report_dir / "repeated_cv_config.json").write_text(json.dumps({
        "configs": {"baseline": {"class_weight": None, "C": 1.0}, "w=10": {"class_weight": {"0": 1, "1": 10}, "C": 1.0}},
        "pipeline": "StandardScaler -> LogisticRegression(max_iter=1000, random_state=42)",
        "seeds": SEEDS,
        "cv": f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state=seed), groups = duplicate group_id",
        "distinct_partitions": len(partitions),
        "ranking_score": "decision_function",
        "alert_budget": TOTAL_ALERT_BUDGET,
        "tie_policy": "within each fold: sort by decision score desc, then row_id asc; take exactly the fold budget",
        "resampling": None, "calibration": None,
        "source_checksums": {k: v for k, v in protected_before.items()
                             if k in ("data/raw/creditcard.csv", "data/processed/split_assignments.csv",
                                      "data/processed/split_metadata.json")},
        "decision_score_sha256": hashes,
        "seed_42_matches_phase_4b": phase_4b_match,
        "test_labels_used": False,
    }, indent=2) + "\n")

    if protected_checksums() != protected_before:
        print("ERROR: an earlier file changed during the experiment.", file=sys.stderr)
        return 1

    print(report)
    print(f"\nProtected files unchanged ({len(protected_before)} checked). Reports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
