"""Investigate score saturation and compare milder class weights (training data only).

Configurations (all C=1.0, otherwise the Phase 3A pipeline):
    unweighted (baseline), {0: 1, 1: 10}, {0: 1, 1: 50}, {0: 1, 1: 100}, "balanced" (reference)

For every configuration, the same five Phase 3B StratifiedGroupKFold folds are used and a
fresh StandardScaler -> LogisticRegression pipeline is fitted on each fold's fitting rows.
Both predict_proba[:, Class 1] and decision_function are collected out-of-fold (OOF).

Analyses:
  1. Saturation: probabilities that are exactly 0 or 1, tied scores, AP from probabilities
     vs AP from decision scores (decision scores are log-odds, not probabilities).
  2. Ranking: average precision on decision scores, per fold and pooled.
  3. Equal alert budget: each fold flags its share of 397 alerts, ranked by decision score.
     Ties are broken deterministically by row_id (ascending), so exactly the budget is flagged
     even when the cutoff falls inside a tied score group.
  4. Probability quality: Brier score, log loss and a reliability table. No calibrator is fitted.
  5. Uncertainty: per-fold paired differences vs the baseline, plus a pooled exact binomial test
     on frauds caught by only one of the two models (descriptive, not proof).

Test labels are never read. Earlier files are checksummed before and after.

Usage:
    python scripts/weighting_experiment.py
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
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from inspect_dataset import DEFAULT_DATA_PATH, PROJECT_ROOT, DatasetValidationError, load_dataset, sha256_of, validate_dataset
from train_baseline import DEFAULT_SPLIT_DIR, POSITIVE_CLASS, RANDOM_STATE, build_models, fraud_probability
from tune_threshold import N_FOLDS, load_training_rows, make_folds, verify_folds

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "weighting_experiment"
BASELINE_NAME = "unweighted"
CONFIGS = {
    "unweighted": None,
    "w=10": {0: 1, 1: 10},
    "w=50": {0: 1, 1: 50},
    "w=100": {0: 1, 1: 100},
    "balanced": "balanced",
}
TOTAL_ALERT_BUDGET = 397  # rows the baseline flags at threshold 0.09 in Phase 3B
RELIABILITY_BINS = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0]
# Reference OOF hashes from earlier phases: same folds + same config must give identical scores.
EARLIER_OOF_HASHES = {
    "unweighted": ("reports/threshold_tuning/selected_threshold.json", ["oof_sha256"]),
    "balanced": ("reports/model_comparison/comparison_config.json", ["oof_sha256", "C=1.0, balanced"]),
}
PROTECTED_DIRS = ["data/raw", "data/processed", "models", "scripts", "reports/eda", "reports/baseline",
                  "reports/threshold_tuning", "reports/final_evaluation", "reports/model_comparison"]


def protected_checksums() -> dict[str, str]:
    paths = sorted(p for d in PROTECTED_DIRS for p in (PROJECT_ROOT / d).glob("*")
                   if p.is_file() and p.name != ".DS_Store" and p.name != Path(__file__).name)
    return {str(p.relative_to(PROJECT_ROOT)): sha256_of(p) for p in paths}


def build_pipelines() -> dict:
    template = build_models()["logistic_regression"]
    return {name: clone(template).set_params(model__class_weight=cw) for name, cw in CONFIGS.items()}


def oof_scores(pipeline, X: pd.DataFrame, y: pd.Series, folds) -> tuple[np.ndarray, np.ndarray, int]:
    """OOF P(Class=1) and decision scores; every fold uses a fresh pipeline fitted on fitting rows only."""
    proba, decision = np.full(len(y), np.nan), np.full(len(y), np.nan)
    n_conv = 0
    for i, (fit_idx, val_idx) in enumerate(folds):
        model = clone(pipeline)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        n_conv += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        X_val = X.iloc[val_idx]
        p, d = fraud_probability(model, X_val), model.decision_function(X_val)
        # Positive-class checks: decision > 0 <=> predicted fraud, and sigmoid(decision) == P(Class=1).
        if list(model.classes_) != [0, POSITIVE_CLASS]:
            raise AssertionError(f"Fold {i}: unexpected class order {model.classes_}.")
        if not np.array_equal((d > 0).astype(int), model.predict(X_val)):
            raise AssertionError(f"Fold {i}: decision_function sign does not match predictions.")
        if not np.allclose(expit(d), p, rtol=0, atol=1e-12):
            raise AssertionError(f"Fold {i}: probabilities are not sigmoid(decision) for Class {POSITIVE_CLASS}.")
        proba[val_idx], decision[val_idx] = p, d
    if np.isnan(proba).any() or np.isnan(decision).any():
        raise AssertionError("Some training rows received no OOF score.")
    return proba, decision, n_conv


def run_all(X, y, folds) -> dict:
    out = {}
    for name, pipeline in build_pipelines().items():
        p, d, conv = oof_scores(pipeline, X, y, folds)
        out[name] = {"proba": p, "decision": d, "convergence_warnings": conv}
    return out


def tie_stats(scores: np.ndarray, fold_of_row: np.ndarray, groups: np.ndarray) -> dict:
    """Ties are counted within a fold (where AP and alerts are computed).
    Identical feature rows (same duplicate group) legitimately share a score; ties across
    different groups are the ones caused by limited numeric resolution (saturation)."""
    frame = pd.DataFrame({"fold": fold_of_row, "score": scores, "group": groups})
    per_value = frame.groupby(["fold", "score"])
    rows_tied = per_value["score"].transform("size") > 1
    rows_tied_across_groups = per_value["group"].transform("nunique") > 1
    return {"rows_in_ties": int(rows_tied.sum()), "rows_tied_across_groups": int(rows_tied_across_groups.sum())}


def allocate_budget(fold_sizes: list[int], total: int) -> list[int]:
    """Split the alert budget across folds in proportion to size (largest remainder, ties -> lower fold)."""
    exact = np.array(fold_sizes) / sum(fold_sizes) * total
    alloc = np.floor(exact).astype(int)
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - alloc[i]), i))
    for i in order[: total - alloc.sum()]:
        alloc[i] += 1
    return alloc.tolist()


def top_k(scores: np.ndarray, row_ids: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """Select exactly k rows: highest score first, ties broken by ascending row_id."""
    order = np.lexsort((row_ids, -scores))
    chosen = order[:k]
    cutoff = scores[order[k - 1]]
    tied_at_cutoff = int((scores == cutoff).sum())
    included = int((scores[chosen] == cutoff).sum())
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    return mask, {"cutoff_score": float(cutoff), "tied_at_cutoff": tied_at_cutoff, "tied_included": included}


def reliability_table(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    bins = pd.cut(proba, RELIABILITY_BINS, include_lowest=True)
    table = pd.DataFrame({"bin": bins, "p": proba, "y": y}).groupby("bin", observed=False).agg(
        rows=("y", "size"), frauds=("y", "sum"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    t = table[table["rows"] > 0]
    return float((t["rows"] * (t["mean_predicted"] - t["observed_rate"]).abs()).sum() / t["rows"].sum())


def analyse(results: dict, y: np.ndarray, folds, groups: np.ndarray, row_ids: np.ndarray) -> dict:
    fold_of_row = np.empty(len(y), dtype=int)
    for i, (_, val_idx) in enumerate(folds):
        fold_of_row[val_idx] = i
    fold_budgets = allocate_budget([len(v) for _, v in folds], TOTAL_ALERT_BUDGET)

    saturation, ranking, budget_rows, reliability, alerts = [], [], [], {}, {}
    for name, r in results.items():
        p, d = r["proba"], r["decision"]
        pt, dt = tie_stats(p, fold_of_row, groups), tie_stats(d, fold_of_row, groups)
        saturation.append({
            "config": name, "proba_exactly_0": int((p == 0).sum()), "proba_exactly_1": int((p == 1).sum()),
            "frauds_at_proba_1": int(y[p == 1].sum()),
            "proba_rows_in_ties": pt["rows_in_ties"], "proba_ties_across_groups": pt["rows_tied_across_groups"],
            "decision_rows_in_ties": dt["rows_in_ties"], "decision_ties_across_groups": dt["rows_tied_across_groups"],
            "max_decision": float(d.max()), "min_decision": float(d.min()),
        })

        alert_mask = np.zeros(len(y), dtype=bool)
        for f, (_, val_idx) in enumerate(folds):
            yf, pf, df_ = y[val_idx], p[val_idx], d[val_idx]
            mask, tie_info = top_k(df_, row_ids[val_idx], fold_budgets[f])
            alert_mask[val_idx[mask]] = True
            tp = int(yf[mask].sum())
            ranking.append({"config": name, "fold": f, "ap_proba": average_precision_score(yf, pf),
                            "ap_decision": average_precision_score(yf, df_)})
            budget_rows.append({"config": name, "fold": f, "alerts": int(mask.sum()), "tp": tp,
                                "fp": int(mask.sum()) - tp, "fn": int(yf.sum()) - tp,
                                "recall": tp / yf.sum(), "precision": tp / mask.sum(), **tie_info})
        alerts[name] = alert_mask

        rel = reliability_table(y, p)
        reliability[name] = rel
        r["pooled"] = {
            "ap_proba_pooled": average_precision_score(y, p),
            # Pooled decision scores mix five fold models' scales; fold-level AP is the primary view.
            "ap_decision_pooled": average_precision_score(y, d),
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
            "mean_predicted": float(p.mean()),
            "ece_binned": expected_calibration_error(rel),
        }

    ranking, budget = pd.DataFrame(ranking), pd.DataFrame(budget_rows)
    base_rank = ranking[ranking["config"] == BASELINE_NAME].set_index("fold")
    base_budget = budget[budget["config"] == BASELINE_NAME].set_index("fold")
    ranking["ap_decision_diff"] = ranking["ap_decision"] - ranking["fold"].map(base_rank["ap_decision"])
    budget["tp_diff"] = budget["tp"] - budget["fold"].map(base_budget["tp"])

    paired = []
    is_fraud = y == 1
    for name in results:
        rk, bd = ranking[ranking["config"] == name], budget[budget["config"] == name]
        only_this = int((alerts[name] & ~alerts[BASELINE_NAME] & is_fraud).sum())
        only_base = int((alerts[BASELINE_NAME] & ~alerts[name] & is_fraud).sum())
        discordant = only_this + only_base
        paired.append({
            "config": name,
            "fold_ap_decision_mean": rk["ap_decision"].mean(), "fold_ap_decision_std": rk["ap_decision"].std(ddof=1),
            "ap_decision_diff_mean": rk["ap_decision_diff"].mean(), "ap_decision_diff_std": rk["ap_decision_diff"].std(ddof=1),
            "ap_folds_better/worse": f"{int((rk['ap_decision_diff'] > 0).sum())}/{int((rk['ap_decision_diff'] < 0).sum())}",
            "budget_tp_total": int(bd["tp"].sum()), "budget_alerts_total": int(bd["alerts"].sum()),
            "tp_diff_per_fold": bd["tp_diff"].astype(int).tolist(),
            "tp_diff_mean": bd["tp_diff"].mean(), "tp_diff_std": bd["tp_diff"].std(ddof=1),
            "frauds_only_this": only_this, "frauds_only_baseline": only_base,
            "binomial_p_two_sided": binomtest(only_this, discordant, 0.5).pvalue if discordant else 1.0,
        })
    return {"saturation": pd.DataFrame(saturation).set_index("config"), "ranking": ranking, "budget": budget,
            "paired": pd.DataFrame(paired).set_index("config"), "reliability": reliability,
            "fold_budgets": fold_budgets}


def format_report(a: dict, results: dict, fold_summary, y, hashes, earlier_match, conv) -> str:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    f4 = lambda x: f"{x:.4f}"
    pooled = pd.DataFrame({n: r["pooled"] for n, r in results.items()}).T
    rk = a["ranking"]
    paired = a["paired"]
    lines = [
        "CLASS-WEIGHTING & SATURATION EXPERIMENT - training data only, out-of-fold",
        "=" * 72,
        f"Folds: StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE}) on duplicate "
        "groups (same folds as Phases 3B/4A). All configs C=1.0.",
        f"Training rows: {len(y):,} | fraud: {int(y.sum())} | fraud rate: {y.mean():.5f} | Test labels were not read.",
        f"OOF probabilities identical to earlier phases: {earlier_match}",
        f"Convergence warnings: {conv}",
        "",
        "Validation folds", fold_summary.to_string(), "",
        "1) SATURATION (ties counted within each fold; 'across groups' excludes identical feature rows,",
        "   which legitimately share a score)",
        a["saturation"].to_string(float_format=lambda x: f"{x:.2f}"), "",
        "2) RANKING - average precision per fold, from probabilities vs decision scores",
        "   Decision scores are log-odds: monotonic with probability but unbounded, so they avoid",
        "   float saturation at 1.0. They are NOT probabilities; 0.09 or 0.5 thresholds do not apply.",
        rk.pivot(index="config", columns="fold", values="ap_proba").add_prefix("proba_f").join(
            rk.pivot(index="config", columns="fold", values="ap_decision").add_prefix("dec_f")).to_string(float_format=f4),
        "",
        "   Paired vs baseline (decision-score AP, same folds)",
        paired[["fold_ap_decision_mean", "fold_ap_decision_std", "ap_decision_diff_mean", "ap_decision_diff_std",
                "ap_folds_better/worse"]].to_string(float_format=f4),
        "",
        "   Pooled OOF AP (secondary: pooling mixes five fold models' score scales)",
        pooled[["ap_proba_pooled", "ap_decision_pooled"]].to_string(float_format=f4),
        "",
        f"3) EQUAL ALERT BUDGET - {TOTAL_ALERT_BUDGET} alerts split across folds as {a['fold_budgets']}",
        "   Rank by decision score within each fold; ties broken by ascending row_id so each fold flags",
        "   exactly its budget. If the cutoff lands inside a tied group, only part of that group is flagged",
        "   (tied_at_cutoff vs tied_included below).",
        a["budget"].pivot(index="config", columns="fold", values="tp").add_prefix("TP_f").to_string(),
        "",
        paired[["budget_alerts_total", "budget_tp_total", "tp_diff_per_fold", "tp_diff_mean", "tp_diff_std",
                "frauds_only_this", "frauds_only_baseline", "binomial_p_two_sided"]].to_string(float_format=f4),
        "   frauds_only_*: frauds flagged by one model but not the baseline (pooled over folds).",
        "   binomial_p: exact two-sided test that those discordant frauds split 50/50. Descriptive only:",
        "   ignores fold structure and the fixed-budget coupling; not proof of superiority.",
        "",
        "   Cutoff ties per fold",
        a["budget"][["config", "fold", "alerts", "cutoff_score", "tied_at_cutoff", "tied_included"]].to_string(index=False),
        "",
        "4) PROBABILITY QUALITY (predict_proba; no calibrator fitted)",
        "   A matching mean probability does not imply calibration; see the per-bin table.",
        pooled[["brier", "log_loss", "mean_predicted", "ece_binned"]].to_string(float_format=lambda x: f"{x:.5f}"),
        "",
    ]
    for name, rel in a["reliability"].items():
        lines += [f"   Reliability: {name}",
                  rel.to_string(float_format=lambda x: f"{x:.4f}"), ""]
    lines += ["OOF score SHA-256 (identical across two independent runs)",
              *[f"  {k}: {v}" for k, v in hashes.items()]]
    return "\n".join(lines)


def plot(a: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    names = list(a["paired"].index)
    for f in sorted(a["ranking"]["fold"].unique()):
        rk = a["ranking"][a["ranking"]["fold"] == f].set_index("config").loc[names]
        axes[0].plot(names, rk["ap_decision"], marker="o", alpha=0.7, label=f"fold {f}")
        bd = a["budget"][a["budget"]["fold"] == f].set_index("config").loc[names]
        axes[1].plot(names, bd["tp"], marker="o", alpha=0.7, label=f"fold {f}")
    axes[0].set_title("Per-fold AP (decision scores)")
    axes[1].set_title(f"Per-fold frauds caught at equal budget ({a['fold_budgets']})")
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7)

    ax = axes[2]
    for name, rel in a["reliability"].items():
        r = rel[rel["rows"] > 0]
        ax.plot(r["mean_predicted"], r["observed_rate"], marker="o", label=name)
    ax.plot([1e-4, 1], [1e-4, 1], color="grey", ls="--", label="perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(0, 1.5)  # rates are non-negative; symlog would otherwise show negative ticks
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraud rate (bin)")
    ax.set_title("Reliability (OOF, binned)")
    ax.legend(fontsize=7)
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
        print(f"ERROR: split verification failed, not running: {exc}", file=sys.stderr)
        return 1

    folds = make_folds(X, y, groups)
    fold_summary = verify_folds(folds, y, groups)  # coverage, group separation, fraud in every fold
    results = run_all(X, y, folds)

    repeat = run_all(X, y, make_folds(X, y, groups))
    for name in results:
        for kind in ("proba", "decision"):
            if not np.array_equal(results[name][kind], repeat[name][kind]):
                raise AssertionError(f"{name}: OOF {kind} differs between two identical runs.")
    hashes = {f"{n} {k}": hashlib.sha256(r[k].tobytes()).hexdigest()
              for n, r in results.items() for k in ("proba", "decision")}

    earlier_match = {}
    for name, (rel_path, keys) in EARLIER_OOF_HASHES.items():
        path = PROJECT_ROOT / rel_path
        if path.is_file():
            ref = json.loads(path.read_text())
            for k in keys:
                ref = ref[k]
            earlier_match[name] = ref == hashes[f"{name} proba"]
    if not all(earlier_match.values()):
        raise AssertionError(f"OOF probabilities do not reproduce earlier phases: {earlier_match}")

    y_arr = y.to_numpy()
    analysis = analyse(results, y_arr, folds, groups.to_numpy(), X.index.to_numpy())
    conv = {n: r["convergence_warnings"] for n, r in results.items()}

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(analysis, results, fold_summary, y_arr, hashes, earlier_match, conv)
    (args.report_dir / "experiment_report.txt").write_text(report + "\n")
    analysis["saturation"].to_csv(args.report_dir / "saturation.csv")
    analysis["ranking"].to_csv(args.report_dir / "ranking_by_fold.csv", index=False, float_format="%.6f")
    analysis["budget"].to_csv(args.report_dir / "budget_by_fold.csv", index=False, float_format="%.6f")
    analysis["paired"].to_csv(args.report_dir / "paired_summary.csv", float_format="%.6f")
    pd.concat(analysis["reliability"], names=["config"]).to_csv(args.report_dir / "reliability.csv", float_format="%.6f")
    pd.DataFrame({n: r["pooled"] for n, r in results.items()}).T.to_csv(args.report_dir / "pooled_metrics.csv",
                                                                        float_format="%.6f")
    plot(analysis, args.report_dir / "experiment_plot.png")
    (args.report_dir / "experiment_config.json").write_text(json.dumps({
        "configs": {k: (v if not isinstance(v, dict) else {str(c): w for c, w in v.items()}) for k, v in CONFIGS.items()},
        "C": 1.0,
        "cv": f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={RANDOM_STATE})",
        "total_alert_budget": TOTAL_ALERT_BUDGET,
        "fold_budgets": analysis["fold_budgets"],
        "ranking_score": "decision_function",
        "tie_policy": "within each fold: sort by decision score desc, then row_id asc; take exactly the fold budget",
        "calibrator_fitted": False,
        "oof_sha256": hashes,
        "earlier_phase_oof_match": earlier_match,
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
