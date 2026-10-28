"""Validate and explore the ULB credit-card fraud dataset.

Read-only: the raw CSV is never modified. Outputs (plots and a text summary)
are written to an ignored reports directory.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --data-path data/raw/creditcard.csv --output-dir reports/eda
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"

# Expected schema, taken from the dataset's published description (Kaggle mlg-ulb/creditcardfraud).
TARGET = "Class"
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", TARGET]
EXPECTED_ROWS = 284_807
EXPECTED_LABELS = {0, 1}
LABEL_NAMES = {0: "Legitimate", 1: "Fraud"}


class DatasetValidationError(Exception):
    """Raised when the dataset does not match the expected schema."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it at data/raw/creditcard.csv."
        )
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise DatasetValidationError(f"Could not parse {path} as CSV: {exc}") from exc


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Raise on schema errors that make analysis meaningless; return softer warnings."""
    warnings: list[str] = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or unexpected:
        raise DatasetValidationError(
            f"Column mismatch. Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
        )

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DatasetValidationError(f"Expected all columns to be numeric, but got: {non_numeric}")

    labels = set(df[TARGET].dropna().unique())
    if not labels <= EXPECTED_LABELS:
        raise DatasetValidationError(f"Target '{TARGET}' has unexpected labels: {sorted(labels)}")

    if len(df) != EXPECTED_ROWS:
        warnings.append(f"Expected {EXPECTED_ROWS:,} rows but found {len(df):,}.")
    if (df["Amount"] < 0).any():
        warnings.append("Found negative values in 'Amount'.")
    if not df["Time"].is_monotonic_increasing:
        warnings.append("'Time' is not sorted in increasing order.")

    return warnings


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    summary = pd.DataFrame({"count": counts, "percent": counts / len(df) * 100})
    summary.index = summary.index.map(LABEL_NAMES)
    return summary


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    features = [c for c in df.columns if c != TARGET]
    dup_mask = df.duplicated(keep="first")  # rows that repeat an earlier row exactly
    # Same features appearing with both labels would mean contradictory labels.
    labels_per_feature_row = df[df.duplicated(subset=features, keep=False)].groupby(features)[TARGET].nunique()
    return {
        "duplicate_rows": int(dup_mask.sum()),
        "duplicate_legitimate": int((dup_mask & (df[TARGET] == 0)).sum()),
        "duplicate_fraud": int((dup_mask & (df[TARGET] == 1)).sum()),
        "rows_in_any_duplicate_group": int(df.duplicated(keep=False).sum()),
        "feature_groups_with_conflicting_labels": int((labels_per_feature_row > 1).sum()),
    }


def plot_class_distribution(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(summary.index, summary["count"], color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")  # linear scale would make the fraud bar invisible
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Class distribution")
    for bar, (count, pct) in zip(bars, summary[["count", "percent"]].itertuples(index=False)):
        ax.annotate(f"{count:,}\n({pct:.3f}%)", (bar.get_x() + bar.get_width() / 2, count),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(top=summary["count"].max() * 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, out_path: Path) -> None:
    # log1p compresses the long right tail (a few very large amounts) and handles Amount == 0.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, np.log1p(df["Amount"].max()), 50)
    for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        subset = np.log1p(df.loc[df[TARGET] == label, "Amount"])
        # density=True so the tiny fraud class is comparable in shape to the large legit class.
        axes[0].hist(subset, bins=bins, density=True, alpha=0.6, color=color, label=LABEL_NAMES[label])
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution by class (normalised)")
    axes[0].legend()

    df.boxplot(column="Amount", by=TARGET, ax=axes[1], showfliers=False)
    axes[1].set_xticklabels([LABEL_NAMES[0], LABEL_NAMES[1]])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Amount")
    axes[1].set_title("Amount by class (outliers hidden)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_time_distribution(df: pd.DataFrame, out_path: Path) -> None:
    hours = df["Time"] / 3600  # seconds since first transaction -> hours (dataset spans ~2 days)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, np.ceil(hours.max()) + 1, 1)
    for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        ax.hist(hours[df[TARGET] == label], bins=bins, density=True, alpha=0.6,
                color=color, label=LABEL_NAMES[label])
    ax.set_xlabel("Hours since first transaction")
    ax.set_ylabel("Density")
    ax.set_title("Transaction timing by class (normalised)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_report(df: pd.DataFrame, path: Path, file_hash: str, warnings: list[str]) -> str:
    classes = class_summary(df)
    baseline_accuracy = classes.loc["Legitimate", "count"] / len(df)
    dups = duplicate_summary(df)
    missing_total = int(df.isna().sum().sum())
    amount_by_class = df.groupby(df[TARGET].map(LABEL_NAMES))["Amount"].describe()

    lines = [
        "CREDIT CARD FRAUD DATASET - VALIDATION & EDA SUMMARY",
        "=" * 52,
        f"File:    {path}",
        f"SHA-256: {file_hash}",
        "",
        "Schema",
        f"  Shape:          {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"  Columns:        {', '.join(df.columns)}",
        f"  Dtypes:         {df.dtypes.value_counts().to_dict()}",
        f"  Missing values: {missing_total}",
        f"  Target labels:  {sorted(df[TARGET].unique().tolist())}",
        f"  Warnings:       {warnings or 'none'}",
        "",
        "Class balance",
        classes.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        f"Always-predict-legitimate baseline accuracy: {baseline_accuracy:.4%}",
        f"  -> catches 0 of {classes.loc['Fraud', 'count']:,} frauds (recall = 0%).",
        "",
        "Duplicates (exact match on all 31 columns; NOT removed)",
        *[f"  {k}: {v:,}" for k, v in dups.items()],
        "",
        "Amount by class",
        amount_by_class.to_string(float_format=lambda x: f"{x:,.2f}"),
        "",
        f"Time span: {df['Time'].max() / 3600:.1f} hours",
        "",
        "Feature summary",
        df.describe().T.to_string(float_format=lambda x: f"{x:,.3f}"),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        hash_before = sha256_of(args.data_path) if args.data_path.is_file() else ""
        df = load_dataset(args.data_path)
        warnings = validate_dataset(df)
    except (FileNotFoundError, DatasetValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(df, args.data_path, hash_before, warnings)
    (args.output_dir / "summary.txt").write_text(report + "\n")
    plot_class_distribution(class_summary(df), args.output_dir / "class_distribution.png")
    plot_amount_distribution(df, args.output_dir / "amount_distribution.png")
    plot_time_distribution(df, args.output_dir / "time_distribution.png")

    if sha256_of(args.data_path) != hash_before:
        print("ERROR: raw dataset changed during inspection.", file=sys.stderr)
        return 1

    print(report)
    print(f"\nRaw file unchanged (SHA-256 verified). Outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
