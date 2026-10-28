"""Create a leakage-safe, reproducible 80/20 train/test split.

Duplicate strategy: every row is retained. Rows with identical feature values
(all columns except the target) share a group id, and whole groups are assigned
to one split, so an identical feature row can never appear in both train and test.
The split is stratified on the target at group level (StratifiedGroupKFold),
which approximately preserves the fraud rate; actual proportions are reported.

Only split assignments are saved (row ids into the raw CSV), not copies of the
data. No learned transformation (scaling, resampling, feature selection) is
fitted here; that happens later, on the training split only.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-path data/raw/creditcard.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Reuse the Phase 2B loading/validation so the schema is defined in one place.
from inspect_dataset import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    TARGET,
    DatasetValidationError,
    load_dataset,
    sha256_of,
    validate_dataset,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
TEST_FOLDS = 5  # 1 of 5 folds held out -> ~20% test


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    if TARGET in X.columns:
        raise DatasetValidationError(f"Target '{TARGET}' leaked into the feature columns.")
    return X, y


def duplicate_groups(X: pd.DataFrame) -> pd.Series:
    """Assign the same integer id to rows whose feature values are exactly identical."""
    return X.groupby(list(X.columns), sort=False, dropna=False).ngroup().rename("group_id")


def check_group_labels(groups: pd.Series, y: pd.Series) -> None:
    """A group with mixed labels cannot be stratified cleanly and signals label noise."""
    conflicting = int((y.groupby(groups).nunique() > 1).sum())
    if conflicting:
        raise DatasetValidationError(
            f"{conflicting} identical feature rows have conflicting '{TARGET}' labels; "
            "investigate before splitting."
        )


def group_stratified_split(y: pd.Series, groups: pd.Series) -> pd.Series:
    splitter = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(X=y.to_frame(), y=y, groups=groups))
    split = pd.Series("train", index=y.index, name="split")
    split.iloc[test_idx] = "test"
    return split


def verify_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series, split: pd.Series) -> None:
    """Hard checks: fail loudly rather than save a leaky or incomplete split."""
    is_test = split == "test"
    if len(split) != len(X) or split.isna().any():
        raise AssertionError("Every row must be assigned to exactly one split.")
    shared_groups = set(groups[is_test]) & set(groups[~is_test])
    if shared_groups:
        raise AssertionError(f"{len(shared_groups)} duplicate groups cross the split boundary.")
    # Independent of group ids: compare actual feature values across splits.
    overlap = X[~is_test].drop_duplicates().merge(X[is_test].drop_duplicates(), how="inner")
    if len(overlap):
        raise AssertionError(f"{len(overlap)} identical feature rows appear in both train and test.")
    for name in ("train", "test"):
        if y[split == name].nunique() != 2:
            raise AssertionError(f"The {name} split must contain both classes.")


def split_report(y: pd.Series, groups: pd.Series, split: pd.Series) -> pd.DataFrame:
    rows = []
    for name in ("train", "test"):
        mask = split == name
        n, fraud = int(mask.sum()), int(y[mask].sum())
        rows.append({
            "split": name,
            "rows": n,
            "share_of_rows_%": round(n / len(y) * 100, 3),
            "fraud": fraud,
            "fraud_%": round(fraud / n * 100, 4),
            "share_of_all_fraud_%": round(fraud / y.sum() * 100, 2),
            "duplicate_rows_within_split": int(groups[mask].duplicated().sum()),
        })
    return pd.DataFrame(rows).set_index("split")


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
        validate_dataset(df)
        X, y = split_features_target(df)
        groups = duplicate_groups(X)
        check_group_labels(groups, y)
    except (FileNotFoundError, DatasetValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    split = group_stratified_split(y, groups)
    verify_split(X, y, groups, split)
    report = split_report(y, groups, split)

    if sha256_of(args.data_path) != hash_before:
        print("ERROR: raw dataset changed during preparation.", file=sys.stderr)
        return 1

    # row_id is the 0-based row position in the raw CSV (excluding the header).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame({"row_id": df.index, "group_id": groups, "split": split})
    assignments.to_csv(args.output_dir / "split_assignments.csv", index=False)
    metadata = {
        "source_file": str(args.data_path.relative_to(PROJECT_ROOT) if args.data_path.is_relative_to(PROJECT_ROOT) else args.data_path),
        "source_sha256": hash_before,
        "target": TARGET,
        "feature_columns": list(X.columns),
        "strategy": "retain all rows; group identical feature rows; StratifiedGroupKFold, 1 of 5 folds as test",
        "random_state": RANDOM_STATE,
        "n_rows": len(df),
        "n_unique_feature_rows": int(groups.nunique()),
        "n_duplicate_groups": int((groups.value_counts() > 1).sum()),
        "splits": report.to_dict(orient="index"),
        "verified_no_feature_row_overlap": True,
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Rows: {len(df):,} | unique feature rows: {metadata['n_unique_feature_rows']:,} | "
          f"duplicate groups: {metadata['n_duplicate_groups']:,}")
    print(f"Overall fraud rate: {y.mean() * 100:.4f}%\n")
    print(report.to_string())
    print("\nVerified: no identical feature row appears in both train and test.")
    print(f"Raw file unchanged (SHA-256 verified). Outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
