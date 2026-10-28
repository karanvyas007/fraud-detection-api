"""Inference for saved fraud-detection pipelines.

Inference only: nothing here fits a scaler or model, reads labels or touches the
held-out test set. Models are loaded from saved joblib artifacts produced by the
training scripts; creating artifacts (including the w=10 challenger) belongs to a
later training phase.

Score semantics
---------------
- ``fraud_score``: the pipeline's ``predict_proba`` output for the fraud class
  (Class = 1), a number in [0, 1]. Higher means "more fraud-like" relative to other
  transactions. It is NOT a calibrated real-world fraud probability for any current
  artifact: the unweighted baseline was only checked out-of-fold and is imperfectly
  calibrated, and class-weighted models are systematically inflated.
- ``flagged``: the decision ``fraud_score >= threshold``. The threshold is always a
  required argument; thresholds are model-specific and do not transfer between a
  weighted and an unweighted model.

Reproducibility: the same input in the same batch gives bit-identical scores. Scoring a
row alone vs inside a larger batch can differ by floating-point rounding (~1e-15), because
the underlying matrix routines sum in a different order; a flag can only differ if the
score lies within that rounding of the threshold.

Security: joblib files are pickles and can execute code when loaded. Only load
artifacts you created; pass ``expected_sha256`` to pin a known file.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
TARGET = "Class"
NON_NEGATIVE_FEATURES = ("Time", "Amount")

# Named artifacts. The challenger file does not exist yet; it is created by a later training phase.
NAMED_ARTIFACTS = {
    "baseline": "final_logreg.joblib",
    "challenger": "challenger_w10_logreg.joblib",
}


class ArtifactError(Exception):
    """The model artifact is missing, malformed or inconsistent."""


class InvalidTransactionError(ValueError):
    """Input transactions do not match the model's expected features."""


@dataclass(frozen=True)
class ModelArtifact:
    """A loaded, already-fitted pipeline plus the metadata needed to use it safely."""

    pipeline: Any
    feature_columns: tuple[str, ...]
    positive_class: int
    source_path: Path
    sha256: str
    class_weight: Any = None
    recorded_thresholds: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_class_weighted(self) -> bool:
        return self.class_weight is not None

    @property
    def score_semantics(self) -> str:
        if self.is_class_weighted:
            return ("Uncalibrated ranking score from a class-weighted model (class_weight="
                    f"{self.class_weight!r}). Inflated relative to real fraud rates; not a probability.")
        return ("Model output of an unweighted Logistic Regression, in [0, 1]. Usable for ranking; "
                "not validated as a calibrated real-world fraud probability.")


@dataclass(frozen=True)
class Prediction:
    fraud_score: float
    flagged: bool
    threshold: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(path: str | Path, *, expected_sha256: str | None = None) -> ModelArtifact:
    """Load a saved pipeline artifact and check it is internally consistent."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Model artifact not found: {path}")
    sha = _sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ArtifactError(f"Checksum mismatch for {path}: expected {expected_sha256}, got {sha}")

    bundle = joblib.load(path)
    if not isinstance(bundle, Mapping):
        raise ArtifactError(f"{path} does not contain an artifact dictionary.")
    missing = [k for k in ("model", "feature_columns", "positive_class") if k not in bundle]
    if missing:
        raise ArtifactError(f"{path} is missing required keys: {missing}")

    pipeline = bundle["model"]
    features = tuple(bundle["feature_columns"])
    positive_class = bundle["positive_class"]
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
        raise ArtifactError(f"{path}: model is not a fitted classifier with predict_proba.")
    if positive_class not in list(pipeline.classes_):
        raise ArtifactError(f"{path}: positive class {positive_class!r} not in model classes {list(pipeline.classes_)}.")
    if TARGET in features:
        raise ArtifactError(f"{path}: target column '{TARGET}' is listed as a feature.")
    if len(set(features)) != len(features):
        raise ArtifactError(f"{path}: feature_columns contains duplicates.")
    fitted_names = getattr(pipeline, "feature_names_in_", None)
    if fitted_names is not None and tuple(fitted_names) != features:
        raise ArtifactError(f"{path}: feature_columns do not match the order the model was fitted with.")

    final_step = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    thresholds = bundle.get("thresholds")
    if thresholds is None and "threshold" in bundle:  # Phase 3A format
        thresholds = {"baseline": bundle["threshold"]}
    return ModelArtifact(
        pipeline=pipeline,
        feature_columns=features,
        positive_class=positive_class,
        source_path=path,
        sha256=sha,
        class_weight=getattr(final_step, "class_weight", None),
        recorded_thresholds=dict(thresholds or {}),
        metadata={k: v for k, v in bundle.items() if k != "model"},
    )


def load_named_artifact(name: str, models_dir: str | Path = DEFAULT_MODELS_DIR, **kwargs) -> ModelArtifact:
    """Load 'baseline' or 'challenger' from the models directory. Never trains a missing model."""
    if name not in NAMED_ARTIFACTS:
        raise ArtifactError(f"Unknown artifact name {name!r}; expected one of {sorted(NAMED_ARTIFACTS)}.")
    path = Path(models_dir) / NAMED_ARTIFACTS[name]
    if not path.is_file():
        raise ArtifactError(
            f"The {name!r} artifact does not exist at {path}. Inference never trains models; "
            "creating this artifact belongs to a later training phase."
        )
    return load_artifact(path, **kwargs)


def _validate_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError(f"threshold must be a real number, got {threshold!r}.")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}.")
    return threshold


def _check_value(value: Any, column: str, row: int) -> float:
    where = f"row {row}, feature '{column}'"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InvalidTransactionError(f"{where}: expected a number, got {type(value).__name__} {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidTransactionError(f"{where}: value must be finite, got {value}.")
    if column in NON_NEGATIVE_FEATURES and value < 0:
        raise InvalidTransactionError(f"{where}: must be non-negative, got {value}.")
    return value


def _check_columns(columns: Sequence[str], features: tuple[str, ...], row_label: str) -> None:
    present = set(columns)
    missing = [c for c in features if c not in present]
    extra = sorted(str(c) for c in present - set(features))
    if TARGET in present:
        raise InvalidTransactionError(f"{row_label}: '{TARGET}' is the label and must not be supplied for inference.")
    if missing or extra:
        raise InvalidTransactionError(f"{row_label}: missing features {missing or 'none'}; unexpected fields {extra or 'none'}.")


def to_feature_frame(transactions: Any, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Validate input and return a float64 DataFrame in the exact training feature order.

    Accepts one transaction (a mapping of feature -> number), a sequence of such mappings,
    or a pandas DataFrame. Never modifies the input.
    """
    features = tuple(feature_columns)
    if isinstance(transactions, pd.DataFrame):
        _check_columns(list(transactions.columns), features, "input frame")
        records = transactions[list(features)].to_dict(orient="records")
    elif isinstance(transactions, Mapping):
        records = [transactions]
    elif isinstance(transactions, Sequence) and not isinstance(transactions, (str, bytes)):
        records = list(transactions)
    else:
        raise InvalidTransactionError(
            f"Expected a mapping, a sequence of mappings or a DataFrame, got {type(transactions).__name__}.")
    if not records:
        raise InvalidTransactionError("No transactions supplied.")

    rows = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise InvalidTransactionError(f"row {i}: expected a mapping of feature -> value, got {type(record).__name__}.")
        _check_columns(list(record.keys()), features, f"row {i}")
        rows.append([_check_value(record[c], c, i) for c in features])
    return pd.DataFrame(rows, columns=list(features), dtype="float64")


def fraud_scores(artifact: ModelArtifact, transactions: Any) -> np.ndarray:
    """Return the fraud-class score for each transaction (see module docstring for semantics)."""
    frame = to_feature_frame(transactions, artifact.feature_columns)
    classes = list(artifact.pipeline.classes_)
    return artifact.pipeline.predict_proba(frame)[:, classes.index(artifact.positive_class)]


def predict(artifact: ModelArtifact, transactions: Any, *, threshold: float) -> list[Prediction]:
    """Score one or many transactions and flag those with fraud_score >= threshold.

    Always returns a list (length 1 for a single transaction), in input order.
    """
    threshold = _validate_threshold(threshold)
    scores = fraud_scores(artifact, transactions)
    return [Prediction(fraud_score=float(s), flagged=bool(s >= threshold), threshold=threshold) for s in scores]


def predict_one(artifact: ModelArtifact, transaction: Mapping[str, Any], *, threshold: float) -> Prediction:
    """Convenience wrapper for a single transaction."""
    if not isinstance(transaction, Mapping):
        raise InvalidTransactionError(f"Expected one transaction mapping, got {type(transaction).__name__}.")
    return predict(artifact, transaction, threshold=threshold)[0]
