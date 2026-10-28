"""Thin FastAPI layer over fraud_detection.predict.

All scoring, feature ordering and threshold checks are done by fraud_detection.predict;
this module only adds HTTP request/response schemas and error mapping.

- The unweighted baseline artifact is loaded explicitly when the app is created.
  If loading fails, app creation fails; there is no fallback model and no training.
- ``fraud_score`` is a model score in [0, 1] for ranking. It is not a calibrated
  real-world fraud probability.
- ``threshold`` is optional per request. The API default (DEFAULT_THRESHOLD = 0.5) is the
  conventional decision rule, not a tuned fraud operating point. The 0.09 threshold
  found in training-only cross-validation is listed by /model-info as an experimental
  option that callers must request explicitly. Neither is a production business decision.

Run locally (from the project root):
    uvicorn fraud_detection.api:create_app --factory --app-dir src
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud_detection.predict import (
    DEFAULT_MODELS_DIR,
    ArtifactError,
    InvalidTransactionError,
    ModelArtifact,
    load_artifact,
    load_named_artifact,
    predict,
)

MODEL_NAME = "baseline"
FEATURE_COLUMNS = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
NON_NEGATIVE_FEATURES = ("Time", "Amount")
DEFAULT_THRESHOLD = 0.5
MAX_BATCH_SIZE = 1000
THRESHOLD_RULE = "flagged = fraud_score >= threshold"
DEFAULT_THRESHOLD_NOTE = (
    "API default 0.5: the conventional decision rule, not tuned for fraud costs. "
    "Override per request; choosing an operating point is a business decision."
)
# Notes for thresholds recorded in the artifact metadata (keyed by their recorded name).
RECORDED_THRESHOLD_NOTES = {
    "baseline": "Conventional default decision rule; not tuned for fraud costs.",
    "provisional": (
        "Experimental operating point: maximised out-of-fold F1 on training data for this "
        "unweighted model. Not a production business decision; does not transfer to other models."
    ),
}


def _feature_field(name: str) -> tuple[Any, Any]:
    # strict: reject strings and booleans (ints are still accepted); allow_inf_nan: reject NaN/Infinity.
    constraints: dict[str, Any] = {"strict": True, "allow_inf_nan": False}
    if name in NON_NEGATIVE_FEATURES:
        constraints["ge"] = 0
    return Annotated[float, Field(**constraints)], ...


Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    __doc__="One card transaction: Time, V1-V28 (anonymised PCA features) and Amount. No other fields.",
    **{name: _feature_field(name) for name in FEATURE_COLUMNS},
)

Threshold = Annotated[
    float,
    Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False,
          description=f"Decision threshold in [0, 1]; {THRESHOLD_RULE}. {DEFAULT_THRESHOLD_NOTE}"),
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: Transaction  # type: ignore[valid-type]
    threshold: Threshold = DEFAULT_THRESHOLD


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: Annotated[list[Transaction], Field(min_length=1, max_length=MAX_BATCH_SIZE)]  # type: ignore[valid-type]
    threshold: Threshold = DEFAULT_THRESHOLD


class PredictionResult(BaseModel):
    index: int = Field(description="Position of the transaction in the request.")
    fraud_score: float = Field(description="Model score in [0, 1] for ranking; not a calibrated probability.")
    flagged: bool = Field(description=THRESHOLD_RULE)


class _ScoredResponse(BaseModel):
    model_name: str
    threshold: float
    threshold_source: Literal["request", "api_default"]
    score_semantics: str


class PredictResponse(_ScoredResponse):
    prediction: PredictionResult


class BatchPredictResponse(_ScoredResponse):
    count: int
    predictions: list[PredictionResult]


class ThresholdInfo(BaseModel):
    name: str
    value: float
    note: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    class_weighted: bool
    positive_class: int
    feature_count: int
    feature_names: list[str]
    score_semantics: str
    threshold_rule: str
    default_threshold: float
    default_threshold_note: str
    documented_thresholds: list[ThresholdInfo]
    max_batch_size: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ErrorResponse(BaseModel):
    detail: Any


def _load_baseline(artifact_path: str | Path | None, models_dir: str | Path, expected_sha256: str | None) -> ModelArtifact:
    if artifact_path is None:
        artifact = load_named_artifact(MODEL_NAME, models_dir, expected_sha256=expected_sha256)
    else:
        artifact = load_artifact(artifact_path, expected_sha256=expected_sha256)
    if artifact.is_class_weighted:
        raise ArtifactError("This API serves only the unweighted baseline; the artifact is class-weighted.")
    if artifact.feature_columns != FEATURE_COLUMNS:
        raise ArtifactError("The artifact's feature columns do not match the API request schema.")
    return artifact


def create_app(
    artifact_path: str | Path | None = None,
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    expected_sha256: str | None = None,
) -> FastAPI:
    """Build the app around an explicitly loaded baseline artifact. Raises ArtifactError on failure."""
    artifact = _load_baseline(artifact_path, models_dir, expected_sha256)

    app = FastAPI(
        title="Credit Card Fraud Detection API (experimental)",
        version="0.1.0",
        description=(
            "Scores transactions with an unweighted Logistic Regression baseline. "
            "fraud_score is a ranking score in [0, 1], not a calibrated fraud probability. "
            "Thresholds are experimental operating points, not business decisions. Not production-ready."
        ),
    )
    app.state.artifact = artifact
    error_responses = {422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep location, message and type only: no echoed input, URLs or internal context.
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(InvalidTransactionError)
    async def _invalid_transaction(request: Request, exc: InvalidTransactionError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    def scored_fields(request: PredictRequest | BatchPredictRequest) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "threshold": request.threshold,
            "threshold_source": "request" if "threshold" in request.model_fields_set else "api_default",
            "score_semantics": artifact.score_semantics,
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=MODEL_NAME,
            model_type=str(artifact.metadata.get("model_type", type(artifact.pipeline).__name__)),
            class_weighted=artifact.is_class_weighted,
            positive_class=artifact.positive_class,
            feature_count=len(artifact.feature_columns),
            feature_names=list(artifact.feature_columns),
            score_semantics=artifact.score_semantics,
            threshold_rule=THRESHOLD_RULE,
            default_threshold=DEFAULT_THRESHOLD,
            default_threshold_note=DEFAULT_THRESHOLD_NOTE,
            documented_thresholds=[
                ThresholdInfo(name=name, value=float(value),
                              note=RECORDED_THRESHOLD_NOTES.get(name, "Recorded with the model artifact."))
                for name, value in artifact.recorded_thresholds.items()
            ],
            max_batch_size=MAX_BATCH_SIZE,
        )

    @app.post("/predict", response_model=PredictResponse, responses=error_responses)
    def predict_single(request: PredictRequest) -> PredictResponse:
        result = predict(artifact, request.transaction.model_dump(), threshold=request.threshold)[0]
        return PredictResponse(
            **scored_fields(request),
            prediction=PredictionResult(index=0, fraud_score=result.fraud_score, flagged=result.flagged),
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse, responses=error_responses)
    def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
        results = predict(artifact, [t.model_dump() for t in request.transactions], threshold=request.threshold)
        return BatchPredictResponse(
            **scored_fields(request),
            count=len(results),
            predictions=[PredictionResult(index=i, fraud_score=r.fraud_score, flagged=r.flagged)
                         for i, r in enumerate(results)],
        )

    return app
