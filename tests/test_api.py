"""Tests for the FastAPI layer (fraud_detection.api).

Uses a small synthetic pipeline saved to a temporary directory. A separate
integration test uses the real baseline artifact and skips if it is absent.
The whole module is skipped if the API dependencies are not installed.

Run:  python -m unittest discover -s tests -v
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as exc:  # starlette raises RuntimeError when httpx is missing
    raise unittest.SkipTest(f"API dependencies not installed ({exc}); run `pip install -r requirements.txt`.")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fraud_detection.api import DEFAULT_THRESHOLD, MAX_BATCH_SIZE, create_app  # noqa: E402
from fraud_detection.predict import ArtifactError, load_artifact, predict  # noqa: E402
from test_predict import FEATURES, make_bundle, make_transaction  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_MODELS_DIR = PROJECT_ROOT / "models"
SCORED_KEYS = {"model_name", "threshold", "threshold_source", "score_semantics"}


def snapshot(directory: Path) -> dict:
    """Name -> (size, mtime) for every file, to detect created or modified model files."""
    return {str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()} if directory.is_dir() else {}


class ApiFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "final_logreg.joblib"
        joblib.dump(make_bundle(), self.path)
        self.artifact = load_artifact(self.path)
        self.client = TestClient(create_app(self.path))
        self.tx = make_transaction()
        self.batch = [make_transaction(s) for s in range(1, 11)]

    def tearDown(self):
        self.tmp.cleanup()

    def post_raw(self, url: str, body: str):
        return self.client.post(url, content=body, headers={"Content-Type": "application/json"})

    def assert_clean_error(self, response, status=422):
        self.assertEqual(response.status_code, status, response.text)
        text = response.text
        for leak in ("Traceback", str(self.dir), ".joblib", '"input"', '"url"'):
            self.assertNotIn(leak, text)
        return response.json()["detail"]


class TestHealthAndModelInfo(ApiFixture):
    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_model_info_contents(self):
        info = self.client.get("/model-info").json()
        self.assertEqual(info["model_name"], "baseline")
        self.assertEqual(info["feature_count"], 30)
        self.assertEqual(info["feature_names"], FEATURES)
        self.assertFalse(info["class_weighted"])
        self.assertEqual(info["positive_class"], 1)
        self.assertEqual(info["default_threshold"], DEFAULT_THRESHOLD)
        self.assertEqual(info["max_batch_size"], MAX_BATCH_SIZE)
        self.assertIn("not validated as a calibrated", info["score_semantics"])
        documented = {t["name"]: t for t in info["documented_thresholds"]}
        self.assertEqual(documented["provisional"]["value"], 0.09)
        self.assertIn("Experimental", documented["provisional"]["note"])
        self.assertIn("not tuned", info["default_threshold_note"])

    def test_model_info_hides_internal_details(self):
        text = self.client.get("/model-info").text
        for leak in (str(self.dir), ".joblib", self.artifact.sha256, "checksum", "sha256", "versions"):
            self.assertNotIn(leak, text)

    def test_openapi_documents_default_threshold_and_semantics(self):
        schema = self.client.get("/openapi.json").json()
        threshold = schema["components"]["schemas"]["PredictRequest"]["properties"]["threshold"]
        self.assertEqual(threshold["default"], DEFAULT_THRESHOLD)
        self.assertIn("not a calibrated", schema["info"]["description"])


class TestPredict(ApiFixture):
    def test_valid_single_uses_api_default(self):
        r = self.client.post("/predict", json={"transaction": self.tx})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(set(body), SCORED_KEYS | {"prediction"})
        self.assertEqual(set(body["prediction"]), {"index", "fraud_score", "flagged"})
        self.assertEqual(body["threshold"], DEFAULT_THRESHOLD)
        self.assertEqual(body["threshold_source"], "api_default")
        self.assertIn("not validated as a calibrated", body["score_semantics"])
        expected = predict(self.artifact, self.tx, threshold=DEFAULT_THRESHOLD)[0]
        self.assertEqual(body["prediction"], {"index": 0, "fraud_score": expected.fraud_score,
                                              "flagged": expected.flagged})

    def test_explicit_threshold(self):
        for threshold in (0.09, DEFAULT_THRESHOLD, 0, 1):
            with self.subTest(threshold=threshold):
                body = self.client.post("/predict", json={"transaction": self.tx, "threshold": threshold}).json()
                self.assertEqual(body["threshold_source"], "request")
                self.assertEqual(body["threshold"], threshold)
                p = body["prediction"]
                self.assertEqual(p["flagged"], p["fraud_score"] >= threshold)

    def test_valid_batch_preserves_order(self):
        r = self.client.post("/predict/batch", json={"transactions": self.batch, "threshold": 0.09})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(set(body), SCORED_KEYS | {"count", "predictions"})
        self.assertEqual(body["count"], len(self.batch))
        self.assertEqual([p["index"] for p in body["predictions"]], list(range(len(self.batch))))
        expected = predict(self.artifact, self.batch, threshold=0.09)
        self.assertEqual([p["fraud_score"] for p in body["predictions"]], [e.fraud_score for e in expected])
        self.assertEqual([p["flagged"] for p in body["predictions"]], [e.flagged for e in expected])

    def test_batch_order_follows_input(self):
        forward = self.client.post("/predict/batch", json={"transactions": self.batch}).json()["predictions"]
        backward = self.client.post("/predict/batch", json={"transactions": self.batch[::-1]}).json()["predictions"]
        for f, b in zip(forward, reversed(backward)):
            self.assertAlmostEqual(f["fraud_score"], b["fraud_score"], delta=1e-12)

    def test_feature_key_order_irrelevant(self):
        shuffled = dict(reversed(list(self.tx.items())))
        a = self.client.post("/predict", json={"transaction": self.tx}).json()["prediction"]
        b = self.client.post("/predict", json={"transaction": shuffled}).json()["prediction"]
        self.assertEqual(a, b)

    def test_integer_feature_values_accepted(self):
        tx = {**self.tx, "Amount": 10, "Time": 0}
        self.assertEqual(self.client.post("/predict", json={"transaction": tx}).status_code, 200)

    def test_requests_are_repeatable(self):
        a = self.client.post("/predict/batch", json={"transactions": self.batch}).json()
        b = self.client.post("/predict/batch", json={"transactions": self.batch}).json()
        self.assertEqual(a, b)


class TestInvalidRequests(ApiFixture):
    def test_missing_feature(self):
        tx = dict(self.tx)
        del tx["V7"]
        detail = self.assert_clean_error(self.client.post("/predict", json={"transaction": tx}))
        self.assertIn({"loc": ["body", "transaction", "V7"], "msg": "Field required", "type": "missing"}, detail)

    def test_extra_feature_and_label(self):
        for extra in ("merchant", "Class"):
            with self.subTest(field=extra):
                detail = self.assert_clean_error(self.client.post("/predict", json={"transaction": {**self.tx, extra: 1}}))
                self.assertEqual(detail[0]["loc"], ["body", "transaction", extra])
                self.assertEqual(detail[0]["type"], "extra_forbidden")

    def test_extra_top_level_field(self):
        self.assert_clean_error(self.client.post("/predict", json={"transaction": self.tx, "model": "challenger"}))

    def test_non_numeric_values(self):
        for bad in ("1.5", "abc", True, None, [1.0], {"x": 1}):
            with self.subTest(value=bad):
                detail = self.assert_clean_error(self.client.post("/predict", json={"transaction": {**self.tx, "V3": bad}}))
                self.assertEqual(detail[0]["loc"], ["body", "transaction", "V3"])

    def test_non_finite_values(self):
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=literal):
                body = json.dumps({"transaction": self.tx}).replace(
                    f'"V3": {json.dumps(self.tx["V3"])}', f'"V3": {literal}')
                self.assertIn(literal, body)
                detail = self.assert_clean_error(self.post_raw("/predict", body))
                self.assertEqual(detail[0]["loc"], ["body", "transaction", "V3"])

    def test_negative_amount_and_time(self):
        for field in ("Amount", "Time"):
            with self.subTest(field=field):
                self.assert_clean_error(self.client.post("/predict", json={"transaction": {**self.tx, field: -1}}))

    def test_malformed_json_and_wrong_shapes(self):
        self.assert_clean_error(self.post_raw("/predict", "{not json"))
        self.assert_clean_error(self.client.post("/predict", json=self.tx))  # missing "transaction" wrapper
        self.assert_clean_error(self.client.post("/predict", json={"transaction": [self.tx]}))
        self.assert_clean_error(self.client.post("/predict/batch", json={"transactions": self.tx}))

    def test_batch_size_limits(self):
        self.assert_clean_error(self.client.post("/predict/batch", json={"transactions": []}))
        too_many = [self.tx] * (MAX_BATCH_SIZE + 1)
        self.assert_clean_error(self.client.post("/predict/batch", json={"transactions": too_many}))

    def test_batch_error_names_row(self):
        batch = [*self.batch[:3], {**self.tx, "V9": "x"}]
        detail = self.assert_clean_error(self.client.post("/predict/batch", json={"transactions": batch}))
        self.assertEqual(detail[0]["loc"], ["body", "transactions", 3, "V9"])

    def test_invalid_thresholds(self):
        for bad in (-0.01, 1.01, "0.5", True, None):
            for url, body in (("/predict", {"transaction": self.tx}), ("/predict/batch", {"transactions": self.batch})):
                with self.subTest(threshold=bad, url=url):
                    detail = self.assert_clean_error(self.client.post(url, json={**body, "threshold": bad}))
                    self.assertEqual(detail[0]["loc"], ["body", "threshold"])
        body = json.dumps({"transaction": self.tx, "threshold": 0.5}).replace('"threshold": 0.5', '"threshold": NaN')
        self.assert_clean_error(self.post_raw("/predict", body))

    def test_unexpected_error_hides_details(self):
        client = TestClient(create_app(self.path), raise_server_exceptions=False)
        with mock.patch("fraud_detection.api.predict", side_effect=RuntimeError(f"secret {self.dir}")):
            r = client.post("/predict", json={"transaction": self.tx})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json(), {"detail": "Internal server error."})
        self.assertNotIn("secret", r.text)


class TestModelLoading(ApiFixture):
    def test_missing_artifact_path(self):
        with self.assertRaisesRegex(ArtifactError, "not found"):
            create_app(self.dir / "missing.joblib")

    def test_missing_default_baseline_has_no_fallback(self):
        empty = self.dir / "empty_models"
        empty.mkdir()
        with self.assertRaisesRegex(ArtifactError, "does not exist"):
            create_app(models_dir=empty)
        self.assertEqual(list(empty.iterdir()), [])

    def test_checksum_mismatch(self):
        with self.assertRaisesRegex(ArtifactError, "Checksum mismatch"):
            create_app(self.path, expected_sha256="0" * 64)
        create_app(self.path, expected_sha256=self.artifact.sha256)

    def test_malformed_artifact(self):
        bad = self.dir / "bad.joblib"
        joblib.dump({"model": "not a model"}, bad)
        with self.assertRaises(ArtifactError):
            create_app(bad)

    def test_class_weighted_artifact_refused(self):
        weighted = self.dir / "w10.joblib"
        joblib.dump(make_bundle(class_weight={0: 1, 1: 10}), weighted)
        with self.assertRaisesRegex(ArtifactError, "class-weighted"):
            create_app(weighted)

    def test_feature_schema_mismatch_refused(self):
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=["a", "b", "c"])
        y = (X["a"] > 0).astype(int)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X, y)
        other = self.dir / "other.joblib"
        joblib.dump({"model": pipeline, "feature_columns": ["a", "b", "c"], "positive_class": 1}, other)
        with self.assertRaisesRegex(ArtifactError, "request schema"):
            create_app(other)


class TestNoTraining(ApiFixture):
    def test_startup_and_requests_never_fit(self):
        with mock.patch.object(Pipeline, "fit", side_effect=AssertionError("fit called")), \
             mock.patch.object(StandardScaler, "fit", side_effect=AssertionError("fit called")), \
             mock.patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit called")):
            client = TestClient(create_app(self.path))
            self.assertEqual(client.post("/predict", json={"transaction": self.tx}).status_code, 200)
            self.assertEqual(client.post("/predict/batch", json={"transactions": self.batch}).status_code, 200)
            self.assertEqual(client.get("/model-info").status_code, 200)

    def test_no_model_files_created_or_modified(self):
        before_tmp, before_real = snapshot(self.dir), snapshot(REAL_MODELS_DIR)
        client = TestClient(create_app(self.path))
        client.get("/health")
        client.get("/model-info")
        client.post("/predict", json={"transaction": self.tx})
        client.post("/predict/batch", json={"transactions": self.batch})
        client.post("/predict", json={"transaction": {}})
        self.assertEqual(snapshot(self.dir), before_tmp)
        self.assertEqual(snapshot(REAL_MODELS_DIR), before_real)

    def test_model_state_unchanged_by_requests(self):
        app = create_app(self.path)
        steps = app.state.artifact.pipeline.named_steps
        before = [a.copy() for a in (steps["scaler"].mean_, steps["scaler"].scale_, steps["model"].coef_)]
        TestClient(app).post("/predict/batch", json={"transactions": self.batch})
        after = [steps["scaler"].mean_, steps["scaler"].scale_, steps["model"].coef_]
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b, a)


REAL_BASELINE = REAL_MODELS_DIR / "final_logreg.joblib"


@unittest.skipUnless(REAL_BASELINE.is_file(), "real baseline artifact not present")
class TestRealBaselineIntegration(unittest.TestCase):
    def test_real_baseline_end_to_end(self):
        before = REAL_BASELINE.read_bytes()
        client = TestClient(create_app())
        info = client.get("/model-info").json()
        self.assertEqual(info["feature_count"], 30)
        self.assertFalse(info["class_weighted"])
        self.assertEqual({t["name"]: t["value"] for t in info["documented_thresholds"]},
                         {"baseline": 0.5, "provisional": 0.09})
        r = client.post("/predict", json={"transaction": make_transaction(), "threshold": 0.09})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(0.0 <= r.json()["prediction"]["fraud_score"] <= 1.0)
        self.assertTrue(math.isfinite(r.json()["prediction"]["fraud_score"]))
        self.assertEqual(REAL_BASELINE.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
