"""Unit tests for fraud_detection.predict.

Uses small synthetic pipelines saved to a temporary directory, so no dataset,
labels or held-out test data are needed. One optional test loads the real
baseline artifact if it exists and checks it is not modified.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.predict import (  # noqa: E402
    ArtifactError,
    InvalidTransactionError,
    Prediction,
    fraud_scores,
    load_artifact,
    load_named_artifact,
    predict,
    predict_one,
    to_feature_frame,
)

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def make_bundle(class_weight=None, seed=0) -> dict:
    """Fit a tiny pipeline on synthetic data (test fixture only)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    X["Time"], X["Amount"] = X["Time"].abs() * 1000, X["Amount"].abs() * 50
    y = (X["V1"] + 0.5 * X["V2"] > 1.2).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("model", LogisticRegression(class_weight=class_weight, max_iter=1000))]).fit(X, y)
    return {"model": pipeline, "feature_columns": FEATURES, "positive_class": 1,
            "thresholds": {"baseline": 0.5, "provisional": 0.09}}


def make_transaction(seed=1) -> dict:
    rng = np.random.default_rng(seed)
    tx = {c: float(v) for c, v in zip(FEATURES, rng.normal(size=len(FEATURES)))}
    tx["Time"], tx["Amount"] = 1234.0, 42.5
    return tx


class ArtifactFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "final_logreg.joblib"
        joblib.dump(make_bundle(), self.path)
        self.artifact = load_artifact(self.path)
        self.tx = make_transaction()
        self.batch = [make_transaction(s) for s in range(1, 11)]

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, name, obj) -> Path:
        p = self.dir / name
        joblib.dump(obj, p)
        return p


class TestLoading(ArtifactFixture):
    def test_loads_valid_artifact(self):
        a = self.artifact
        self.assertEqual(a.feature_columns, tuple(FEATURES))
        self.assertEqual(a.positive_class, 1)
        self.assertFalse(a.is_class_weighted)
        self.assertEqual(dict(a.recorded_thresholds), {"baseline": 0.5, "provisional": 0.09})
        self.assertEqual(len(a.sha256), 64)

    def test_missing_file(self):
        with self.assertRaisesRegex(ArtifactError, "not found"):
            load_artifact(self.dir / "nope.joblib")

    def test_not_a_dict(self):
        with self.assertRaisesRegex(ArtifactError, "dictionary"):
            load_artifact(self.dump("bad.joblib", [1, 2, 3]))

    def test_missing_keys(self):
        bundle = make_bundle()
        del bundle["feature_columns"]
        with self.assertRaisesRegex(ArtifactError, "feature_columns"):
            load_artifact(self.dump("bad.joblib", bundle))

    def test_feature_order_mismatch_with_fitted_model(self):
        bundle = make_bundle()
        bundle["feature_columns"] = list(reversed(FEATURES))
        with self.assertRaisesRegex(ArtifactError, "order"):
            load_artifact(self.dump("bad.joblib", bundle))

    def test_target_listed_as_feature(self):
        bundle = make_bundle()
        bundle["feature_columns"] = FEATURES + ["Class"]
        with self.assertRaisesRegex(ArtifactError, "target"):
            load_artifact(self.dump("bad.joblib", bundle))

    def test_positive_class_not_in_model(self):
        bundle = make_bundle()
        bundle["positive_class"] = 7
        with self.assertRaisesRegex(ArtifactError, "positive class"):
            load_artifact(self.dump("bad.joblib", bundle))

    def test_unfitted_model_rejected(self):
        bundle = make_bundle()
        bundle["model"] = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
        with self.assertRaisesRegex(ArtifactError, "fitted classifier"):
            load_artifact(self.dump("bad.joblib", bundle))

    def test_checksum_pinning(self):
        load_artifact(self.path, expected_sha256=self.artifact.sha256)
        with self.assertRaisesRegex(ArtifactError, "Checksum mismatch"):
            load_artifact(self.path, expected_sha256="0" * 64)

    def test_phase_3a_single_threshold_format(self):
        bundle = make_bundle()
        del bundle["thresholds"]
        bundle["threshold"] = 0.5
        a = load_artifact(self.dump("old.joblib", bundle))
        self.assertEqual(dict(a.recorded_thresholds), {"baseline": 0.5})

    def test_named_baseline_loads(self):
        self.assertEqual(load_named_artifact("baseline", self.dir).sha256, self.artifact.sha256)

    def test_missing_challenger_is_not_trained(self):
        with self.assertRaisesRegex(ArtifactError, "later training phase"):
            load_named_artifact("challenger", self.dir)
        self.assertFalse((self.dir / "challenger_w10_logreg.joblib").exists())

    def test_unknown_name(self):
        with self.assertRaisesRegex(ArtifactError, "Unknown artifact"):
            load_named_artifact("best", self.dir)


class TestFeatureOrdering(ArtifactFixture):
    def test_frame_uses_training_order(self):
        shuffled = dict(reversed(list(self.tx.items())))
        frame = to_feature_frame(shuffled, FEATURES)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.iloc[0].tolist(), [self.tx[c] for c in FEATURES])

    def test_key_order_does_not_change_score(self):
        shuffled = dict(reversed(list(self.tx.items())))
        self.assertEqual(fraud_scores(self.artifact, self.tx)[0], fraud_scores(self.artifact, shuffled)[0])

    def test_dataframe_column_order_does_not_change_score(self):
        df = pd.DataFrame(self.batch)
        reordered = df[list(reversed(FEATURES))]
        np.testing.assert_array_equal(fraud_scores(self.artifact, df), fraud_scores(self.artifact, reordered))

    def test_matches_direct_pipeline_call(self):
        expected = self.artifact.pipeline.predict_proba(pd.DataFrame(self.batch)[FEATURES])[:, 1]
        np.testing.assert_array_equal(fraud_scores(self.artifact, self.batch), expected)


class TestInvalidInputs(ArtifactFixture):
    def assertInvalid(self, tx, pattern):
        with self.assertRaisesRegex(InvalidTransactionError, pattern):
            predict(self.artifact, tx, threshold=0.5)

    def test_missing_feature(self):
        tx = dict(self.tx)
        del tx["V7"]
        self.assertInvalid(tx, r"missing features \['V7'\]")

    def test_extra_feature(self):
        self.assertInvalid({**self.tx, "merchant": 3.0}, "unexpected fields.*merchant")

    def test_label_rejected(self):
        self.assertInvalid({**self.tx, "Class": 0}, "label")

    def test_label_rejected_in_dataframe(self):
        self.assertInvalid(pd.DataFrame(self.batch).assign(Class=0), "label")

    def test_non_numeric_values(self):
        for bad in ["abc", "1.5", None, [1.0], True, np.bool_(False)]:
            with self.subTest(value=bad):
                self.assertInvalid({**self.tx, "V3": bad}, "V3.*expected a number")

    def test_non_finite_values(self):
        for bad in [math.nan, math.inf, -math.inf, np.float64("nan")]:
            with self.subTest(value=bad):
                self.assertInvalid({**self.tx, "V3": bad}, "finite")

    def test_negative_amount_and_time(self):
        self.assertInvalid({**self.tx, "Amount": -1.0}, "Amount.*non-negative")
        self.assertInvalid({**self.tx, "Time": -5}, "Time.*non-negative")

    def test_error_names_the_row(self):
        batch = [*self.batch[:3], {**self.tx, "V9": "x"}]
        self.assertInvalid(batch, "row 3")

    def test_empty_batch_and_wrong_container(self):
        self.assertInvalid([], "No transactions")
        self.assertInvalid(pd.DataFrame(columns=FEATURES), "No transactions")
        self.assertInvalid("not a transaction", "Expected a mapping")
        self.assertInvalid([1.0, 2.0], "row 0: expected a mapping")

    def test_numpy_and_int_values_accepted(self):
        tx = {**self.tx, "V1": np.float32(0.25), "V2": 3, "Amount": np.int64(10)}
        self.assertEqual(len(predict(self.artifact, tx, threshold=0.5)), 1)

    def test_invalid_thresholds(self):
        for bad in [-0.01, 1.01, math.nan, math.inf, True, "0.5", None]:
            with self.subTest(threshold=bad), self.assertRaisesRegex(ValueError, "threshold"):
                predict(self.artifact, self.tx, threshold=bad)

    def test_threshold_is_required(self):
        with self.assertRaises(TypeError):
            predict(self.artifact, self.tx)  # no silent default


class TestPredictions(ArtifactFixture):
    def test_single_returns_one_prediction(self):
        result = predict(self.artifact, self.tx, threshold=0.5)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Prediction)
        self.assertTrue(0.0 <= result[0].fraud_score <= 1.0)

    def test_batch_preserves_order_and_matches_single(self):
        batch = predict(self.artifact, self.batch, threshold=0.5)
        self.assertEqual(len(batch), len(self.batch))
        for tx, p in zip(self.batch, batch):
            single = predict_one(self.artifact, tx, threshold=0.5)
            # Batch vs single-row BLAS summation order can differ by ~1e-15 (documented in predict.py).
            self.assertAlmostEqual(single.fraud_score, p.fraud_score, delta=1e-12)
            self.assertEqual((single.flagged, single.threshold), (p.flagged, p.threshold))

    def test_dataframe_and_list_inputs_agree(self):
        self.assertEqual(predict(self.artifact, self.batch, threshold=0.3),
                         predict(self.artifact, pd.DataFrame(self.batch), threshold=0.3))

    def test_predict_one_rejects_batches(self):
        with self.assertRaises(InvalidTransactionError):
            predict_one(self.artifact, self.batch, threshold=0.5)


class TestThresholds(ArtifactFixture):
    def test_flag_is_score_at_or_above_threshold(self):
        for p in predict(self.artifact, self.batch, threshold=0.2):
            self.assertEqual(p.flagged, p.fraud_score >= 0.2)
            self.assertEqual(p.threshold, 0.2)

    def test_score_equal_to_threshold_is_flagged(self):
        score = fraud_scores(self.artifact, self.tx)[0]
        self.assertTrue(predict(self.artifact, self.tx, threshold=float(score))[0].flagged)

    def test_extremes(self):
        self.assertTrue(all(p.flagged for p in predict(self.artifact, self.batch, threshold=0.0)))
        scores = fraud_scores(self.artifact, self.batch)
        flags = [p.flagged for p in predict(self.artifact, self.batch, threshold=1.0)]
        self.assertEqual(flags, [bool(s >= 1.0) for s in scores])

    def test_higher_threshold_flags_a_subset(self):
        low = {i for i, p in enumerate(predict(self.artifact, self.batch, threshold=0.09)) if p.flagged}
        high = {i for i, p in enumerate(predict(self.artifact, self.batch, threshold=0.5)) if p.flagged}
        self.assertTrue(high <= low)

    def test_scores_do_not_depend_on_threshold(self):
        a = [p.fraud_score for p in predict(self.artifact, self.batch, threshold=0.09)]
        b = [p.fraud_score for p in predict(self.artifact, self.batch, threshold=0.9)]
        self.assertEqual(a, b)


class TestReproducibilityAndNoLeakage(ArtifactFixture):
    def test_repeated_calls_identical(self):
        self.assertEqual(predict(self.artifact, self.batch, threshold=0.5),
                         predict(self.artifact, self.batch, threshold=0.5))

    def test_reloaded_artifact_identical(self):
        again = load_artifact(self.path)
        self.assertEqual(again.sha256, self.artifact.sha256)
        self.assertEqual(predict(again, self.batch, threshold=0.5), predict(self.artifact, self.batch, threshold=0.5))

    def test_inference_never_fits(self):
        with mock.patch.object(Pipeline, "fit", side_effect=AssertionError("fit called")), \
             mock.patch.object(StandardScaler, "fit", side_effect=AssertionError("fit called")), \
             mock.patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit called")):
            predict(self.artifact, self.batch, threshold=0.5)

    def test_model_state_unchanged(self):
        scaler, model = self.artifact.pipeline.named_steps["scaler"], self.artifact.pipeline.named_steps["model"]
        before = (scaler.mean_.copy(), scaler.scale_.copy(), model.coef_.copy(), model.intercept_.copy())
        predict(self.artifact, self.batch, threshold=0.5)
        for b, a in zip(before, (scaler.mean_, scaler.scale_, model.coef_, model.intercept_)):
            np.testing.assert_array_equal(b, a)

    def test_input_not_mutated(self):
        batch = copy.deepcopy(self.batch)
        df = pd.DataFrame(self.batch)
        df_before = df.copy()
        predict(self.artifact, batch, threshold=0.5)
        predict(self.artifact, df, threshold=0.5)
        self.assertEqual(batch, self.batch)
        pd.testing.assert_frame_equal(df, df_before)


class TestScoreSemantics(ArtifactFixture):
    def test_unweighted_not_called_calibrated(self):
        text = self.artifact.score_semantics
        self.assertIn("not validated as a calibrated", text)

    def test_weighted_model_flagged_as_not_probability(self):
        a = load_artifact(self.dump("w10.joblib", make_bundle(class_weight={0: 1, 1: 10})))
        self.assertTrue(a.is_class_weighted)
        self.assertIn("not a probability", a.score_semantics)


REAL_BASELINE = Path(__file__).resolve().parents[1] / "models" / "final_logreg.joblib"


@unittest.skipUnless(REAL_BASELINE.is_file(), "real baseline artifact not present")
class TestRealBaselineArtifact(unittest.TestCase):
    def test_loads_without_modifying_file(self):
        before = REAL_BASELINE.read_bytes()
        a = load_named_artifact("baseline")
        self.assertEqual(len(a.feature_columns), 30)
        self.assertFalse(a.is_class_weighted)
        self.assertEqual(dict(a.recorded_thresholds), {"baseline": 0.5, "provisional": 0.09})
        p1 = predict(a, make_transaction(), threshold=a.recorded_thresholds["provisional"])
        p2 = predict(load_named_artifact("baseline"), make_transaction(), threshold=0.09)
        self.assertEqual(p1, p2)
        self.assertEqual(REAL_BASELINE.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
