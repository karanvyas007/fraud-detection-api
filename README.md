# Credit Card Fraud Detection API

A portfolio project that trains a fraud classifier on the public ULB credit-card dataset and serves it through a small FastAPI application. The focus is on leakage-safe evaluation and honest reporting rather than a headline accuracy number.

> **Status:** experimental. An unweighted Logistic Regression baseline is trained, evaluated and served locally. It is **not production-ready**: no authentication, rate limiting, deployment configuration or production probability calibration.

## Architecture

```
data/raw/creditcard.csv            (downloaded manually, never committed)
        │
scripts/prepare_data.py            leakage-safe 80/20 split → data/processed/
        │
scripts/train_*.py, tune_*.py      training, threshold selection, evaluation → models/, reports/
        │
src/fraud_detection/predict.py     inference only: loads a saved pipeline, validates input, scores
        │
src/fraud_detection/api.py         thin FastAPI layer over predict.py
```

- **Model:** `Pipeline(StandardScaler → LogisticRegression)`, unweighted, trained on the training split only.
- **Inference never trains.** `predict.py` loads a saved joblib artifact; the API refuses to start if that artifact is missing or invalid, and never falls back to another model.

```
├── scripts/                 training and experiment scripts (run from the project root)
├── src/fraud_detection/     predict.py (inference), api.py (FastAPI)
├── tests/                   test_predict.py, test_api.py (unittest)
├── data/  models/  reports/ generated or downloaded locally; ignored by Git
├── requirements.txt
└── README.md
```

## Dataset

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group, Université Libre de Bruxelles (ULB). Kaggle lists the licence as *"Database: Open Database, Contents: Database Contents"*. The dataset page asks users to cite Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification* (IEEE CIDM, 2015).
- **Contents:** 284,807 transactions by European cardholders over two days in September 2013. One file, `creditcard.csv`, of 150,828,752 bytes.
- **Columns:** `Time` (seconds since the first transaction), `V1`–`V28`, `Amount`, and the label `Class` (1 = fraud, 0 = legitimate).
- **Class imbalance:** 492 frauds (0.173%). A model that always predicts "legitimate" scores 99.83% accuracy and catches no fraud, so this project uses precision, recall and average precision (PR-AUC) instead of accuracy.
- **Duplicates:** there are 1,081 exact duplicate rows. Identical feature rows are always kept in the same split so they can't leak between training and test.
- **Limitations of `V1`–`V28`:** these are anonymised PCA components. They can't be interpreted, and a real caller couldn't produce them without the bank's original transformation. The API therefore accepts these synthetic features, not real transaction fields.

### Getting the data

The dataset is not included in this repository. To get it:

1. Sign in to Kaggle and download the dataset from the page above.
2. Extract `creditcard.csv` into `data/raw/`:
   ```bash
   mkdir -p data/raw && unzip ~/Downloads/<downloaded-file>.zip -d data/raw/
   ```
3. Check the file (expected: 150828752 bytes, 284808 lines including the header). The scripts also check its SHA-256 against the split metadata:
   ```bash
   ls -l data/raw/creditcard.csv && wc -l data/raw/creditcard.csv
   ```

## Setup

You need Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the data and ML libraries, FastAPI, Pydantic and Uvicorn, plus `httpx`, which FastAPI's test client needs. pytest is not required.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

There are 78 tests: 46 for the inference module and 32 for the API.
- They use small synthetic models saved in temporary folders, so **they don't need the dataset**.
- Two integration tests use `models/final_logreg.joblib` and are skipped automatically if it doesn't exist.
- If the API dependencies aren't installed, `test_api.py` is skipped with a message rather than failing.

## Reproducing the pipeline

Run the scripts from the project root with the virtual environment active, **in this order**: later scripts check the outputs of earlier ones and stop if they're missing or don't match. Every script is deterministic and accepts `--help`.

| Step | Command | Writes | Approx. time |
|---|---|---|---|
| 1. Validate and explore the data | `python scripts/inspect_dataset.py` | `reports/eda/` | seconds |
| 2. Leakage-safe split | `python scripts/prepare_data.py` | `data/processed/` | ~16 s |
| 3. Baseline + dummy reference | `python scripts/train_baseline.py` | `models/baseline_logreg.joblib`, `reports/baseline/` | seconds |
| 4. Threshold selection (training data only) | `python scripts/tune_threshold.py` | `reports/threshold_tuning/` | ~30 s |
| 5. Final baseline + one-time test evaluation | `python scripts/train_final.py` | `models/final_logreg.joblib`, `reports/final_evaluation/` | seconds |
| 6. Logistic Regression configuration comparison | `python scripts/compare_logistic_models.py` | `reports/model_comparison/` | ~35 s |
| 7. Score saturation and milder class weights | `python scripts/weighting_experiment.py` | `reports/weighting_experiment/` | ~30 s |
| 8. Repeated group-aware CV (unweighted vs w=10) | `python scripts/repeated_cv.py` | `reports/repeated_cv/` | ~4–5 min |

**Notes on the steps:**
- **Overwriting:** rerunning a script overwrites its own outputs. Steps 3 and 5 rewrite the model files.
- **Step 5:**
  - It accepts only the thresholds 0.5 and 0.09, and refuses to run if `reports/threshold_tuning/selected_threshold.json` records anything else.
  - It reads the test labels once, for the final report. Steps 4 and 6–8 never read test labels.
- **Steps 7 and 8:** both check that their results reproduce earlier steps exactly (bit-identical scores), so run steps 4 and 6 first.

## Running the API

Start the server from the project root. It needs `models/final_logreg.joblib`, which step 5 creates.

```bash
uvicorn fraud_detection.api:create_app --factory --app-dir src
```

Then open:
- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Model information: http://127.0.0.1:8000/model-info

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` while the process is running |
| `GET /model-info` | Model name and type, the 30 feature names, score description, default and documented thresholds, maximum batch size. No file paths or checksums. |
| `POST /predict` | Scores one transaction |
| `POST /predict/batch` | Scores 1–1000 transactions and returns results in input order |

### Example requests

> **The feature values below are made up for illustration.** They aren't real transactions, and the scores they produce mean nothing.

The request must include all 30 features. Extra fields, including `Class`, are rejected.

`POST /predict`:

```json
{
  "transaction": {
    "Time": 406.0,
    "V1": -1.2, "V2": 0.8, "V3": -0.5, "V4": 1.1, "V5": -0.3, "V6": 0.2, "V7": -0.4,
    "V8": 0.1, "V9": 0.3, "V10": -0.6, "V11": 0.9, "V12": -1.0, "V13": 0.2, "V14": -1.3,
    "V15": 0.4, "V16": -0.7, "V17": -0.8, "V18": 0.1, "V19": 0.2, "V20": 0.05, "V21": 0.1,
    "V22": -0.2, "V23": 0.03, "V24": 0.1, "V25": 0.2, "V26": -0.1, "V27": 0.05, "V28": 0.02,
    "Amount": 149.62
  },
  "threshold": 0.5
}
```

`POST /predict/batch`:

```json
{
  "transactions": [
    {
      "Time": 406.0,
      "V1": -1.2, "V2": 0.8, "V3": -0.5, "V4": 1.1, "V5": -0.3, "V6": 0.2, "V7": -0.4,
      "V8": 0.1, "V9": 0.3, "V10": -0.6, "V11": 0.9, "V12": -1.0, "V13": 0.2, "V14": -1.3,
      "V15": 0.4, "V16": -0.7, "V17": -0.8, "V18": 0.1, "V19": 0.2, "V20": 0.05, "V21": 0.1,
      "V22": -0.2, "V23": 0.03, "V24": 0.1, "V25": 0.2, "V26": -0.1, "V27": 0.05, "V28": 0.02,
      "Amount": 149.62
    },
    {
      "Time": 3600.0,
      "V1": 1.1, "V2": 0.1, "V3": 0.4, "V4": 0.3, "V5": 0.0, "V6": -0.1, "V7": 0.1,
      "V8": 0.0, "V9": -0.2, "V10": 0.1, "V11": 0.5, "V12": 0.6, "V13": -0.3, "V14": 0.2,
      "V15": 0.9, "V16": 0.3, "V17": -0.2, "V18": -0.1, "V19": 0.0, "V20": -0.05, "V21": -0.2,
      "V22": -0.5, "V23": 0.1, "V24": -0.3, "V25": 0.2, "V26": 0.1, "V27": -0.01, "V28": 0.01,
      "Amount": 2.69
    }
  ],
  "threshold": 0.09
}
```

To send a body from the command line, save it to a file and run:
```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @single.json
```

Example response (abbreviated; the score depends on the input):

```json
{
  "model_name": "baseline",
  "threshold": 0.5,
  "threshold_source": "request",
  "score_semantics": "Model output of an unweighted Logistic Regression, in [0, 1]. Usable for ranking; not validated as a calibrated real-world fraud probability.",
  "prediction": {"index": 0, "fraud_score": 0.00092, "flagged": false}
}
```

**Invalid requests** get a `422` response with the location and reason for each problem. Examples: missing or extra features, strings, booleans, `null`, NaN or Infinity, a negative `Time` or `Amount`, or a threshold outside [0, 1]. Unexpected server errors return a generic `500` without internal details.

### What the output means

- **`fraud_score`:** the model's output for the fraud class, between 0 and 1. A higher score means more fraud-like, **compared with other transactions**.
  - **It is not a calibrated real-world fraud probability.** Cross-validation on the training data showed the model is underconfident in the middle (transactions scored 5–10% were fraud about 30% of the time) and overconfident at the top (scores of 99% or more were fraud about 88% of the time).
  - Treat it as a ranking score.
- **`flagged`:** `fraud_score >= threshold`.
- **`threshold`:** optional in each request.
  - If you leave it out, the API uses **0.5**, the conventional default, which isn't tuned for fraud costs. The response's `threshold_source` says whether the default (`"api_default"`) or your value (`"request"`) was used.
  - `0.09` is an **experimental operating point**: it maximised F1 in cross-validation on the training data for this model. You must pass it explicitly.
  - Neither threshold is a business decision. A real deployment would set the threshold from fraud losses, review costs or review capacity.
  - Thresholds don't transfer between models.

## Results

There are two kinds of results here, and they shouldn't be mixed up:

- **Held-out test evaluation:** done once, with thresholds fixed beforehand. This is the project's reported performance.
- **Training-only cross-validation experiments:** used to choose thresholds and compare model variants. They never read test labels.

### Held-out test set (step 5)

56,962 transactions, of which 99 are fraud.

| Model | Threshold | Average precision | Precision | Recall | F1 | False alarms | Frauds missed |
|---|---|---|---|---|---|---|---|
| Always "legitimate" (reference) | — | 0.0017 | undefined | 0.000 | 0.000 | 0 | 99 |
| Logistic Regression | 0.50 | 0.805 | 0.931 | 0.545 | 0.688 | 4 | 45 |
| Logistic Regression | 0.09 | 0.805 | 0.859 | 0.737 | 0.794 | 12 | 26 |

- **Accuracy is above 99.8% for every row, which is why it isn't used to compare models.**
- **Noise:** with 99 test frauds, each fraud is worth about 1 percentage point of recall.
- **The test set has now been used:** any further model choices should rely on cross-validation or new data, not on this test set.

### Training-only cross-validation (steps 4 and 6–8)

All experiments use group-aware `StratifiedGroupKFold` on the training split (227,845 rows, 393 frauds).

- **Threshold (step 4):** the out-of-fold average precision is 0.746. At 0.09: precision 0.758, recall 0.766. The best threshold varied from 0.05 to 0.42 between folds, so 0.09 is approximate.
- **Regularisation (step 6):** `C` = 0.1, 1 or 10 made no meaningful difference.
- **Full class weighting (`balanced`) (steps 6–7):** it lowered average precision (0.715 vs 0.746) and pushed 150 scores to exactly 1.0. That breaks probability calibration and makes probability thresholds unreliable.
- **Mild weighting `{0: 1, 1: 10}` (steps 7–8):** across 10 different fold seeds it was ahead of the unweighted baseline every time.
  - Average precision was higher by 0.008 on average.
  - At a fixed budget of 397 alerts it caught 4–10 more frauds.

  All seeds reuse the same 393 frauds, so this shows the result is stable, not that it's independently confirmed. **The weighted model hasn't been selected, and no saved artifact exists for it.** The API serves only the unweighted baseline.

## Files kept out of Git

The project isn't a Git repository yet. When it becomes one, `.gitignore` excludes:

| Path | What it is | How to recreate it |
|---|---|---|
| `data/raw/`, `creditcard.csv`, `*.zip` | Raw dataset and the Kaggle download | Manual download (see [Getting the data](#getting-the-data)) |
| `data/processed/` | Split assignments and metadata | `scripts/prepare_data.py` |
| `models/`, `*.joblib`, `*.pkl` | Trained model artifacts | `scripts/train_baseline.py`, `scripts/train_final.py` |
| `reports/` | Generated reports and plots | The script listed for each folder above |
| `.venv/`, `.env`, `.env.*` | Local environment and secrets | `python3.12 -m venv .venv` |
| Caches (`__pycache__/`, `.pytest_cache/`, …) | Generated | — |

Only source code, tests, `requirements.txt`, `.gitignore` and this README are meant to be committed. Joblib files are pickles and can run code when loaded, so load only artifacts you created yourself. `predict.load_artifact()` can also check a file against a known SHA-256 before loading it.

## Limitations

- **Not production-ready:** no authentication, rate limiting, logging/monitoring, HTTPS, container image or deployment configuration.
- **Scores are not calibrated probabilities.** No production calibration step exists.
- **Thresholds are experimental** (0.5 is conventional; 0.09 comes from training-only CV). They aren't based on business costs.
- **Data limits:** two days of 2013 data, anonymised PCA features, random (not time-ordered) splits. Nothing here tests changes in fraud patterns over time.
- **One model:** the API serves a single model, loaded at startup. Changing the model file requires a restart.
- **Range checks:** only `Time` and `Amount` (must be non-negative) are checked. Unusual `V1`–`V28` values are scored without any warning.
- **Setup notes:**
  - The package isn't installed: the API is started with `--app-dir src`, and the tests add `src/` to the import path themselves.
  - `requirements.txt` sets minimum versions rather than pinning exact ones.
  - Running the tests with the current Starlette/httpx versions prints a deprecation warning about `httpx`; the tests still pass.
