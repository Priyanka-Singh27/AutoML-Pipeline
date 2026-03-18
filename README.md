# AutoML Pipeline

**End-to-end automated machine learning — from raw CSV to deployed inference API, with zero manual feature engineering or model selection.**

---

## Overview
AutoML Pipeline accepts any CSV file, automatically audits the data, detects the problem type, selects features, tunes models under a time budget, explains predictions with SHAP, and optionally deploys a REST API — all from a single command. 

Supports binary and multiclass classification, regression, and unsupervised clustering. Architected across eight decoupled modules with frozen interface contracts, validated on four real-world datasets, and containerised for zero-dependency deployment.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run on any CSV
python main.py --file data/titanic.csv --target Survived --time-budget 120

# Clustering (no target required)
python main.py --file data/mall_customers.csv --unsupervised --time-budget 60

# Regression with PDF report
python main.py --file data/house_prices.csv --target SalePrice --report both
```

## Validation Results

| Dataset | Problem | Best Model | Key Metrics | Trials |
|---|---|---|---|---|
| Titanic (891 rows) | Binary Classification | LightGBM | F1=0.851 · ROC-AUC=0.876 | 731 |
| Telco Churn (7,043 rows) | Imbalanced Classification | Logistic Regression | F1=0.805 · ROC-AUC=0.847 | 438 |
| Mall Customers (200 rows) | Clustering | KMeans (k=4) | Silhouette=0.320 · DB=1.179 | — |
| Battery Degradation (6,966 rows) | Regression | LightGBM | RMSE=0.006 · R²=0.991 | 121 |

## CLI Reference
```bash
python main.py --file <path> [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--file` | path | **required** | Path to input CSV file |
| `--target` | str | `None` | Target column name for supervised problems |
| `--problem` | choice | `auto` | `classification` / `regression` / `clustering` |
| `--unsupervised`| flag | `False` | Force clustering mode — no target column needed |
| `--drop` | str | `None` | Comma-separated column names to drop before processing |
| `--no-smote` | flag | `False` | Disable SMOTE oversampling even if recommended |
| `--no-shap` | flag | `False` | Skip SHAP computation for faster runs |
| `--clusters` | int | `auto` | Override auto-detected number of clusters |
| `--time-budget` | int | `120` | Optuna tuning timeout in seconds |
| `--report` | choice | `both` | `terminal` / `pdf` / `both` |
| `--random-state`| int | `42` | Global random seed for full reproducibility |
| `--debug` | flag | `False` | Show full Python tracebacks on error |

## Pipeline Architecture

```text
CSV Input
    │
    ▼
┌─────────────┐
│   Auditor   │  16-step data profiling — leakage, imbalance,
│  auditor.py │  correlations, outliers, missing values
└──────┬──────┘
       │ audit dict
       ▼
┌─────────────┐
│  Detector   │  Signal Fusion Detection — normality tests,
│ detector.py │  entropy, gap variance, dtype, column name
└──────┬──────┘
       │ detection dict
       ▼
┌──────────────────┐
│  Preprocessor    │  Imputation, encoding, scaling, SMOTE,
│ preprocessor.py  │  LabelEncoder for string targets
└──────┬───────────┘
       │ df_clean + fitted ColumnTransformer
       ▼
┌──────────────────────┐
│  Feature Selector    │  Level 0: leakage removal
│ feature_selector.py  │  Level 1: variance filter
└──────┬───────────────┘  Level 2: correlation filter
       │                  Level 3: SHAP consensus importance
       │ X_train, X_test, y_train, y_test
       ▼
┌──────────────┐
│    Tuner     │  Unified Optuna search over model type
│   tuner.py   │  + hyperparameters under time budget
└──────┬───────┘
       │ study + best_model
       ▼
┌───────────────┐
│   Evaluator   │  Metrics, SHAP plots, confusion matrix,
│ evaluator.py  │  auto-generated limitations, pipeline stitch
└──────┬────────┘
       │ evaluation dict
       ▼
┌──────────────────┐     ┌─────────────┐
│    Reporter      │     │  Deployer   │
│  generator.py    │     │  app.py     │
│  PDF + terminal  │     │  FastAPI    │
└──────────────────┘     └─────────────┘
```

## Module Responsibilities

| Module | Responsibility | Never Does |
|---|---|---|
| `auditor.py` | Profiles raw data across 16 analytical dimensions | Modifies any data |
| `detector.py` | Infers problem type using statistical signal fusion | Reads raw data directly |
| `preprocessor.py` | Cleans, encodes, scales, applies SMOTE | Selects features |
| `feature_selector.py`| 4-level feature selection fitted on X_train only | Trains final model |
| `tuner.py` | Unified Optuna model + hyperparameter search | Evaluates on test set |
| `evaluator.py` | Metrics, SHAP, limitations, pipeline serialisation | Retrains model |
| `generator.py` | PDF and terminal report generation | Calls any ML code |
| `app.py` | FastAPI inference with strict schema validation | Retrains or re-evaluates |

## Model Pool
*   **Classification:** XGBoost · LightGBM · Random Forest · Logistic Regression · SVC (removed if n > 10,000 rows)
*   **Regression:** XGBoost · LightGBM · Random Forest · Ridge · SVR (removed if n > 10,000 rows)
*   **Clustering:** KMeans (elbow + silhouette) · DBSCAN (kNN eps estimation) · Agglomerative (ward / complete / average)

All supervised models are selected and tuned simultaneously in a single Optuna study using TPE sampling with `MedianPruner`.

## Feature Selection
Selection runs in four levels, all fitted on `X_train` only to prevent data leakage:
1.  **Leakage removal** — drop features with Spearman correlation > 0.95 against target
2.  **Variance filter** — drop quasi-constant columns (`VarianceThreshold`)
3.  **Correlation filter** — for each highly-correlated pair, keep the feature with higher SHAP consensus importance
4.  **SHAP consensus** — train LightGBM probe with 3 seeds, drop features below 1% of top SHAP score

*Post-selection cross-validation validates that selection did not degrade CV score by more than 2%.*

## Explainability
Every run produces SHAP-based explainability in `outputs/plots`:
-   **Global summary plot** — feature importance across all test samples
-   **Waterfall plot** — local explanation for a single prediction
-   **Auto-generated limitations** — dynamically triggered caveats based on detected data pathologies

SHAP explainer selection is automatic: `TreeExplainer` for tree models, `LinearExplainer` for linear models, `KernelExplainer` (capped at 200 samples) for SVC/SVR.

## FastAPI Inference
After training, spin up a REST endpoint:
```bash
python main.py --file data.csv --target y
# Select option 2 or 3 at the deployment prompt
```
The server loads the full inference pipeline once at boot (preprocessor + feature selector + model). Users send raw pre-preprocessing column values — the pipeline handles encoding and scaling internally.

| Route | Method | Purpose |
|---|---|---|
| `/health` | `GET` | Server status, model name, problem type |
| `/metrics` | `GET` | Training metrics and limitations |
| `/predict` | `POST` | Single-sample prediction |
| `/predict/batch` | `POST` | Batch prediction (max 1000 records) |

**Example Request:**
```python
import requests

response = requests.post("http://127.0.0.1:8000/predict", json={
    "Age": 25,
    "Annual Income (k$)": 60,
    "Spending Score (1-100)": 70,
    "Gender_Female": 0,
    "Gender_Male": 1
})
print(response.json())  # {"cluster": 2}
```

## Docker Deployment
```bash
# Build
docker build -t automl .

# Train (CLI mode — default)
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  automl --file data/titanic.csv --target Survived

# Serve (API mode)
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs \
  -e AUTOML_MODE=api automl

# Or use docker-compose
docker compose up train   # Train
docker compose up serve   # Serve
```
The same image handles both modes via the `AUTOML_MODE` environment variable (`cli` by default, `api` to serve). This means no GTK system dependencies are required — `weasyprint` runs natively inside the container.

## Project Structure
```text
AutoML/
├── core/
│   ├── auditor.py           # 16-step data profiling
│   ├── preprocessor.py      # Cleaning, encoding, scaling, SMOTE
│   ├── detector.py          # Signal Fusion problem type detection
│   ├── feature_selector.py  # 4-level feature selection
│   ├── tuner.py             # Unified Optuna model + HPO search
│   ├── evaluator.py         # Metrics, SHAP, pipeline stitching
│   ├── narrator.py          # Single stdout wrapper
│   ├── headers.py           # Section enum constants
│   └── exceptions.py        # ContractViolationError, PipelineStepError
├── api/
│   └── app.py               # FastAPI inference endpoint
├── reporting/
│   └── generator.py         # PDF + terminal report generation
├── tests/
│   └── test_all_mocks.py    # end-to-end mock test suite
├── data/                    # Input datasets
├── outputs/
│   ├── models/              # model.joblib + model_metadata.joblib
│   ├── plots/               # SHAP summary, waterfall, confusion matrix
│   └── reports/             # Generated PDF reports
├── main.py                  # CLI orchestration
├── entrypoint.py            # Docker dual-mode entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack
| Library | Purpose |
|---|---|
| `scikit-learn` | Pipeline, transformers, CV, metrics |
| `XGBoost` | Gradient boosted trees (classification + regression) |
| `LightGBM` | Fast gradient boosted trees |
| `Optuna` | Hyperparameter optimisation with TPE + pruning |
| `SHAP` | Model-agnostic explainability |
| `imbalanced-learn` | SMOTE oversampling for class imbalance |
| `FastAPI` + `uvicorn` | REST inference endpoint |
| `pandas` + `NumPy` | Data manipulation |
| `SciPy` | Statistical tests for signal fusion detection |
| `Jinja2` + `weasyprint` | PDF report generation |
| `joblib` | Pipeline serialisation |
| `pytest` | Test suite |
| `Docker` | Containerised deployment |

## Known Limitations
*   **No nested cross-validation** — Optuna tunes against fixed folds, so reported metrics may be optimistically biased by a few percent on small datasets.
*   **PDF reports require GTK3 on Windows** — Install from GTK for Windows or use `--report terminal`. Automatically natively handled if run via Docker.
*   **SHAP approximation for SVC/SVR** — `KernelExplainer` uses 200-row sampling; importance scores are approximations.
*   **Audit correlation scores computed pre-split** — Level 2 feature selection uses correlation scores from the full dataset, a minor form of leakage with negligible practical impact.
*   **Time series not supported** — Pipeline assumes i.i.d. rows with no temporal ordering.

## License
MIT
