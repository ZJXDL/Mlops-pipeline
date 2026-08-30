```markdown
# MLOps Pipeline: Vehicle Maintenance & Failure Prediction

An end-to-end, production-grade MLOps system built to automate data preprocessing, model training, experiment tracking, automated testing, data drift monitoring, and containerized CI/CD deployment.

---

## Architecture & Workflow Overview


```

[Raw Data]
│
▼
[Preprocessing (pandas / sklearn)] ──► [Train/Test Split]
│
├──► [Model Training (RandomForest)] ──► [MLflow Tracking / DAGsHub]
│
└──► [Evidently AI Data Drift Monitor] ──► [HTML Drift Report Artifact]
│
└──► [Pytest Unit Tests]
│
▼
[GitHub Actions CI/CD Pipeline]
│
▼
[Docker Build & Push to GHCR]

```

---

## Project Structure

```text
Mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI/CD & scheduled retraining pipeline
├── data/
│   ├── raw/                # Initial raw datasets
│   └── processed/          # Cleaned train.csv and test.csv splits
├── models/
│   └── model.joblib        # Serialized production model binary
├── reports/
│   └── data_drift_report.html # Automatically generated Evidently AI drift report
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Data cleaning and feature engineering script
│   ├── train.py            # Model training & MLflow logging script
│   └── monitor.py          # Evidently AI data drift check script
├── tests/
│   └── test_pipeline.py    # Pytest unit tests for pipeline components
├── Dockerfile              # Containerization configuration
├── requirements.txt        # Project dependencies and pinned versions
└── README.md

```

---

## Step-by-Step Implementation Breakdown (From A to Z)

### 1. Data Engineering & Preprocessing (`src/preprocess.py`)

* Loads raw maintenance logs and telemetry data.
* Cleans missing fields, normalizes numerical parameters, encodes categorical flags, and structures features.
* Automatically splits data into deterministic training and testing subsets, saving them neatly under `data/processed/`.

### 2. Machine Learning Training & Experiment Tracking (`src/train.py`)

* Trains a `RandomForestClassifier` optimized to predict vehicle mechanical failures (`failure` target variable).
* Integrates **MLflow** for experiment tracking.
* Automatically logs hyperparameters (`n_estimators`, `max_depth`), evaluation metrics (`accuracy`), and saves the trained model binary using `mlflow.sklearn` and `joblib`.

### 3. Automated Testing (`tests/`)

* Built comprehensive unit tests using `pytest` to validate data schemas, verify preprocessing integrity, and confirm model inference correctness before deployment.

### 4. Data Drift Monitoring (`src/monitor.py`)

* Uses **Evidently AI** to perform statistical data drift analyses (comparing baseline training features against production test distributions).
* Automatically compiles and exports an interactive HTML analytical dashboard saved to `reports/data_drift_report.html`.

### 5. CI/CD Pipeline & Automated Retraining (`.github/workflows/ci.yml`)

* Configured a robust GitHub Actions workflow triggered on every `push`, `pull_request`, and via a weekly cron schedule (`0 0 * * 1`) for automated model retraining.
* **Pipeline Execution Steps:**
1. Checks out the repository and configures Python 3.10.
2. Installs all project dependencies from `requirements.txt`.
3. Executes data preprocessing and model training.
4. Runs unit test suites via `pytest`.
5. Executes Evidently AI data drift monitoring.
6. Uploads the generated data drift HTML report as a downloadable build artifact.
7. Authenticates with the GitHub Container Registry (`GHCR`).
8. Builds and pushes a containerized Docker image tagged with `latest` and the unique Git commit SHA.



---

## Local Setup & Installation

If you want to run or test the pipeline locally on your machine, follow these steps:

1. **Clone the repository:**
```bash
git clone [https://github.com/ZJXDL/Mlops-pipeline.git](https://github.com/ZJXDL/Mlops-pipeline.git)
cd Mlops-pipeline

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run data preprocessing:**
```bash
python src/preprocess.py

```


4. **Train the model:**
```bash
python src/train.py

```


5. **Run data drift checks:**
```bash
python src/monitor.py

```


6. **Execute unit tests:**
```bash
PYTHONPATH=. python -m pytest

```



---

## Tech Stack

* **Language:** Python 3.10
* **Core ML & Data:** Scikit-Learn, Pandas, NumPy, Joblib
* **Experiment Tracking:** MLflow, DAGsHub
* **Monitoring & Drift:** Evidently AI
* **Testing:** Pytest
* **CI/CD & Automation:** GitHub Actions, GitHub Container Registry (GHCR), Docker

```

```
