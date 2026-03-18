import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression

import api.app
from api.app import app

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def create_mock_pipeline_and_meta(problem_type="clustering", algorithm="K-Means"):
    if problem_type == "clustering":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=2, random_state=42, n_init=1))
        ])
        X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
        model.fit(X)
        meta = {
            "best_model_name": f"{algorithm}_Mock",
            "problem_type": "clustering",
            "algorithm": algorithm,
            "trained_at": "2026-03-18T10:00:00",
            "expected_columns": ["feat1", "feat2"],
            "limitations": []
        }
    elif problem_type == "classification":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])
        X = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [4, 5, 6, 7]})
        y = [0, 1, 0, 1]
        model.fit(X, y)
        meta = {
            "best_model_name": "LogReg_Mock",
            "problem_type": "classification",
            "algorithm": "LogisticRegression",
            "trained_at": "2026-03-18T10:00:00",
            "expected_columns": ["feat1", "feat2"],
            "class_labels": ["Class_A", "Class_B"],
            "limitations": []
        }
    else: # regression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
        y = [1.0, 2.0, 3.0]
        model.fit(X, y)
        meta = {
            "best_model_name": "LinReg_Mock",
            "problem_type": "regression",
            "algorithm": "LinearRegression",
            "trained_at": "2026-03-18T10:00:00",
            "expected_columns": ["feat1", "feat2"],
            "limitations": []
        }
    return model, meta

@pytest.fixture
def mock_clustering_fixture():
    pipe, meta = create_mock_pipeline_and_meta("clustering", "K-Means")
    with patch("api.app.model", pipe), patch("api.app.metadata", meta):
        yield

@pytest.fixture
def mock_classification_fixture():
    pipe, meta = create_mock_pipeline_and_meta("classification")
    with patch("api.app.model", pipe), patch("api.app.metadata", meta):
        yield

@pytest.fixture
def mock_regression_fixture():
    pipe, meta = create_mock_pipeline_and_meta("regression")
    with patch("api.app.model", pipe), patch("api.app.metadata", meta):
        yield

@pytest.fixture
def mock_dbscan_fixture():
    pipe, meta = create_mock_pipeline_and_meta("clustering", "DBSCAN")
    with patch("api.app.model", pipe), patch("api.app.metadata", meta):
        yield

@pytest.fixture
def client(mock_clustering_fixture):
    # Default client uses clustering mock
    return TestClient(app)

# --------------------------------------------------------------------------
# Tests: Health and startup
# --------------------------------------------------------------------------

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert all(k in data for k in ["status", "model", "problem_type", "trained_at"])
    assert data["status"] == "healthy"

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert all(k in data for k in ["model", "problem_type", "n_features", "limitations"])
    assert data["problem_type"] == "clustering"

# --------------------------------------------------------------------------
# Tests: Happy path
# --------------------------------------------------------------------------

def test_predict_happy_path(client):
    response = client.post("/predict", json={"feat1": 10, "feat2": 20})
    assert response.status_code == 200
    data = response.json()
    assert "cluster" in data

def test_predict_batch_happy_path(client):
    records = [{"feat1": i, "feat2": i * 2} for i in range(5)]
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == 5
    assert len(data["predictions"]) == 5

# --------------------------------------------------------------------------
# Tests: Edge cases
# --------------------------------------------------------------------------

def test_missing_columns(client):
    response = client.post("/predict", json={"feat1": 10}) # Missing feat2
    assert response.status_code == 422
    data = response.json()
    assert "missing" in data["detail"]
    assert "feat2" in data["detail"]["missing"]

def test_extra_columns(client):
    response = client.post("/predict", json={"feat1": 10, "feat2": 20, "feat3": 30})
    assert response.status_code == 422
    data = response.json()
    assert "unexpected" in data["detail"]
    assert "feat3" in data["detail"]["unexpected"]

def test_malformed_json(client):
    # TestClient abstracts the connection, so we trigger JSON parse error by sending a string containing invalid JSON logic
    response = client.post("/predict", data="{'invalid_json': True}")
    assert response.status_code == 422
    data = response.json()
    # FastAPI returns specific validation errors for Unprocessable Entity when JSON decoding fails
    assert any("JSON" in str(detail.get("msg", "")).upper() for detail in data["detail"]) or \
           any("json" in detail.get("type", "") for detail in data["detail"])

def test_empty_payload(client):
    # Missing fields
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_empty_batch(client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422

def test_oversized_batch(client):
    records = [{"feat1": i, "feat2": i} for i in range(1001)]
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 422
    assert "Maximum 1000 records" in response.json()["detail"]

# --------------------------------------------------------------------------
# Tests: Schema validation & Method Limits
# --------------------------------------------------------------------------

def test_schema_clustering(client): # uses default fixture
    response = client.post("/predict", json={"feat1": 1.5, "feat2": 2.5})
    assert response.status_code == 200
    data = response.json()
    assert "cluster" in data
    assert isinstance(data["cluster"], int)

def test_schema_classification(mock_classification_fixture):
    client = TestClient(app)
    response = client.post("/predict", json={"feat1": 1.5, "feat2": 2.5})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    # Can be float, decimal, etc.
    assert "probability" in data or "probabilities" in data

def test_schema_regression(mock_regression_fixture):
    client = TestClient(app)
    response = client.post("/predict", json={"feat1": 1.5, "feat2": 2.5})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_type" in data
    assert data["prediction_type"] == "regression"

def test_dbscan_predict_error(mock_dbscan_fixture):
    client = TestClient(app)
    response = client.post("/predict", json={"feat1": 1.5, "feat2": 2.5})
    assert response.status_code == 400
    data = response.json()
    assert "DBSCAN" in data["detail"]["error"]
