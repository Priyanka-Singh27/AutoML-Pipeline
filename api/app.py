import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any

app = FastAPI(title="AutoML Deployment API")

# Global state
model = None
metadata = None

@app.on_event("startup")
def load_artifacts():
    global model, metadata
    
    model_path = "outputs/models/model.joblib"
    meta_path = "outputs/models/model_metadata.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print(f"WARNING: Artifacts not found at {model_path} or {meta_path}. API endpoints will fail until models are generated.")
        return
        
    try:
        model = joblib.load(model_path)
        metadata = joblib.load(meta_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")

@app.get("/health")
def health_check():
    """Returns the core health status and metadata of the deployed model."""
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Models unavailable or failed to load on boot.")
        
    return {
        "status": "healthy",
        "model": metadata.get('best_model_name', 'Unknown'),
        "problem_type": metadata.get('problem_type', 'Unknown'),
        "trained_at": metadata.get('trained_at', 'Unknown')
    }

@app.get("/metrics")
def get_metrics():
    """Returns the training metrics, feature count, and core caveats for API consumers."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Models unavailable.")
        
    return {
        "model": metadata.get('best_model_name'),
        "problem_type": metadata.get('problem_type'),
        "f1_weighted": metadata.get('f1_weighted'),
        "roc_auc": metadata.get('roc_auc'),
        "rmse": metadata.get('rmse'),
        "r2": metadata.get('r2'),
        "trained_at": metadata.get('trained_at'),
        "n_features": len(metadata.get('expected_columns', [])),
        "limitations": metadata.get('limitations', []),
    }

def validate_and_predict(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Internal validation and prediction logic, shared across single and batch modes."""
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Models unavailable.")
        
    expected_cols = metadata.get('expected_columns', [])
    if not expected_cols:
        expected_cols = list(records[0].keys())

    # Build DataFrame
    try:
        df = pd.DataFrame(records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse records into pandas DataFrame: {str(e)}")
    
    # 1. Exact Column Difference Validation
    missing_cols = set(expected_cols) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_cols)
    
    if missing_cols:
        raise HTTPException(status_code=422, detail={
            "error": "Missing required features",
            "missing": list(missing_cols)
        })
        
    if extra_cols:
        raise HTTPException(status_code=422, detail={
            "error": "Unexpected features provided",
            "unexpected": list(extra_cols),
            "hint": "Send only the exact columns the model was trained on."
        })
        
    # 2. Strict Column Ordering Enforcement
    df = df[expected_cols]
    
    # 3. Dynamic Type Coercion
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass # Keep categorical strings
            
    # 4. Clustering Boundary Control
    prob_type = metadata.get('problem_type')
    if prob_type == 'clustering':
        if metadata.get('algorithm') != 'K-Means':
            raise HTTPException(status_code=400, detail={
                "error": "DBSCAN and Agglomerative models do not support "
                         "single-sample prediction. Retrain with K-Means for inference."
            })
            
    # 5. Inference
    try:
        preds = model.predict(df)
        
        # Probabilities if applicable
        probs = None
        if hasattr(model, 'predict_proba') and prob_type == 'classification':
            probs = model.predict_proba(df)
            
        class_labels = metadata.get('class_labels', [])
        predictions_out = []

        for i in range(len(df)):
            # Classification inference unroll
            if prob_type == 'classification':
                if len(class_labels) == 2:
                    p_val = float(probs[i][1]) if probs is not None else None
                    predictions_out.append({
                        "prediction": int(preds[i]) if pd.api.types.is_numeric_dtype(preds.dtype) else str(preds[i]),
                        "probability": p_val,
                        "class_labels": class_labels
                    })
                else:
                    prob_dict = {}
                    if probs is not None:
                        prob_dict = {str(label): float(probs[i][j]) for j, label in enumerate(class_labels)}
                    predictions_out.append({
                        "prediction": int(preds[i]) if pd.api.types.is_numeric_dtype(preds.dtype) else str(preds[i]),
                        "probabilities": prob_dict,
                        "class_labels": class_labels
                    })
            
            # Clustering inference unroll
            elif prob_type == 'clustering':
                predictions_out.append({"cluster": int(preds[i])})
            
            # Regression inference unroll
            else:
                predictions_out.append({
                    "prediction": float(preds[i]),
                    "prediction_type": "regression"
                })

        return predictions_out

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "error": "Prediction pipeline aborted during inference.",
            "message": "Encountered an internal model transformation error."
        })

@app.post("/predict")
def predict_single(payload: Dict[str, Any]):
    """Run inference against a single JSON sample."""
    if not payload:
        raise HTTPException(status_code=422, detail="Request body cannot be empty. Send a JSON payload mapped to features.")
        
    # Route payload directly treating payload as first and only dict row
    results = validate_and_predict([payload])
    return results[0]

@app.post("/predict/batch")
def predict_batch(payload: Dict[str, Any]):
    """Run inference against an array of up to 1000 JSON samples."""
    records = payload.get("records", [])
    if not records or not isinstance(records, list):
        raise HTTPException(status_code=422, detail="'records' array is required in the payload.")
    
    if len(records) > 1000:
        raise HTTPException(status_code=422, detail="Maximum 1000 records per batch exceeded.")
        
    results = validate_and_predict(records)
    return {"predictions": results, "count": len(results)}
