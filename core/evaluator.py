"""
evaluator.py - STUB
Phase 2 end-to-end validation stub.
Produces a correctly-shaped evaluation_object with real metrics but no SHAP plots.
Full SHAP explainability + limitation generation comes in Phase 6.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os


def run_evaluation(model, X, y, detection, audit, study):
    """
    Stub: computes real metrics and packages the full evaluation_object.
    SHAP plots, limitations generation, and confusion matrix inference
    will be added in Phase 6.

    Parameters
    ----------
    model    : fitted sklearn model
    X        : pd.DataFrame  Feature-selected data
    y        : pd.Series or None
    detection: dict
    audit    : dict
    study    : _StubStudy or optuna.Study

    Returns
    -------
    evaluation : dict   Full evaluation object (Person 3 contract)
    """
    problem_type = detection['problem_type']
    print(f"\n[EVALUATION - STUB]")

    # Build base evaluation object (all keys present, unused ones = None)
    evaluation = {
        # Optuna
        'best_model': model,
        'best_model_name': getattr(study, 'model_name', study.best_params.get('model', 'unknown')),
        'best_params': study.best_params,
        'training_time': 0.0,
        'n_trials': len(study.trials),
        'all_trials': [],

        # Feature selection (populated by feature_selector - passed through here)
        'features_original': audit.get('shape', (0, 0))[1],
        'features_remaining': X.shape[1],
        'features_dropped': {},

        # Classification (None unless classification)
        'confusion_matrix': None,
        'confusion_matrix_inference': None,
        'classification_report': None,
        'roc_auc': None,
        'f1_weighted': None,
        'precision_weighted': None,
        'recall_weighted': None,

        # Regression (None unless regression)
        'rmse': None,
        'mae': None,
        'r2': None,
        'heteroscedastic': None,
        'residual_plot': None,

        # Clustering (None unless clustering)
        'silhouette_score': None,
        'davies_bouldin': None,
        'cluster_profiles': None,
        'cluster_visualization': None,
        'n_clusters_final': None,

        # Explainability (stub - no SHAP yet)
        'shap_values': None,
        'feature_importance': {},
        'shap_summary_plot': None,
        'shap_waterfall_plot': None,

        # Limitations (stub - full generation in Phase 6)
        'limitations': _stub_limitations(audit, detection, study),

        # Model save
        'model_path': None,
        'model_saved': False,
    }

    # 1: Classification
    if problem_type == 'classification':
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        evaluation['confusion_matrix'] = confusion_matrix(y_te, y_pred).tolist()
        evaluation['classification_report'] = classification_report(y_te, y_pred, output_dict=True)
        evaluation['f1_weighted'] = round(f1_score(y_te, y_pred, average='weighted'), 4)
        evaluation['precision_weighted'] = round(precision_score(y_te, y_pred, average='weighted', zero_division=0), 4)
        evaluation['recall_weighted'] = round(recall_score(y_te, y_pred, average='weighted', zero_division=0), 4)
        evaluation['confusion_matrix_inference'] = (
            f"[STUB] Model evaluated on {len(y_te)} test samples. "
            f"F1 (weighted) = {evaluation['f1_weighted']}."
        )

        if hasattr(model, 'predict_proba'):
            n_classes = detection.get('num_classes', 2)
            if n_classes == 2:
                y_prob = model.predict_proba(X_te)[:, 1]
                evaluation['roc_auc'] = round(roc_auc_score(y_te, y_prob), 4)
            else:
                y_prob = model.predict_proba(X_te)
                evaluation['roc_auc'] = round(roc_auc_score(y_te, y_prob, multi_class='ovr', average='weighted'), 4)

        print(f"  -> F1 (weighted) : {evaluation['f1_weighted']}")
        print(f"  -> ROC-AUC       : {evaluation['roc_auc']}")

    # 2: Regression
    elif problem_type == 'regression':
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        evaluation['rmse'] = round(np.sqrt(mean_squared_error(y_te, y_pred)), 4)
        evaluation['mae'] = round(mean_absolute_error(y_te, y_pred), 4)
        evaluation['r2'] = round(r2_score(y_te, y_pred), 4)
        evaluation['heteroscedastic'] = False  # stub - real check in Phase 6

        print(f"  -> RMSE : {evaluation['rmse']}")
        print(f"  -> MAE  : {evaluation['mae']}")
        print(f"  -> R²   : {evaluation['r2']}")

    # 3: Clustering
    elif problem_type == 'clustering':
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        n_clusters = len(set(labels))
        evaluation['n_clusters_final'] = n_clusters

        if n_clusters > 1:
            evaluation['silhouette_score'] = round(silhouette_score(X, labels), 4)
            evaluation['davies_bouldin'] = round(davies_bouldin_score(X, labels), 4)
        else:
            evaluation['silhouette_score'] = None
            evaluation['davies_bouldin'] = None

        df_cluster = pd.DataFrame(X.values if hasattr(X, 'values') else X,
                                  columns=X.columns if hasattr(X, 'columns') else None)
        df_cluster['cluster'] = labels
        evaluation['cluster_profiles'] = df_cluster.groupby('cluster').mean().round(3).to_dict()

        print(f"  -> Clusters found    : {n_clusters}")
        print(f"  -> Silhouette score  : {evaluation['silhouette_score']}")
        print(f"  -> Davies-Bouldin    : {evaluation['davies_bouldin']}")

    return evaluation


def _stub_limitations(audit, detection, study):
    """Generate a minimal set of limitations. Full version in Phase 6."""
    limitations = []
    n_rows = audit.get('shape', (0,))[0]

    if n_rows < 500:
        limitations.append(f"Very small dataset ({n_rows} rows). Model may overfit.")
    elif n_rows < 2000:
        limitations.append(f"Small dataset ({n_rows} rows). Results may vary across splits.")

    if audit.get('imbalance_detected'):
        ratio = audit.get('imbalance_ratio', 0) * 100
        limitations.append(
            f"Class imbalance present (original minority class: {ratio:.0f}%). "
            f"SMOTE was applied - synthetic samples may not represent real minority instances."
        )

    if len(study.trials) < 20:
        limitations.append(
            f"Only {len(study.trials)} Optuna trials completed (stub). "
            f"Increasing --time-budget may yield better hyperparameters."
        )

    return limitations
