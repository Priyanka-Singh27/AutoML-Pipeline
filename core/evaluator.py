"""
evaluator.py
End-to-end model evaluation, interpretation, and explanation phase.
Generates metrics, confusion matrix analysis, SHAP feature importances, 
limitations, and stitches the final inference pipeline.
"""

from core.narrator import narrate
from core.headers import Section


import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import joblib
import shap


def _interpret_confusion_matrix(cm, class_labels, audit, problem_subtype):
    """Generates human-readable insights from the confusion matrix."""
    insights = []
    
    # False negative rate per class
    fn_rates = []
    for i in range(len(class_labels)):
        total = cm[i].sum()
        fn = total - cm[i][i]
        fn_rates.append(fn / total if total > 0 else 0)
        
    worst_idx = np.argmax(fn_rates)
    worst_class = class_labels[worst_idx] if class_labels is not None else str(worst_idx)
    worst_rate  = fn_rates[worst_idx] * 100
    
    insights.append(
        f"Model struggles most with class '{worst_class}' — "
        f"misclassifying {worst_rate:.1f}% of actual instances."
    )
    
    # Binary specific - false positive rate & thresholds
    if problem_subtype == 'binary' and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        insights.append(
            f"False positive rate: {fpr*100:.1f}% — "
            f"model incorrectly flags {fpr*100:.1f}% of negative cases."
        )
        if fn_rates[worst_idx] > 0.3:
            insights.append(
                "High false-negative rate detected. Consider lowering decision "
                "threshold below 0.5 if missing positive cases is costly."
            )
            
    if audit.get('imbalance_detected'):
        insights.append(
            "Note: class imbalance was present in training data — "
            "minority class performance may be unreliable."
        )
        
    return " ".join(insights)


def _evaluate_classification(model, X_tr, X_te, y_tr, y_te, detection, audit, evaluation):
    """Handles logic for Classification evaluation."""
    n_classes = detection.get('num_classes', 2)
    class_labels = detection.get('class_labels')
    if class_labels is None:
        class_labels = [str(i) for i in range(n_classes)] if n_classes else ['0', '1']
    subtype = detection.get('classification_subtype', 'binary')
    
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    cm = confusion_matrix(y_te, y_pred)
    evaluation['confusion_matrix'] = cm.tolist()
    evaluation['classification_report'] = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    evaluation['f1_weighted'] = round(f1_score(y_te, y_pred, average='weighted'), 4)
    evaluation['precision_weighted'] = round(precision_score(y_te, y_pred, average='weighted', zero_division=0), 4)
    evaluation['recall_weighted'] = round(recall_score(y_te, y_pred, average='weighted', zero_division=0), 4)
    
    evaluation['insights'] = _interpret_confusion_matrix(cm, class_labels, audit, subtype)
    
    # Probability processing (SVC fallback)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_te)
        except Exception:
            pass
            
    if y_prob is None and hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_te)
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
            # Normalize binary decision bounds
            denom = y_prob.max() - y_prob.min() + 1e-10
            y_prob = (y_prob - y_prob.min()) / denom
        else:
            # Normalize multiclass decision bounds array
            y_min = y_prob.min(axis=1, keepdims=True)
            y_max = y_prob.max(axis=1, keepdims=True)
            y_prob = (y_prob - y_min) / (y_max - y_min + 1e-10)
            
    # ROC-AUC Processing
    if y_prob is not None:
        try:
            if n_classes == 2:
                y_prob_bin = y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob
                evaluation['roc_auc'] = round(roc_auc_score(y_te, y_prob_bin), 4)
            else:
                evaluation['roc_auc'] = round(roc_auc_score(y_te, y_prob, multi_class='ovr', average='weighted'), 4)
        except ValueError as e:
            evaluation['roc_auc'] = None
            narrate(f"  [!] ROC-AUC could not be computed on subset: {e}")
            
    narrate(f"  -> F1 (weighted) : {evaluation['f1_weighted']}")
    narrate(f"  -> ROC-AUC       : {evaluation['roc_auc']}")
    

def _evaluate_regression(model, X_tr, X_te, y_tr, y_te, detection, audit, evaluation):
    """Handles logic for Regression evaluation."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    r2 = r2_score(y_te, y_pred)
    n = len(y_te)
    p = X_te.shape[1]
    
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
    
    # MAPE cleanly bypassing zero division 
    mape = np.mean(np.abs((y_te - y_pred) / np.where(y_te == 0, 1e-10, y_te))) * 100
    
    residuals = y_te - y_pred
    # Heteroscedasticity test (correlation between pred and abs(res))
    corr, p_val = spearmanr(y_pred, np.abs(residuals))
    
    evaluation['rmse'] = round(np.sqrt(mean_squared_error(y_te, y_pred)), 4)
    evaluation['mae'] = round(mean_absolute_error(y_te, y_pred), 4)
    evaluation['r2'] = round(r2, 4)
    evaluation['adj_r2'] = round(adj_r2, 4) if adj_r2 else None
    evaluation['mape'] = round(mape, 4)
    evaluation['residuals'] = residuals.tolist()
    evaluation['heteroscedastic'] = bool(p_val < 0.05 and abs(corr) > 0.3)
    
    insights = []
    insights.append(f"Model errors average ~{evaluation['mape']:.1f}% relative to actual values.")
    if evaluation['heteroscedastic']:
        insights.append(f"Errors scale with target magnitude (Heteroscedasticity detected). Reliability weakens at extremes.")
    evaluation['insights'] = " ".join(insights)
    
    narrate(f"  -> RMSE : {evaluation['rmse']}")
    narrate(f"  -> MAE  : {evaluation['mae']} (Off by {evaluation['mape']:.1f}%)")
    narrate(f"  -> Adj R² : {evaluation['adj_r2']}")


def _evaluate_clustering(model, X, detection, audit, evaluation):
    """Handles logic for Clustering evaluation."""
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
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

    # Robust profiles including density bounds
    means = df_cluster.groupby('cluster').mean().round(3)
    stds = df_cluster.groupby('cluster').std().round(3)
    sizes = df_cluster.groupby('cluster').size().rename('n_samples')
    
    profiles = pd.concat([sizes, means.add_suffix('_mean'), stds.add_suffix('_std')], axis=1)
    evaluation['cluster_profiles'] = profiles.to_dict(orient='index')
    
    # PCA projection & variance bounds
    try:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(df_cluster.drop('cluster', axis=1))
        explained = sum(pca.explained_variance_ratio_)
        evaluation['cluster_visualization'] = {"x": X_2d[:,0].tolist(), "y": X_2d[:,1].tolist(), "labels": labels.tolist(), "explained_var": explained}
        
        insights = [f"Data segmented into {n_clusters} clusters. "]
        if explained < 0.5:
            insights.append(f"Warning: PCA 2D captures only {explained*100:.1f}% of variance. Visual cluster rendering may not represent true geometric structure.")
        evaluation['insights'] = " ".join(insights)
    except Exception as e:
        evaluation['cluster_visualization'] = None
        
    # Replace non-ascii chars that break Windows console logs
    narrate(f"  -> Clusters found    : {n_clusters}")
    narrate(f"  -> Silhouette score  : {evaluation['silhouette_score']}")


def _generate_shap(model, X_tr, X_te, detection, evaluation, model_name):
    """Robust SHAP implementation handling Kernel limits and Multiclass array lists."""
    problem_type = detection['problem_type']
    if problem_type == 'clustering':
        return
        
    narrate(f"  -> Generating Model Explainability (SHAP)...")
    
    try:
        if model_name in ['svc', 'svr', 'logistic_regression', 'ridge']:
            narrate("     [!] Computing SHAP via KernelExplainer. Sampling background to preserve time budget.")
            background = shap.sample(X_tr, min(100, len(X_tr)), random_state=42)
            X_test_shap = X_te.sample(min(200, len(X_te)), random_state=42)
            
            # For KernelExplainer, use predict function
            pred_func = getattr(model, 'predict_proba', getattr(model, 'predict'))
            explainer = shap.KernelExplainer(pred_func, background)
            shap_vals = explainer.shap_values(X_test_shap)
            used_cols = X_test_shap.columns
            shap_data = X_test_shap
        else:
            # Tree-based
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_te)
            used_cols = X_te.columns
            shap_data = X_te

        evaluation['shap_explainer'] = explainer
        
        # Multiclass TreeExplainer returns a list of arrays (one per class)
        if isinstance(shap_vals, list):
            n_classes = len(shap_vals)
            if problem_type == 'classification' and n_classes == 2:
                # Binary - just take class 1 target SHAP
                mean_abs = np.abs(shap_vals[1]).mean(axis=0)
            else:
                # Multiclass - Average absolute impact across all classes to find globally important features
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        elif len(np.shape(shap_vals)) == 3:
            n_classes = np.shape(shap_vals)[2]
            if problem_type == 'classification' and n_classes == 2:
                mean_abs = np.abs(shap_vals[:, :, 1]).mean(axis=0)
            else:
                mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            # Standard Regression or collapsed binary output
            mean_abs = np.abs(shap_vals).mean(axis=0)
            
        # Map to feature names
        imp_dict = {}
        for idx, col in enumerate(used_cols):
            imp_dict[col] = round(float(mean_abs[idx]), 5)
            
        evaluation['shap_feature_importance'] = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))
        
        import matplotlib.pyplot as plt
        fig_summary, ax_summary = plt.subplots(figsize=(8, 6))
        plt.figure(fig_summary.number)
        
        if isinstance(shap_vals, list):
            shap.summary_plot(shap_vals, shap_data, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_vals, shap_data, show=False)
            
        evaluation['shap_summary_plot'] = fig_summary
        evaluation['shap_waterfall_plot'] = None
        
    except Exception as e:
        narrate(f"  [!] SHAP generation failed: {type(e).__name__} - {str(e)}")


def _generate_limitations(audit, detection, study, evaluation):
    """Analyzes audit metadata to append strict warnings over data quality limitations."""
    limitations = []
    n_rows = audit.get('shape', (0,))[0]
    
    if n_rows < 500:
        limitations.append(f"Very small dataset ({n_rows} rows). Model may be highly susceptible to overfitting.")
    elif n_rows < 2000:
        limitations.append(f"Small dataset ({n_rows} rows). Out-of-sample performance bounds may vary.")
        
    if audit.get('imbalance_detected'):
        ratio = audit.get('imbalance_ratio', 0) * 100
        limitations.append(f"Class imbalance present (original minority class: {ratio:.0f}%). Minority class precision/recall bounds may be volatile.")
        
    if audit.get('high_cardinality_detected') and n_rows < 5000:
        limitations.append("High cardinality columns were target-encoded. On small datasets this encoding may be noisy and could cause overfitting.")
        
    if audit.get('datetime_columns'):
        limitations.append("Datetime features were extracted. Model performance explicitly assumes future data follows the exact historical temporal bounds seen here.")
        
    if detection['problem_type'] == 'clustering':
        limitations.append("Clustering results are sensitive to hyperparameters. Segmentations map relative geometric distance densities, not objective empirical truth.")
        
    if getattr(study, 'model_name', '').lower() in ['svc', 'svr', 'logistic_regression']:
        limitations.append("SHAP values were estimated using a sampled KernelExplainer, making them mathematical approximations rather than exact derivations.")
        
    evaluation['limitations'] = limitations


def run_evaluation(model, X, y, detection, audit, study, preprocessor_pipeline=None, feature_selector_pipeline=None, run_shap=True):
    """
    Main entrypoint enforcing single responsibility over model reporting.
    Returns explicit dict fulfilling downstream serialization contracts.
    """
    narrate(f"\n[EVALUATION]")
    problem_type = detection['problem_type']
    
    # Safely extract study metadata regardless of mock/tuple structures
    best_params = {}
    if hasattr(study, 'best_params'):
        best_params = study.best_params
    elif isinstance(study, dict) and 'best_params' in study:
        best_params = study['best_params']
        
    model_name = 'unknown'
    if hasattr(study, 'model_name'):
        model_name = study.model_name
    elif best_params and 'model' in best_params:
        model_name = best_params['model']
    elif hasattr(study, 'best_trial') and hasattr(study.best_trial, 'params'):
        model_name = study.best_trial.params.get('model', 'unknown')
        
    n_trials = 0
    if hasattr(study, 'trials'):
        n_trials = len(study.trials)

    evaluation = {
        'best_model': model,
        'best_model_name': model_name,
        'best_params': best_params,
        'n_trials': n_trials,
        
        'features_original': audit.get('shape', (0, 0))[1],
        'features_remaining': X.shape[1],
        'features_dropped': audit.get('features_dropped', {}),

        'insights': "",
        'limitations': [],

        'confusion_matrix': None,
        'classification_report': None,
        'roc_auc': None,
        'f1_weighted': None,
        'precision_weighted': None,
        'recall_weighted': None,

        'rmse': None,
        'mae': None,
        'r2': None,
        'adj_r2': None,
        'mape': None,
        'heteroscedastic': None,
        'residuals': None,

        'n_clusters_final': None,
        'silhouette_score': None,
        'davies_bouldin': None,
        'cluster_profiles': None,
        'cluster_visualization': None,
        
        'shap_values': None,
        'shap_feature_importance': {},
        'shap_explainer': None,
        
        'model_saved': False
    }

    if problem_type == 'classification':
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        _evaluate_classification(model, X_tr, X_te, y_tr, y_te, detection, audit, evaluation)
    elif problem_type == 'regression':
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        _evaluate_regression(model, X_tr, X_te, y_tr, y_te, detection, audit, evaluation)
    elif problem_type == 'clustering':
        _evaluate_clustering(model, X, detection, audit, evaluation)
        
    if run_shap and problem_type != 'clustering':
        _generate_shap(model, X_tr, X_te, detection, evaluation, model_name)
        
    _generate_limitations(audit, detection, study, evaluation)
    
    # Bundle full pipeline explicitly if components provided
    steps = []
    if preprocessor_pipeline:
        steps.append(('preprocessor', preprocessor_pipeline))
    if feature_selector_pipeline:
        steps.append(('feature_selector', feature_selector_pipeline))
    steps.append(('model', model))
    
    evaluation['full_pipeline'] = Pipeline(steps) if len(steps) > 1 else model

    return evaluation
