"""
feature_selector.py
Reduces dimensionality before Optuna tuning using rigorous stability protocols.
Implements Leakage removal, VarianceThreshold, Consensus Importance (Multi-seed SHAP),
and Correlation resolution.
"""

from core.narrator import narrate
from core.headers import Section


import numpy as np
import pandas as pd
import shap
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor

# Ensure pandas output for all sklearn transformers to maintain column names
import sklearn
sklearn.set_config(transform_output="pandas")


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drops the weaker of two highly correlated features based on Consensus Importance.
    """
    def __init__(self, high_correlations, consensus_importance):
        self.high_correlations = high_correlations
        self.consensus_importance = consensus_importance
        self.to_drop_ = []

    def fit(self, X, y=None):
        drop_set = set()
        for pair in self.high_correlations:
            f1 = pair.get('feature1')
            f2 = pair.get('feature2')
            if f1 not in X.columns or f2 not in X.columns:
                continue
            
            # Retrieve consensus importance (default to 0 if not found)
            score1 = self.consensus_importance.get(f1, {'mean': 0.0})['mean']
            score2 = self.consensus_importance.get(f2, {'mean': 0.0})['mean']
            
            loser = f1 if score1 < score2 else f2
            drop_set.add(loser)
            
        self.to_drop_ = list(drop_set)
        return self

    def transform(self, X):
        cols_to_drop = [c for c in self.to_drop_ if c in X.columns]
        if not cols_to_drop:
            return X
        return X.drop(columns=cols_to_drop)


class WeakFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Drops features whose Consensus Importance falls below a relative threshold.
    """
    def __init__(self, consensus_importance, relative_threshold=0.01):
        self.consensus_importance = consensus_importance
        self.relative_threshold = relative_threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        if not self.consensus_importance:
            return self
            
        max_imp = max([v['mean'] for v in self.consensus_importance.values()]) if self.consensus_importance else 0.0
        abs_threshold = max_imp * self.relative_threshold

        drop_set = set()
        for col in X.columns:
            imp_dict = self.consensus_importance.get(col)
            if not imp_dict:
                continue
            
            mean_imp = imp_dict['mean']
            # Drop if mean importance is below threshold
            if mean_imp < abs_threshold:
                drop_set.add(col)
                
        self.to_drop_ = list(drop_set)
        return self

    def transform(self, X):
        cols_to_drop = [c for c in self.to_drop_ if c in X.columns]
        if not cols_to_drop:
            return X
        return X.drop(columns=cols_to_drop)


def _split_data(X, y, problem_type, test_size, random_state):
    """Splits data strictly according to problem type."""
    if problem_type == 'clustering':
        return X.copy(), None, None, None
        
    stratify = y if problem_type == 'classification' else None
    
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=random_state
        )
        return X_tr, X_te, y_tr, y_te
    except ValueError as e:
        # Fallback if classes are too small to stratify
        narrate(f"  [!] Stratified split failed ({str(e)}). Falling back to random split.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )
        return X_tr, X_te, y_tr, y_te


def _remove_leakage(X_train, X_test, audit, dropped_dict):
    """Drops features flagged as leakage candidates in the audit object."""
    candidates = audit.get('leakage_candidates', {})
    leakage_cols = [c for c in candidates if c in X_train.columns]
    
    if leakage_cols:
        narrate(f"  -> Level 0 | Dropping {len(leakage_cols)} leakage candidates: {leakage_cols}")
        X_train = X_train.drop(columns=leakage_cols)
        if X_test is not None:
            X_test = X_test.drop(columns=leakage_cols)
        dropped_dict['leakage'] = leakage_cols
        
    return X_train, X_test


def _apply_variance_filter(X_train, X_test, audit, dropped_dict):
    """Drops quasi-constant features explicitly, then fits VarianceThreshold."""
    # Explicit quasi-constant from audit
    quasi_cols = [c for c in audit.get('quasi_constant', []) if c in X_train.columns]
    if quasi_cols:
        narrate(f"  -> Level 1 | Dropping {len(quasi_cols)} quasi-constant columns from audit: {quasi_cols}")
        X_train = X_train.drop(columns=quasi_cols)
        if X_test is not None:
            X_test = X_test.drop(columns=quasi_cols)
        dropped_dict['quasi_constant'] = quasi_cols

    # Fit VarianceThreshold for safety netting
    vf = VarianceThreshold(threshold=0.01)
    try:
        vf.fit(X_train)
        dropped_vars = [c for c in X_train.columns if c not in vf.get_feature_names_out()]
        if dropped_vars:
            narrate(f"  -> Level 1 | Dropping {len(dropped_vars)} low variance columns: {dropped_vars}")
            dropped_dict['variance'] = dropped_vars
            
        X_train = vf.transform(X_train)
        if X_test is not None:
            X_test = vf.transform(X_test)
    except ValueError:
        # Happens if ALL features are zero variance
        pass

    return X_train, X_test, vf


def _calculate_consensus_importance(X_train, y_train, detection, base_random_state=42):
    """Trains 3 LGBM probes to generate a stable Consensus SHAP Importance array."""
    problem_type = detection.get('problem_type')
    n_classes = detection.get('num_classes', 2)
    
    if problem_type == 'clustering' or X_train.shape[1] <= 1:
        return {}
        
    narrate(f"  -> Level 2 | Calculating Consensus Importance (3 seeds)")
    importances = []
    seeds = [base_random_state, base_random_state + 42, base_random_state + 123]
    
    # Suppress lightgbm warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    for seed in seeds:
        if problem_type == 'classification':
            obj = 'binary' if n_classes == 2 else 'multiclass'
            probe = LGBMClassifier(n_estimators=50, random_state=seed, verbosity=-1, objective=obj, 
                                   num_class=n_classes if n_classes > 2 else None)
        else:
            probe = LGBMRegressor(n_estimators=50, random_state=seed, verbosity=-1)
            
        try:
            probe.fit(X_train, y_train)
            explainer = shap.TreeExplainer(probe)
            # handle multidimensional output for multiclass gracefully
            shap_values = explainer.shap_values(X_train)
            if isinstance(shap_values, list): 
                # Multiclass SHAP returns list of arrays, sum abs across classes
                seed_imp = np.zeros(X_train.shape[1])
                for cl_vals in shap_values:
                    seed_imp += np.abs(cl_vals).mean(axis=0)
            else:
                seed_imp = np.abs(shap_values).mean(axis=0)
                
            importances.append(seed_imp)
        except Exception as e:
            narrate(f"  [!] Failed to generate SHAP importance for seed {seed}: {type(e).__name__}")
            continue

    if not importances:
        return {}
        
    mean_imp = np.mean(importances, axis=0)
    std_imp = np.std(importances, axis=0)
    
    consensus = {}
    for i, col in enumerate(X_train.columns):
        consensus[col] = {'mean': mean_imp[i], 'std': std_imp[i]}
        
    return consensus


def _post_selection_validation(X_train_full, X_train_sel, y_train, detection, random_state=42):
    """Compares cross-val metrics of full vs selected feature sets."""
    problem = detection.get('problem_type')
    if problem == 'clustering' or X_train_full.shape[1] <= 1:
        return
        
    n_classes = detection.get('num_classes', 2)
    
    narrate(f"  -> Level 5 | Post-Selection Validation (3-Fold CV)")
    if problem == 'classification':
        obj = 'binary' if n_classes == 2 else 'multiclass'
        probe = LGBMClassifier(n_estimators=30, random_state=random_state, verbosity=-1, objective=obj, 
                               num_class=n_classes if n_classes > 2 else None)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scoring = 'f1_weighted'
    else:
        probe = LGBMRegressor(n_estimators=30, random_state=random_state, verbosity=-1)
        cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scoring = 'neg_root_mean_squared_error'
        
    try:
        # Need to re-align indices if we dropped rows, but we only dropped columns
        score_full = cross_val_score(probe, X_train_full, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
        score_sel = cross_val_score(probe, X_train_sel, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
        
        # Regression scores are negative RMSE
        if problem == 'regression':
            score_full = -score_full
            score_sel = -score_sel
            
            if score_sel > score_full * 1.05:  # 5% worse error
                narrate(f"  [!] Warning: Selection increased RMSE from {score_full:.4f} to {score_sel:.4f}.")
            else:
                narrate(f"  -> Validation: RMSE remained stable ({score_full:.4f} -> {score_sel:.4f})")
        else:
            if score_sel < score_full - 0.02: # 2% worse F1
                narrate(f"  [!] Warning: Selection reduced F1 from {score_full:.4f} to {score_sel:.4f}.")
            else:
                narrate(f"  -> Validation: F1 remained stable ({score_full:.4f} -> {score_sel:.4f})")
                
    except Exception as e:
        narrate(f"  [!] Could not generate validation scores: {type(e).__name__}")


def run_feature_selection(X, y, audit, detection, test_size=0.2, random_state=42):
    """
    Executes the robust 5-Level feature selection pipeline.
    """
    narrate("\n[FEATURE SELECTION]")
    
    n_original = X.shape[1]
    dropped_dict = {'leakage': [], 'quasi_constant': [], 'variance': [], 'correlation': [], 'shap': []}
    
    # 0. Data Splitting
    X_train, X_test, y_train, y_test = _split_data(X, y, detection['problem_type'], test_size, random_state)
    X_train_full_backup = X_train.copy() # For post-level validation later
    
    # Level 0 + Level 1: Hard Filters
    X_train, X_test = _remove_leakage(X_train, X_test, audit, dropped_dict)
    X_train, X_test, vf = _apply_variance_filter(X_train, X_test, audit, dropped_dict)
    
    # Early Stop Guard
    if X_train.shape[1] == 0:
        raise ValueError("Feature selection failed: 100% of features were dropped by Hard Filters.")
        
    # Level 2: Consensus Importance
    consensus_imp = _calculate_consensus_importance(X_train, y_train, detection, random_state)
    
    # Level 3: Correlation Resolution
    cf = CorrelationFilter(audit.get('high_correlations', []), consensus_imp)
    cf.fit(X_train)
    if cf.to_drop_:
        narrate(f"  -> Level 3 | Dropping {len(cf.to_drop_)} weaker correlated columns: {cf.to_drop_}")
        dropped_dict['correlation'] = cf.to_drop_
        X_train = cf.transform(X_train)
        if X_test is not None:
            X_test = cf.transform(X_test)
            
    # Level 4: Weak Feature Deletion (SHAP)
    wf = WeakFeatureFilter(consensus_imp, relative_threshold=0.01)
    wf.fit(X_train)
    if wf.to_drop_:
        narrate(f"  -> Level 4 | Dropping {len(wf.to_drop_)} weak features (SHAP < 1% of top): {wf.to_drop_}")
        dropped_dict['shap'] = wf.to_drop_
        X_train = wf.transform(X_train)
        if X_test is not None:
            X_test = wf.transform(X_test)
            
    # Final Pipeline Build
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('variance', vf),
        ('correlation', cf),
        ('weak_shap', wf)
    ])
    
    # Final Checks
    n_remaining = X_train.shape[1]
    if n_remaining == 0:
        raise ValueError("Feature selection failed: 100% of features were dropped across all filters.")
        
    narrate(f"  -> Features remaining: {n_remaining} / {n_original}")
    
    # Level 5: Post Selection Validation
    _post_selection_validation(X_train_full_backup, X_train, y_train, detection, random_state)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'pipeline': pipeline,
        'selected_features': list(X_train.columns),
        'dropped_features': dropped_dict,
        'features_original': n_original,
        'features_remaining': n_remaining
    }
