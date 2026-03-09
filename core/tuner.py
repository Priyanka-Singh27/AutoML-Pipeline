"""
tuner.py
Handles Optuna hyperparameter optimization and algorithm search, adhering to 
Single Responsibility, robust trial-level error handling, and dispatch architecture.
"""

from core.narrator import narrate
from core.headers import Section


import time
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Suppress chatty warnings for cleaner console narration
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _compute_scale_pos_weight(y):
    """Calculates scaling weight for binary imbalance mathematically."""
    if y is None or len(np.unique(y)) != 2:
        return None
        
    val_counts = pd.Series(y).value_counts()
    if len(val_counts) == 2:
        maj_count = val_counts.iloc[0] 
        min_count = val_counts.iloc[1] 
        if min_count > 0:
            return float(maj_count / min_count)
    return None

def _get_time_callback(time_budget, start_time):
    """Returns a callback tracking time and narrating progression."""
    def narration_callback(study, trial):
        elapsed = time.time() - start_time
        remaining = max(0, time_budget - elapsed)
        
        is_best = False
        try:
            if study.best_value is not None:
                is_best = (trial.value == study.best_value)
        except ValueError:
            pass # No best value yet / failed trials
            
        marker = "  <- new best" if is_best else ""
        model = trial.params.get('model', 'unknown')
        
        # Format duration safely
        duration = trial.duration.total_seconds() if trial.duration else 0.0
        score_val = trial.value if trial.value is not None else float('-inf')
        
        narrate(f"  Trial {trial.number:>3} | {model:<20} | "
              f"Score={score_val:.4f} | "
              f"[{duration:.1f}s] | ~{remaining:.0f}s remaining{marker}")
        
        if 0 < remaining < 15:
            narrate(f"  -> Time budget nearly exhausted — wrapping up...")
            
    return narration_callback


# ---------------------------------------------------------
# Objective Builder
# ---------------------------------------------------------

def build_objective(X_train, y_train, detection, audit, random_state=42):
    """
    Returns an Optuna objective closure with explicit constraints and CV.
    """
    problem_type = detection['problem_type']
    n_rows, n_cols = X_train.shape
    
    # Dataset Constraints Configuration
    size_class = audit.get('dataset_size_class', 'medium')
    imbalance = audit.get('imbalance_detected', False)
    scale_weight = _compute_scale_pos_weight(y_train) if imbalance and problem_type == 'classification' else None
    
    # Deterministic CV splits
    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial):
        # 1. Model Pool Construction
        if problem_type == 'classification':
            pool = ['lightgbm', 'xgboost', 'random_forest', 'logistic_regression']
            if n_rows <= 10000:
                pool.append('svc')
            elif trial.number == 0:
                narrate("  -> Removed SVC from pool (dataset > 10,000 rows)")
                
            model_name = trial.suggest_categorical('model', pool)
        else: # regression
            pool = ['lightgbm', 'xgboost', 'random_forest', 'ridge']
            if n_rows <= 10000:
                pool.append('svr')
            elif trial.number == 0:
                narrate("  -> Removed SVR from pool (dataset > 10,000 rows)")
                
            model_name = trial.suggest_categorical('model', pool)

        # 2. Extract Hyperparameters explicitly via separated functions to prevent spaghetti code inside objective
        model = _instantiate_model_for_trial(trial, model_name, problem_type, size_class, random_state, scale_weight)

        # 3. Cross Validate with Pruning logic directly in fold evaluation
        scores = []
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_f_tr, y_f_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_f_va, y_f_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
                
                model.fit(X_f_tr, y_f_tr)
                preds = model.predict(X_f_va)
                
                if problem_type == 'classification':
                    score = f1_score(y_f_va, preds, average='weighted')
                else:
                    # Maximizing negative RMSE 
                    score = -np.sqrt(mean_squared_error(y_f_va, preds))
                    
                scores.append(score)
                running_mean = np.mean(scores)
                
                # Interactively report status back to MedianPruner
                trial.report(running_mean, step=fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            return np.mean(scores)
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            # Swallow crash to not kill the overarching study, returning absolute worst score
            narrate(f"  [!] Trial {trial.number} failed ({model_name}): {type(e).__name__} - {str(e)}")
            return float('-inf')

    return objective, scale_weight

def _instantiate_model_for_trial(trial, model_name, problem_type, size_class, random_state, scale_weight):
    """Suggests params for a given framework mathematically constrained by the audit metadata."""
    if model_name == 'lightgbm':
        lr = trial.suggest_float('lgbm_lr', 1e-3, 0.1, log=True)
        n_est = trial.suggest_int('lgbm_n', 50, 200 if size_class == 'very_small' else 500)
        leaves = trial.suggest_int('lgbm_leaves', 20, 31 if size_class == 'very_small' else 100)
        min_child = trial.suggest_int('lgbm_min_child', 10, 50)
        
        if problem_type == 'classification':
            return LGBMClassifier(learning_rate=lr, n_estimators=n_est, num_leaves=leaves, 
                                  min_child_samples=min_child, random_state=random_state, 
                                  scale_pos_weight=scale_weight, verbosity=-1)
        return LGBMRegressor(learning_rate=lr, n_estimators=n_est, num_leaves=leaves, 
                             min_child_samples=min_child, random_state=random_state, verbosity=-1)

    elif model_name == 'xgboost':
        lr = trial.suggest_float('xgb_lr', 1e-3, 0.1, log=True)
        max_depth = trial.suggest_int('xgb_depth', 3, 4 if size_class == 'very_small' else 9)
        n_est = trial.suggest_int('xgb_n', 50, 200 if size_class == 'very_small' else 500)
        subsample = trial.suggest_float('xgb_sub', 0.5, 1.0)
        colsample = trial.suggest_float('xgb_col', 0.5, 1.0)
        
        if problem_type == 'classification':
            return XGBClassifier(learning_rate=lr, max_depth=max_depth, n_estimators=n_est,
                                 subsample=subsample, colsample_bytree=colsample,
                                 random_state=random_state, scale_pos_weight=scale_weight, verbosity=0)
        return XGBRegressor(learning_rate=lr, max_depth=max_depth, n_estimators=n_est,
                            subsample=subsample, colsample_bytree=colsample,
                            random_state=random_state, verbosity=0)

    elif model_name == 'random_forest':
        n_est = trial.suggest_int('rf_n', 50, 300)
        max_depth = trial.suggest_int('rf_depth', 3, 15)
        min_samples = trial.suggest_int('rf_min', 2, 10)
        
        if problem_type == 'classification':
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, 
                                          min_samples_split=min_samples, random_state=random_state, n_jobs=-1)
        return RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, 
                                     min_samples_split=min_samples, random_state=random_state, n_jobs=-1)

    elif model_name == 'logistic_regression':
        c = trial.suggest_float('lr_c', 1e-4, 10.0, log=True)
        return LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=random_state)
        
    elif model_name == 'ridge':
        alpha = trial.suggest_float('ridge_alpha', 1e-3, 10.0, log=True)
        return Ridge(alpha=alpha, random_state=random_state)
        
    elif model_name == 'svc':
        c = trial.suggest_float('svc_c', 1e-3, 10.0, log=True)
        return SVC(C=c, random_state=random_state)
        
    elif model_name == 'svr':
        c = trial.suggest_float('svr_c', 1e-3, 10.0, log=True)
        return SVR(C=c)
        
    raise ValueError(f"Unknown model_name in objective: {model_name}")


# ---------------------------------------------------------
# Post-Study Rebuilding Dispatchers
# ---------------------------------------------------------

def _build_lightgbm(params, random_state, scale_weight, problem_type):
    if problem_type == 'classification':
        return LGBMClassifier(learning_rate=params['lgbm_lr'], n_estimators=params['lgbm_n'],
                              num_leaves=params['lgbm_leaves'], min_child_samples=params['lgbm_min_child'],
                              random_state=random_state, scale_pos_weight=scale_weight, verbosity=-1)
    return LGBMRegressor(learning_rate=params['lgbm_lr'], n_estimators=params['lgbm_n'],
                         num_leaves=params['lgbm_leaves'], min_child_samples=params['lgbm_min_child'],
                         random_state=random_state, verbosity=-1)

def _build_xgboost(params, random_state, scale_weight, problem_type):
    if problem_type == 'classification':
        return XGBClassifier(learning_rate=params['xgb_lr'], max_depth=params['xgb_depth'],
                             n_estimators=params['xgb_n'], subsample=params['xgb_sub'],
                             colsample_bytree=params['xgb_col'], random_state=random_state,
                             scale_pos_weight=scale_weight, verbosity=0)
    return XGBRegressor(learning_rate=params['xgb_lr'], max_depth=params['xgb_depth'],
                        n_estimators=params['xgb_n'], subsample=params['xgb_sub'],
                        colsample_bytree=params['xgb_col'], random_state=random_state, verbosity=0)

def _build_random_forest(params, random_state, scale_weight, problem_type):
    if problem_type == 'classification':
        return RandomForestClassifier(n_estimators=params['rf_n'], max_depth=params['rf_depth'],
                                      min_samples_split=params['rf_min'], random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(n_estimators=params['rf_n'], max_depth=params['rf_depth'],
                                 min_samples_split=params['rf_min'], random_state=random_state, n_jobs=-1)

def _build_logistic_regression(params, random_state, scale_weight, problem_type):
    return LogisticRegression(C=params['lr_c'], solver='lbfgs', max_iter=1000, random_state=random_state)

def _build_ridge(params, random_state, scale_weight, problem_type):
    return Ridge(alpha=params['ridge_alpha'], random_state=random_state)

def _build_svc(params, random_state, scale_weight, problem_type):
    return SVC(C=params['svc_c'], random_state=random_state)

def _build_svr(params, random_state, scale_weight, problem_type):
    return SVR(C=params['svr_c'])

def rebuild_model(best_params, problem_type, random_state, scale_weight):
    """
    Dispatches architecture construction to independent functions.
    This guarantees explicit structural decoupling—meaning adding generic new 
    models like CatBoost requires exactly one function addition.
    """
    builders = {
        'lightgbm': _build_lightgbm,
        'xgboost': _build_xgboost,
        'random_forest': _build_random_forest,
        'logistic_regression': _build_logistic_regression,
        'ridge': _build_ridge,
        'svc': _build_svc,
        'svr': _build_svr
    }
    model_name = best_params['model']
    return builders[model_name](best_params, random_state, scale_weight, problem_type)


# ---------------------------------------------------------
# Clustering Flow Exception
# ---------------------------------------------------------

def _run_clustering(X, random_state, _auto_input):
    """Fully distinct mathematical discovery layer for Clustering data."""
    narrate("  -> Clustering mode detected. Initiating pattern discovery...")
    n_samples = len(X)
    
    # K-Means Auto-Detect
    max_k = min(11, max(3, n_samples // 50))
    k_range = list(range(2, max_k + 1))
    silhouettes_km = []
    inertias = []
    
    narrate(f"  -> Testing K-Means (k=2 to {max_k})...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        inertias.append(km.inertia_)
        silhouettes_km.append(silhouette_score(X, km.labels_))
        
    best_k_sil = k_range[np.argmax(silhouettes_km)]
    
    # Simple elbow math (distance from curve point to chord line)
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    distances = []
    for i, k in enumerate(k_range):
        p3 = np.array([k, inertias[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
    best_k_elb = k_range[np.argmax(distances)]
    
    narrate(f"     Elbow suggests: {best_k_elb} | Silhouette suggests: {best_k_sil}")
    target_k = best_k_sil
    if best_k_sil != best_k_elb:
        narrate("     [!] Methods disagree. Falling back to Silhouette mechanically.")
        
    if _auto_input is None:
        user_k = input(f"  -> Press Enter to accept [{target_k}], or manually specify K: ").strip()
        if user_k.isdigit():
            target_k = int(user_k)
            narrate(f"     User override accepted. Using K={target_k}.")
            
    # Baseline K-Means Candidate
    km_final = KMeans(n_clusters=target_k, random_state=random_state, n_init=10).fit(X)
    best_score = silhouette_score(X, km_final.labels_)
    best_model = km_final
    best_name = f"KMeans (k={target_k})"
    
    # DBSCAN k-NN Auto-Detect
    narrate(f"  -> Testing DBSCAN (k-NN distance estimated eps)...")
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    distances_nn, _ = nn.kneighbors(X)
    eps_est = np.percentile(distances_nn[:, -1], 90) # Robust density threshold
    
    for min_s in [3, 5, 10]:
        db = DBSCAN(eps=eps_est, min_samples=min_s).fit(X)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        
        # Guard against zero clusters or single massive noise clusters
        if n_clusters > 1:
            try:
                score = silhouette_score(X, db.labels_)
                if score > best_score:
                    best_score = score
                    best_model = db
                    best_name = f"DBSCAN (eps={eps_est:.2f}, min={min_s})"
            except ValueError:
                pass 
                
    # Agglomerative Grid
    narrate(f"  -> Testing Agglomerative Hierarchical (target k={target_k})...")
    for linkage in ['ward', 'complete', 'average']:
        agg = AgglomerativeClustering(n_clusters=target_k, linkage=linkage).fit(X)
        try:
            score = silhouette_score(X, agg.labels_)
            if score > best_score:
                best_score = score
                best_model = agg
                best_name = f"Agglomerative ({linkage})"
        except ValueError:
            pass
            
    narrate(f"  [RESULT] Selected {best_name} with silhouette = {best_score:.4f}")
    
    # Wrapper pattern required to conform to Optuna contract
    class _StubStudy:
        def __init__(self, val, mname):
            self.best_value = val
            self.trials = [1, 2, 3] 
            self.model_name = mname
        @property
        def best_trial(self):
            class _T:
                params = {'model': mname}
                value = self.best_value
            return _T()
            
    return _StubStudy(best_score, best_name), best_model


# ---------------------------------------------------------
# Main Export
# ---------------------------------------------------------

def run_optuna_study(X, y, detection, audit, time_budget=120, random_state=42, _auto_input=None):
    """Execution entrypoint strictly isolating the Tuning layers."""
    narrate(f"\n[OPTUNA TUNING]")
    problem_type = detection['problem_type']
    
    if problem_type == 'clustering':
        return _run_clustering(X, random_state, _auto_input)
        
    start_time = time.time()
    
    # 1. Deterministic Pruner & Sampler
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    
    study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
    
    # 2. Scope constraints
    objective, scale_weight = build_objective(X, y, detection, audit, random_state)
    
    # 3. Optimize (Timeout execution safely governed by Catch mechanism wrapper)
    try:
        study.optimize(
            objective, 
            timeout=time_budget, 
            n_jobs=1,
            callbacks=[_get_time_callback(time_budget, start_time)]
        )
    except Exception as e:
        narrate(f"  [!] Study execution failed structurally: {e}")
        
    # 4. Dispatch construction mechanism
    elapsed = time.time() - start_time
    if len(study.trials) == 0 or study.best_value == float('-inf'):
        raise RuntimeError("Optuna study completed 0 successful trials.")
        
    narrate(f"\n  -> Found best model: {study.best_trial.params['model']} "
          f"(Score: {study.best_value:.4f}) in {elapsed:.1f}s")
          
    narrate(f"  -> Retraining optimized parameters strictly across entire dataset...")
    final_model = rebuild_model(study.best_trial.params, problem_type, random_state, scale_weight)
    final_model.fit(X, y)
    
    # Shim to fulfill Evaluator contract bounds
    study.model_name = study.best_trial.params['model']
    
    return study, final_model
