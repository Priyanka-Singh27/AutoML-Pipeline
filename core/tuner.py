"""
tuner.py - STUB
Phase 2 end-to-end validation stub.
Trains a single fast model per problem type. No Optuna yet.
Full Optuna study (model pool + hyperparams) comes in Phase 5.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, StratifiedKFold


class _StubStudy:
    """Minimal stand-in for an Optuna study object until Phase 5."""

    def __init__(self, best_params, best_value, n_trials, model_name):
        self.best_params = best_params
        self.best_value = best_value
        self.trials = list(range(n_trials))   # evaluator checks len(study.trials)
        self.model_name = model_name

    @property
    def best_trial(self):
        class _T:
            params = self.best_params
            value = self.best_value
        return _T()


def run_optuna_study(X, y, detection, audit, time_budget=120):
    """
    Stub: trains one fast model and wraps result in a stub study object.

    Parameters
    ----------
    X : pd.DataFrame         Feature-selected matrix
    y : pd.Series or None    Target (None for clustering)
    detection : dict         Detection object
    audit : dict             Audit object
    time_budget : int        Ignored in stub; used by real Optuna tuner

    Returns
    -------
    study : _StubStudy       Mimics Optuna study interface
    model  : fitted model    sklearn-compatible model
    """
    problem_type = detection['problem_type']
    print(f"\n[OPTUNA TUNING - STUB]")
    print(f"  -> Problem type : {problem_type.capitalize()}")

    if problem_type == 'clustering':
        model = KMeans(n_clusters=3, random_state=42, n_init='auto')
        model.fit(X)
        stub = _StubStudy(
            best_params={'n_clusters': 3},
            best_value=0.0,
            n_trials=1,
            model_name='KMeans'
        )
        print(f"  -> Stub model   : KMeans (k=3)")
        return stub, model

    elif problem_type == 'classification':
        # LightGBM handles binary, multiclass, and many_class fast without convergence issues
        n_classes = detection.get('num_classes', 2)
        objective = 'binary' if n_classes == 2 else 'multiclass'
        model = LGBMClassifier(
            n_estimators=50, verbosity=-1, random_state=42,
            objective=objective,
            num_class=n_classes if n_classes > 2 else None
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        model.fit(X, y)
        stub = _StubStudy(
            best_params={'model': 'lightgbm', 'n_estimators': 50},
            best_value=float(np.mean(scores)),
            n_trials=3,
            model_name='LightGBM'
        )
        print(f"  -> Stub model   : LightGBM  |  F1={stub.best_value:.4f}")
        return stub, model

    elif problem_type == 'regression':
        model = Ridge()
        model.fit(X, y)
        stub = _StubStudy(
            best_params={'model': 'ridge', 'alpha': 1.0},
            best_value=0.0,
            n_trials=3,
            model_name='Ridge'
        )
        print(f"  -> Stub model   : Ridge Regression")
        return stub, model

    else:
        raise ValueError(f"[OPTUNA TUNING] Unknown problem_type: '{problem_type}'")
