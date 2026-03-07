"""
feature_selector.py - STUB
Phase 2 end-to-end validation stub.
Returns X with quasi-constant and zero-variance columns removed.
Full 3-level implementation (correlation filter + SHAP) comes in Phase 4.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def run_feature_selection(X, y, audit):
    """
    Minimal stub: drops audit-flagged quasi-constants and zero-variance columns.

    Parameters
    ----------
    X : pd.DataFrame       Clean feature matrix from Person 1
    y : pd.Series or None  Target (None for clustering)
    audit : dict           Audit object from Person 1

    Returns
    -------
    X_selected : pd.DataFrame   Filtered feature matrix
    features_dropped : dict     {col_name: reason_string}
    n_remaining : int           Number of features left
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    features_dropped = {}

    print("\n[FEATURE SELECTION]")

    # 1a: Drop audit-flagged quasi-constant columns
    quasi_constant = audit.get('quasi_constant', [])
    to_drop = [c for c in quasi_constant if c in X.columns]
    for col in to_drop:
        features_dropped[col] = 'quasi-constant (audit flagged)'
        print(f"  -> Level 1 - Variance threshold : dropped '{col}' (quasi-constant, audit flagged)")
    X = X.drop(columns=to_drop)

    # 1b: VarianceThreshold on remaining columns
    if len(X.columns) > 0:
        selector = VarianceThreshold(threshold=0.01)
        try:
            selector.fit(X)
            kept_mask = selector.get_support()
            dropped_vt = [col for col, keep in zip(X.columns, kept_mask) if not keep]
            for col in dropped_vt:
                features_dropped[col] = 'zero/near-zero variance'
                print(f"  -> Level 1 - Variance threshold : dropped '{col}' (zero variance)")
            X = X.loc[:, kept_mask]
        except ValueError as e:
            if "No feature in X meets the variance threshold" in str(e):
                for col in X.columns:
                    features_dropped[col] = 'zero/near-zero variance'
                    print(f"  -> Level 1 - Variance threshold : dropped '{col}' (zero variance)")
                X = X.iloc[:, :0]  # Drop all columns
            else:
                print(f"  -> Level 1 - VarianceThreshold skipped: {e}")
        except Exception as e:
            print(f"  -> Level 1 - VarianceThreshold skipped: {e}")

    # Guard: all features dropped
    if len(X.columns) == 0:
        raise ValueError(
            "[FEATURE SELECTION] All features were removed. "
            "Please review data quality - no informative columns remain."
        )

    n_remaining = len(X.columns)
    total = audit.get('shape', (0, 0))[1]
    print(f"  -> Features remaining: {n_remaining} of {total}")
    print(f"  -> Proceeding to Optuna with {n_remaining} features")

    return X, features_dropped, n_remaining
