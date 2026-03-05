import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs

def make_classification_mock(
    n_samples=5000, n_features=10, n_classes=2,
    weights=None, n_informative=6, n_redundant=2, random_state=42
):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_classes=n_classes, weights=weights,
        n_informative=n_informative, n_redundant=n_redundant,
        random_state=random_state
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    return df

def make_regression_mock(n_samples=5000, n_features=10, noise=0.1):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        noise=noise, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    return df

def make_clustering_mock(n_samples=5000, n_clusters=4, n_features=10):
    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=n_clusters, random_state=42
    )
    # No target column for clustering
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    return df

# ── Feature injection helpers ─────────────────────────────────────────────────

def add_correlated_features(df, base_col, n=2, noise=0.05):
    """Injects features that are heavily correlated with an existing column."""
    for i in range(n):
        df[f'{base_col}_corr_{i}'] = df[base_col] + np.random.normal(0, noise, len(df))
    return df

def add_zero_variance_feature(df, col_name='constant_col', value=1):
    """Adds a constant column that should be dropped by VarianceThreshold."""
    df[col_name] = value
    return df

def add_weak_features(df, n_weak=3):
    """Adds pure noise columns that should be dropped by SHAP Level 3."""
    for i in range(n_weak):
        df[f'noise_{i}'] = np.random.normal(0, 1, len(df))
    return df

def add_leakage_feature(df, target_col, noise=0.01):
    """Adds a column that leaks the target."""
    df['leakage_col'] = df[target_col] + np.random.normal(0, noise, len(df))
    return df
