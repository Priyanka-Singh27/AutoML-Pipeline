import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Only one feature remains after aggressive feature selection
X, y = make_classification(n_samples=5000, n_features=1, n_informative=1,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
mock_df = pd.DataFrame({'feature_0': X.ravel(), 'target': y})

mock_audit = {
    'shape': (5000, 1),
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 5000,
    'dimensionality_risk': False,

    'target_column': 'target',
    'target_dtype': 'int64',
    'target_unique_values': 2,
    'target_distribution': {0: 0.5, 1: 0.5},
    'target_valid': True,
    'target_min_class_size': 2500,

    'column_types': {'numerical_continuous': ['feature_0']},
    'missing': {},
    'drop_candidates': {},
    'high_cardinality': [],

    'high_correlations': [],
    'leakage_candidates': {},
    'feature_target_correlation': {
        'feature_0': {'score': 0.75, 'signal': 'strong'},
    },

    'skewed_columns': {},
    'outlier_columns': {},
    'quasi_constant': [],

    'imbalance_detected': False,
    'imbalance_ratio': None,
    'imbalance_severity': None,
    'smote_recommended': False,

    'sampling_recommended': False,
    'sampling_fraction': None,
}

mock_detection = {
    'problem_type': 'classification',
    'detection_method': 'inferred',
    'confidence': 'high',
    'classification_subtype': 'binary',
    'num_classes': 2,
    'class_labels': [0, 1],
    'metrics_averaging': 'binary',
    'regression_subtype': None,
    'target_log_transform': False,
    'signals': {
        'unique_values': {'value': 2, 'vote': 'classification', 'weight': 2},
    }
}
