import pandas as pd
import numpy as np

# Target has been log-transformed by Person 1. This tests that detector.py
# sets target_log_transform=True and evaluator.py back-transforms predictions
# before computing RMSE/MAE (otherwise metrics are on a log scale).
np.random.seed(42)
n = 5000
X = np.random.randn(n, 5)
# Simulate an already log-transformed target (originally skewed salary-like data)
y_original = np.random.lognormal(mean=10, sigma=0.5, size=n)
y_log_transformed = np.log(y_original)  # Person 1 applied this

mock_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
mock_df['target'] = y_log_transformed

mock_audit = {
    'shape': (5000, 5),
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 1000,
    'dimensionality_risk': False,

    'target_column': 'target',
    'target_dtype': 'float64',
    'target_unique_values': 5000,
    'target_distribution': {},
    'target_valid': True,
    'target_min_class_size': None,

    'column_types': {'numerical_continuous': [f'feature_{i}' for i in range(5)]},
    'missing': {},
    'drop_candidates': {},
    'high_cardinality': [],

    'high_correlations': [],
    'leakage_candidates': {},
    'feature_target_correlation': {
        f'feature_{i}': {'score': round(0.4 - i*0.05, 2), 'signal': 'moderate'} for i in range(5)
    },

    'skewed_columns': {'target': {'skew': 3.8, 'action': 'log_transform'}},
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
    'problem_type': 'regression',
    'detection_method': 'inferred',
    'confidence': 'high',

    'classification_subtype': None,
    'num_classes': None,
    'class_labels': None,
    'metrics_averaging': None,

    'regression_subtype': 'standard',
    'target_log_transform': True,  # <-- triggers back-transform in evaluator.py

    'signals': {
        'unique_values': {'value': 5000, 'vote': 'regression', 'weight': 2},
        'dtype': {'value': 'float64', 'vote': 'regression', 'weight': 2},
    }
}
