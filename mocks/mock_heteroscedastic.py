import pandas as pd
import numpy as np

# Regression dataset where residuals fan out (heteroscedastic pattern)
# Variance of noise grows with x — evaluator should detect this via Spearman test
np.random.seed(42)
n = 5000
X = np.random.randn(n, 5)
# Noise scales with feature_0 — classic heteroscedasticity
y = (3 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(n) * np.abs(X[:, 0]) * 2)

mock_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
mock_df['target'] = y

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
        'feature_0': {'score': 0.70, 'signal': 'strong'},
        'feature_1': {'score': 0.45, 'signal': 'moderate'},
        'feature_2': {'score': 0.08, 'signal': 'weak'},
        'feature_3': {'score': 0.05, 'signal': 'weak'},
        'feature_4': {'score': 0.02, 'signal': 'none'},
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
    'problem_type': 'regression',
    'detection_method': 'inferred',
    'confidence': 'high',

    'classification_subtype': None,
    'num_classes': None,
    'class_labels': None,
    'metrics_averaging': None,

    'regression_subtype': 'standard',
    'target_log_transform': False,

    'signals': {
        'unique_values': {'value': 5000, 'vote': 'regression', 'weight': 2},
        'dtype': {'value': 'float64', 'vote': 'regression', 'weight': 2},
    }
}
