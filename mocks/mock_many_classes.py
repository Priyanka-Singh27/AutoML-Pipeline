from .mock_generator import make_classification_mock

# 20 classes — triggers 'many_class' subtype warning in detector.py
mock_df = make_classification_mock(
    n_samples=5000, n_features=10, n_classes=20,
    n_informative=10, n_redundant=0, random_state=42
)

mock_audit = {
    'shape': (5000, 10),
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 500,
    'dimensionality_risk': False,

    'target_column': 'target',
    'target_dtype': 'int64',
    'target_unique_values': 20,
    'target_distribution': {i: 0.05 for i in range(20)},
    'target_valid': True,
    'target_min_class_size': 250,

    'column_types': {'numerical_continuous': [f'feature_{i}' for i in range(10)]},
    'missing': {},
    'drop_candidates': {},
    'high_cardinality': [],

    'high_correlations': [],
    'leakage_candidates': {},
    'feature_target_correlation': {
        f'feature_{i}': {'score': round(0.4 - i*0.02, 2), 'signal': 'moderate'} for i in range(10)
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
    'confidence': 'medium',
    'classification_subtype': 'many_class',   # triggers warning
    'num_classes': 20,
    'class_labels': list(range(20)),
    'metrics_averaging': 'weighted',
    'regression_subtype': None,
    'target_log_transform': False,
    'signals': {
        'unique_values': {'value': 20, 'vote': 'classification', 'weight': 1},
    }
}
