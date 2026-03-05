from .mock_generator import make_classification_mock

# 95/5 imbalance ratio
mock_df = make_classification_mock(n_samples=5000, weights=[0.95, 0.05])

mock_audit = {
    'shape': (5000, 10), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 500, 
    'dimensionality_risk': False,
    
    'target_column': 'target', 
    'target_dtype': 'int64',
    'target_unique_values': 2, 
    'target_distribution': {0: 0.95, 1: 0.05},
    'target_valid': True, 
    'target_min_class_size': 250,
    
    'column_types': {'numerical_continuous': [f'feature_{i}' for i in range(10)]},
    'missing': {}, 
    'drop_candidates': {}, 
    'high_cardinality': [],
    
    'high_correlations': [], 
    'leakage_candidates': {},
    'feature_target_correlation': {
        f'feature_{i}': {'score': round(0.5 - i*0.03, 2), 'signal': 'moderate'} for i in range(10)
    },
    
    'skewed_columns': {}, 
    'outlier_columns': {}, 
    'quasi_constant': [],
    
    'imbalance_detected': True, 
    'imbalance_ratio': 0.05,
    'imbalance_severity': 'severe', 
    'smote_recommended': True,
    
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
