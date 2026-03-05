from .mock_generator import make_classification_mock

mock_df = make_classification_mock(n_samples=100000, weights=[0.7, 0.3])

mock_audit = {
    'shape': (100000, 10), 
    'dataset_size_class': 'large',
    'rows_to_features_ratio': 10000, 
    'dimensionality_risk': False,
    
    'target_column': 'target', 
    'target_dtype': 'int64',
    'target_unique_values': 2, 
    'target_distribution': {0: 0.7, 1: 0.3},
    'target_valid': True, 
    'target_min_class_size': 30000,
    
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
    'imbalance_ratio': 0.3,
    'imbalance_severity': 'slight', 
    'smote_recommended': False,
    
    'sampling_recommended': True, 
    'sampling_fraction': 0.2, # We will use this in Optuna to sample the X_train
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
