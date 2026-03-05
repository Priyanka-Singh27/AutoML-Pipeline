from .mock_generator import make_classification_mock

mock_df = make_classification_mock(n_samples=5000, n_classes=4, n_informative=8, n_redundant=0)

mock_audit = {
    'shape': (5000, 10), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 500, 
    'dimensionality_risk': False,
    
    'target_column': 'target', 
    'target_dtype': 'int64',
    'target_unique_values': 4, 
    'target_distribution': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    'target_valid': True, 
    'target_min_class_size': 1250,
    
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
    'classification_subtype': 'multiclass',
    'num_classes': 4, 
    'class_labels': [0, 1, 2, 3],
    'metrics_averaging': 'weighted', 
    'regression_subtype': None,
    'target_log_transform': False,
    'signals': {
        'unique_values': {'value': 4, 'vote': 'classification', 'weight': 2},
    }
}
