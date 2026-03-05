from .mock_generator import make_regression_mock

mock_df = make_regression_mock(n_samples=5000, n_features=10)

mock_audit = {
    'shape': (5000, 10), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 500, 
    'dimensionality_risk': False,
    
    'target_column': 'target', 
    'target_dtype': 'float64',
    'target_unique_values': 5000, 
    'target_distribution': {},
    'target_valid': True, 
    'target_min_class_size': None,
    
    'column_types': {'numerical_continuous': [f'feature_{i}' for i in range(10)]},
    'missing': {}, 
    'drop_candidates': {}, 
    'high_cardinality': [],
    
    'high_correlations': [], 
    'leakage_candidates': {},
    'feature_target_correlation': {
        f'feature_{i}': {
            'score': round(0.5 - i*0.04, 2), 
            'signal': 'strong' if i < 3 else 'moderate'
        } for i in range(10)
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
        'distribution': {'value': 'continuous', 'vote': 'regression', 'weight': 1},
    }
}
