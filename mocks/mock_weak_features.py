from .mock_generator import make_classification_mock, add_weak_features

mock_df = make_classification_mock(n_samples=5000)
# adding 3 pure noise features
mock_df = add_weak_features(mock_df, n_weak=3) 

mock_audit = {
    'shape': (5000, 13), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 384, 
    'dimensionality_risk': False,
    
    'target_column': 'target', 
    'target_dtype': 'int64',
    'target_unique_values': 2, 
    'target_distribution': {0: 0.5, 1: 0.5},
    'target_valid': True, 
    'target_min_class_size': 2500,
    
    'column_types': {'numerical_continuous': list(mock_df.columns.drop('target'))},
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
