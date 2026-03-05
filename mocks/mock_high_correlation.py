from .mock_generator import make_classification_mock, add_correlated_features

mock_df = make_classification_mock(n_samples=5000)
mock_df = add_correlated_features(mock_df, base_col='feature_0', n=2, noise=0.02)

mock_audit = {
    'shape': (5000, 12), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 416, 
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
    
    'high_correlations': [
        ('feature_0', 'feature_0_corr_0', 0.97),
        ('feature_0', 'feature_0_corr_1', 0.95),
    ], 
    'leakage_candidates': {},
    'feature_target_correlation': {
        'feature_0': {'score': 0.61, 'signal': 'strong'},
        'feature_0_corr_0': {'score': 0.12, 'signal': 'weak'}, # should be dropped
        'feature_0_corr_1': {'score': 0.09, 'signal': 'weak'}, # should be dropped
        **{f'feature_{i}': {'score': round(0.5 - i*0.03, 2), 'signal': 'moderate'} for i in range(1, 10)}
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
