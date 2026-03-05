from .mock_generator import make_clustering_mock

mock_df = make_clustering_mock(n_samples=5000, n_clusters=4, n_features=10)

mock_audit = {
    'shape': (5000, 10), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 500, 
    'dimensionality_risk': False,
    
    'target_column': None, 
    'target_dtype': None,
    'target_unique_values': None, 
    'target_distribution': {},
    'target_valid': False, 
    'target_min_class_size': None,
    
    'column_types': {'numerical_continuous': [f'feature_{i}' for i in range(10)]},
    'missing': {}, 
    'drop_candidates': {}, 
    'high_cardinality': [],
    
    'high_correlations': [], 
    'leakage_candidates': {},
    'feature_target_correlation': {},
    
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
    'problem_type': 'clustering', 
    'detection_method': 'inferred',
    'confidence': 'high', 
    
    'classification_subtype': None,
    'num_classes': None, 
    'class_labels': None,
    'metrics_averaging': None, 
    
    'regression_subtype': None,
    'target_log_transform': False,
    
    'signals': {
        'target_column': {'value': None, 'vote': 'clustering', 'weight': 10},
    }
}
