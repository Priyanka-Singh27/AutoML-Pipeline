import pandas as pd
import numpy as np

# Create completely noisy, useless dataframe
mock_df = pd.DataFrame({
    'constant_1': [1] * 5000,
    'constant_2': [0] * 5000,
    'noise_1': np.random.normal(0, 0.00001, 5000), # near-zero variance
    'noise_2': np.random.normal(0, 0.00001, 5000), 
    'target': np.random.randint(0, 2, 5000),
})

mock_audit = {
    'shape': (5000, 5), 
    'dataset_size_class': 'medium',
    'rows_to_features_ratio': 1000, 
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
        f'noise_{i}': {'score': 0.00, 'signal': 'weak'} for i in range(1, 3)
    },
    
    'skewed_columns': {}, 
    'outlier_columns': {}, 
    'quasi_constant': ['constant_1', 'constant_2'], 
    
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
