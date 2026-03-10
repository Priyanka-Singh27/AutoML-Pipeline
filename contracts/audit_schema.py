"""
audit_schema.py

Provides the frozen template for the Audit object returned by Person 1's Auditor.
Every downstream pipeline component relies on these exact keys being present.
"""

def get_base_audit_schema() -> dict:
    return {
        # Shape
        "shape":                  (0, 0),
        "dataset_size_class":     "",           # very_small/small/medium/large/very_large
        "memory_mb":              0.0,
        
        # Target
        "target_column":          None,
        "target_dtype":           None,
        "target_unique_values":   None,
        "target_distribution":    None,         # {class: proportion}
        "target_missing":         None,         # proportion
        "target_valid":           False,
        
        # Column types
        "column_types": {
            "numeric":            [],
            "categorical":        [],
            "datetime":           [],
            "boolean":            [],
            "id_like":            [],
            "free_text":          [],
        },
        
        # Pathologies
        "missing":                {},           # {col: float}
        "quasi_constant":         [],
        "high_cardinality":       [],
        "id_columns_candidates":  [],
        "leakage_candidates":     {},           # {col: score}
        "high_correlations":      [],           # [(col_a, col_b, corr)]
        
        # Imbalance
        "imbalance_detected":     False,
        "imbalance_ratio":        None,
        "imbalance_severity":     None,         # balanced/mild/moderate/severe
        "smote_recommended":      False,
        
        # Feature-target correlation
        "feature_target_correlation": {},       # {col: {"score": float, "signal": str}}
        
        # Skewness
        "skewed_columns":         {},           # {col: {"skew": float, "action": str}}
        
        # Outliers
        "outlier_columns":        {},           # {col: {"iqr_pct": float, "zscore_pct": float, "severity": str}}
        
        # Stats
        "stats":                  {},           # {col: {"mean": float, "median": float, "std": float, "min": float, "max": float}}
        
        # Duplicate rows
        "duplicates":             0,            # count of exact duplicate rows
        "duplicate_indices":      [],           # indices for safe removal
        
        # Sampling
        "sampling_recommended":   False,
        "sampling_fraction":      None,
        
        # Drop candidates — single source of truth
        "drop_candidates":        {}            # {col: "reason for dropping"}
    }
