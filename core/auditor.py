import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from core.narrator import narrate
from core.headers import Section
from contracts.audit_schema import get_base_audit_schema
from core.constants import LEAKAGE_THRESHOLD, DEF_LARGE

def run_auditor(df: pd.DataFrame, args) -> dict:
    """
    Analyzes raw dataset and extracts structural metadata.
    Output conforms rigidly to contracts/audit_schema.py keys.
    """
    narrate(f"\n{Section.DATA_AUDIT}")
    
    audit = get_base_audit_schema()
    n_rows, n_cols = df.shape
    narrate(f"  -> Profiling {n_rows} rows and {n_cols} columns")
    
    # --- 1. Basic Metadata & Duplicates ---
    audit['shape'] = (n_rows, n_cols)
    audit['memory_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    if n_rows < 500:
        audit['dataset_size_class'] = 'very_small'
    elif n_rows < 5000:
        audit['dataset_size_class'] = 'small'
    elif n_rows < 50000:
        audit['dataset_size_class'] = 'medium'
    else:
        audit['dataset_size_class'] = 'large'
        
    audit['sampling_recommended'] = (n_rows > DEF_LARGE)
    if audit['sampling_recommended']:
        audit['sampling_fraction'] = round(DEF_LARGE / n_rows, 3)

    # Duplicates check
    dup_mask = df.duplicated()
    audit['duplicates'] = int(dup_mask.sum())
    audit['duplicate_indices'] = list(df[dup_mask].index)
    if audit['duplicates'] > 0:
        narrate(f"  [!] Found {audit['duplicates']} exact duplicate rows.")

    # --- 2. Target Profiling ---
    target = args.target
    audit['target_column'] = target
    feature_cols = list(df.columns)
    
    if target and target in df.columns:
        audit['target_valid'] = True
        feature_cols.remove(target)
        target_series = df[target]
        
        audit['target_dtype'] = str(target_series.dtype)
        audit['target_missing'] = float(target_series.isnull().mean())
        
        valid_targets = target_series.dropna()
        n_unique_targets = valid_targets.nunique()
        audit['target_unique_values'] = n_unique_targets
        
        if n_unique_targets < 50:
            counts = valid_targets.value_counts(normalize=True)
            audit['target_distribution'] = counts.to_dict()
            min_class = counts.min()
            audit['imbalance_detected'] = (min_class < 0.10)
            audit['imbalance_ratio'] = float(min_class)
            
            if min_class < 0.05:
                audit['imbalance_severity'] = 'severe'
                audit['smote_recommended'] = True
            elif min_class < 0.10:
                audit['imbalance_severity'] = 'moderate'
                audit['smote_recommended'] = True
            else:
                audit['imbalance_severity'] = 'balanced'

    # --- 3. Feature Type Binning ---
    for col in feature_cols:
        col_series = df[col]
        
        if pd.api.types.is_bool_dtype(col_series) or \
           set(col_series.dropna().unique()) <= {True, False, 'True', 'False', 1, 0, '1', '0'}:
            audit['column_types']['boolean'].append(col)
            continue
            
        if pd.api.types.is_datetime64_any_dtype(col_series):
            audit['column_types']['datetime'].append(col)
            continue
            
        if col_series.dtype == 'object':
            first_val = col_series.dropna().iloc[0] if not col_series.dropna().empty else None
            if isinstance(first_val, str) and len(first_val) > 8 and ('-' in first_val or '/' in first_val):
                try:
                    pd.to_datetime(first_val)
                    audit['column_types']['datetime'].append(col)
                    continue
                except (ValueError, TypeError):
                    pass
            
        if pd.api.types.is_numeric_dtype(col_series):
            audit['column_types']['numeric'].append(col)
        else:
            audit['column_types']['categorical'].append(col)

    # --- 4. Missing Sweep & Base Stats ---
    for col in feature_cols:
        series = df[col]
        missing_ratio = float(series.isnull().mean())
        if missing_ratio > 0:
            audit['missing'][col] = missing_ratio
            if missing_ratio > 0.90:
                audit['drop_candidates'][col] = "missing > 90%"
                narrate(f"  [!] '{col}' missing {missing_ratio*100:.1f}% data (Drop candidate)")

        if col in audit['column_types']['numeric']:
            valid = series.dropna()
            if not valid.empty:
                audit['stats'][col] = {
                    "mean": float(valid.mean()),
                    "median": float(valid.median()),
                    "std": float(valid.std()),
                    "min": float(valid.min()),
                    "max": float(valid.max())
                }

    # --- 5. Cardinality, Pathologies, Outliers, Skew ---
    for col in feature_cols:
        nunique = df[col].nunique()
        series = df[col].dropna()
        if series.empty: continue
        
        # Quasi-constant
        if nunique <= 1 or (nunique == 2 and series.value_counts(normalize=True).iloc[0] > 0.99):
            audit['quasi_constant'].append(col)
            audit['drop_candidates'][col] = "quasi-constant"
            narrate(f"  [!] '{col}' is quasi-constant (Drop candidate)")
            continue
            
        # ID Columns
        if nunique == len(series) and df[col].dtype in ['object', 'int64']:
            audit['id_columns_candidates'].append(col)
            audit['drop_candidates'][col] = "id-like column"
            narrate(f"  [!] '{col}' looks like an ID column (Drop candidate)")
            continue
            
        # Categorical High Cardinality
        if col in audit['column_types']['categorical'] and nunique > 15:
            audit['high_cardinality'].append(col)
            
        # Numeric checks: Skewness & Outliers
        if col in audit['column_types']['numeric']:
            # Outliers (IQR)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
            outlier_pct = float(outlier_mask.mean())
            
            if outlier_pct > 0.05:
                audit['outlier_columns'][col] = {
                    "iqr_pct": outlier_pct,
                    "zscore_pct": 0.0, # Approximate, relying on IQR for speed
                    "severity": "significant" if outlier_pct > 0.1 else "moderate"
                }

            # Skewness
            s_val = float(skew(series))
            if abs(s_val) > 1.0:
                # Log-transform only if strictly positive
                action = 'log_transform' if series.min() >= 0 else 'none'
                audit['skewed_columns'][col] = {"skew": s_val, "action": action}

    # --- 6. Correlation & Leakage (Subsampled for speed) ---
    sample_df = df if n_rows < 10000 else df.sample(10000, random_state=42)
    
    # Inter-feature numeric correlation
    numeric_cols = [c for c in audit['column_types']['numeric'] if c not in audit['drop_candidates']]
    if len(numeric_cols) > 1:
        corr_matrix = sample_df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper_tri.columns:
            highly_correlated_with = upper_tri.index[upper_tri[col] > 0.95].tolist()
            for target_col in highly_correlated_with:
                audit['high_correlations'].append((col, target_col, float(upper_tri.loc[target_col, col])))
                narrate(f"  [!] '{col}' & '{target_col}' highly correlated (>0.95)")

    # Target Leakage Detection
    if audit['target_valid']:
        y_samp = sample_df[target].dropna()
        samp_indices = y_samp.index
        
        # A. Numerical Leakage using Spearman
        for col in numeric_cols:
            x_samp = sample_df.loc[samp_indices, col]
            if not x_samp.empty and x_samp.nunique() > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(x_samp, y_samp)
                if abs(corr) > LEAKAGE_THRESHOLD:
                    audit['leakage_candidates'][col] = float(abs(corr))
                    audit['feature_target_correlation'][col] = {"score": float(abs(corr)), "signal": "spearman"}
                    audit['drop_candidates'][col] = f"target leakage (Spearman={abs(corr):.2f})"
                    narrate(f"  [!] '{col}' shows severe leakage (Spearman={abs(corr):.2f})")
                    
        # B. Categorical Leakage using Mutual Information
        cat_cols = [c for c in audit['column_types']['categorical'] if c not in audit['drop_candidates']]
        if cat_cols and audit['target_unique_values'] < 50:
            X_cat = sample_df.loc[samp_indices, cat_cols].fillna('missing')
            # Fast ordinal encode for MI calculation
            from sklearn.preprocessing import OrdinalEncoder
            X_ord = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X_cat)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi_scores = mutual_info_classif(X_ord, y_samp, random_state=42)
                
            for i, col in enumerate(cat_cols):
                mi_norm = mi_scores[i] / np.log(len(y_samp)) if len(y_samp) > 1 else 0  # roughly normalized
                
                if mi_norm > LEAKAGE_THRESHOLD:
                    audit['leakage_candidates'][col] = float(mi_norm)
                    audit['feature_target_correlation'][col] = {"score": float(mi_norm), "signal": "mutual_info"}
                    audit['drop_candidates'][col] = f"target leakage (MI={mi_norm:.2f})"
                    narrate(f"  [!] '{col}' shows severe categorical leakage (MI={mi_norm:.2f})")

    # --- 7. User Drop Overrides ---
    if args.drop:
        requested_drops = [c.strip() for c in args.drop.split(',')]
        # Explicit removes from candidate list if user already manually selected it 
        # (This stops feature selector from failing if it drops it twice)
        for c in requested_drops:
            audit['drop_candidates'].pop(c, None)

    narrate(f"  -> Found {len(audit['column_types']['numeric'])} numeric, {len(audit['column_types']['categorical'])} categorical features.")
    
    return audit
