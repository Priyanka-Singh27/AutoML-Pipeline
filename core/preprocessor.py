import pandas as pd
import numpy as np
from core.narrator import narrate
from core.headers import Section
from core.constants import ONEHOT_THRESHOLD, ORDINAL_THRESHOLD

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    # Fallback for Scikit-Learn < 1.3
    TargetEncoder = None

def get_feature_names(ct, input_features):
    """Recover explicit column names from a fitted ColumnTransformer"""
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == 'remainder' or name == 'drop':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                names.extend(transformer.get_feature_names_out(cols))
            except Exception:
                names.extend(cols)
        else:
            names.extend(cols)
    return names

def run_preprocessor(df: pd.DataFrame, audit: dict, args) -> tuple[pd.DataFrame, object]:
    """
    Transforms raw dataframe into standardized numerical matrix.
    Returns:
        df_clean            : pd.DataFrame
            Cleaned, encoded, scaled DataFrame ready for ML processing.
        fitted_preprocessor : sklearn ColumnTransformer
            The fitted transformer pipeline ready for production deployment.
    """
    narrate("\n[PREPROCESSING]")
    df_clean = df.copy()
    
    # --- 1. Remove Duplicates Early ---
    dup_indices = audit.get('duplicate_indices', [])
    if dup_indices:
        narrate(f"  -> Dropping {len(dup_indices)} duplicate rows")
        df_clean = df_clean.drop(index=dup_indices)

    # --- 2. User & Structural Drops ---
    if args.drop:
        requested = [c.strip() for c in args.drop.split(',')]
        not_found = [c for c in requested if c not in df_clean.columns]
        to_drop   = [c for c in requested if c in df_clean.columns]
        
        if not_found:
            narrate(f"  [!] Warning: --drop columns not found: {not_found}")
        if to_drop:
            narrate(f"  -> Dropping user-specified: {to_drop}")
            df_clean = df_clean.drop(columns=to_drop)

    audit_drops = [c for c, reason in audit.get('drop_candidates', {}).items() if c in df_clean.columns]
    if audit_drops:
        narrate(f"  -> Dropping structural pathologies: {audit_drops}")
        df_clean = df_clean.drop(columns=audit_drops)

    target = args.target
    if target and target in df_clean.columns:
        features = [c for c in df_clean.columns if c != target]
        y_train = df_clean[target]
    else:
        features = list(df_clean.columns)
        y_train = None

    problem_type = args.problem
    if not problem_type and target:
        problem_type = 'classification' if df_clean[target].nunique() <= 50 else 'regression'
    elif not problem_type:
        problem_type = 'clustering'

    # --- 3. Outlier Capping (Clip 1st/99th percentile) ---
    for col, info in audit.get('outlier_columns', {}).items():
        if info['severity'] == 'significant' and col in features:
            lower = df_clean[col].quantile(0.01)
            upper = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower, upper)
            narrate(f"  -> '{col}': outliers capped at [{lower:.2f}, {upper:.2f}]")

    # --- 4. Log Transforms (Highly Skewed Numerics) ---
    skewed_cols = [col for col, info in audit.get('skewed_columns', {}).items() 
                   if info.get('action') == 'log_transform' and col in features]
    for col in skewed_cols:
        # log1p handles zero values safely
        df_clean[col] = np.log1p(df_clean[col])
        skew_amt = audit['skewed_columns'][col]['skew']
        narrate(f"  -> '{col}': log transform applied (skew={skew_amt:.2f})")

    # --- 5. Rare Category Grouping ---
    cat_cols = [c for c in audit.get('column_types', {}).get('categorical', []) if c in features]
    for col in cat_cols:
        freq = df_clean[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        if len(rare) > 0:
            df_clean[col] = df_clean[col].replace(rare, '__other__')
            narrate(f"  -> '{col}': grouped {len(rare)} rare categories (<1%) into '__other__'")

    # --- 6. Strict Boolean Type Casting ---
    bool_cols = [c for c in audit.get('column_types', {}).get('boolean', []) if c in features]
    for col in bool_cols:
        narrate(f"  -> '{col}' : Boolean -> int (0/1)")
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, '1': 1, '0': 0, 1: 1, 0: 0, True: 1, False: 0})
        else:
            df_clean[col] = df_clean[col].astype(int)

    # --- 7. Datetime Feature Engineering ---
    dt_cols = [c for c in audit.get('column_types', {}).get('datetime', []) if c in features]
    for col in dt_cols:
        narrate(f"  -> '{col}' : Extracting Datetime Features")
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        df_clean[f"{col}_year"] = df_clean[col].dt.year
        df_clean[f"{col}_month"] = df_clean[col].dt.month
        df_clean[f"{col}_day"] = df_clean[col].dt.day
        df_clean[f"{col}_dow"] = df_clean[col].dt.dayofweek
        df_clean.drop(columns=[col], inplace=True)
        features.remove(col)
        features.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dow"])
    
    # --- 8. Assemble Pipeline Column Definitions ---
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df_clean[c])]
    categorical_features = [c for c in features if c not in numeric_features]
    
    transformers = []
    
    # A. Numeric Transformation
    if numeric_features:
        # Check overall outlier density explicitly logic mapped
        sample_df = df_clean[numeric_features].fillna(0)
        q1 = sample_df.quantile(0.25)
        q3 = sample_df.quantile(0.75)
        iqr = q3 - q1
        outlier_ratio = float(((sample_df < (q1 - 1.5 * iqr)) | (sample_df > (q3 + 1.5 * iqr))).mean().mean())
        
        Scaler = RobustScaler if outlier_ratio > 0.15 else StandardScaler
        narrate(f"  -> Mapping {len(numeric_features)} numerical features ({Scaler.__name__})")
        
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', Scaler())
        ])
        transformers.append(('num', numeric_pipeline, numeric_features))

    # B. Categorical Encoder Allocation
    for col in categorical_features:
        nu = df_clean[col].nunique()
        
        if nu <= ONEHOT_THRESHOLD:
            narrate(f"  -> '{col}' ({nu} unique) : OneHotEncoder")
            step_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append((f'cat_ohe_{col}', step_pipeline, [col]))
            
        elif problem_type in ('classification', 'regression') and y_train is not None and TargetEncoder is not None:
            narrate(f"  -> '{col}' ({nu} unique) : TargetEncoder")
            encoder = TargetEncoder(smooth='auto')
            step_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', encoder)
            ])
            transformers.append((f'cat_target_{col}', step_pipeline, [col]))
            
        elif nu <= ORDINAL_THRESHOLD:
            # Medium cardinality without target — ordinal is safe
            narrate(f"  -> '{col}' ({nu} unique) : OrdinalEncoder (Fallback)")
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            step_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', encoder)
            ])
            transformers.append((f'cat_ordinal_{col}', step_pipeline, [col]))
            
        else:
            # Absolute Fallback (High Cardinals, Cluster Mode or legacy sklearn versions)
            try:
                from category_encoders import HashingEncoder
                narrate(f"  -> '{col}' ({nu} unique) : HashingEncoder (Fallback)")
                encoder = HashingEncoder(cols=[col], n_components=16)
                transformers.append((f'cat_hash_{col}', encoder, [col]))
            except ImportError:
                narrate(f"  [!] category_encoders missing. Forcing {col} to Ordinal.")
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                step_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', encoder)
                ])
                transformers.append((f'cat_ordinal_{col}', step_pipeline, [col]))

    # --- 9. Build and Fit Transformer ---
    preprocessor = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
    
    X_train = df_clean[features]
    narrate("  -> Fitting Scikit-Learn Processor...")
    X_transformed = preprocessor.fit_transform(X_train, y=y_train)
    
    # 10. Robust Feature Name Recovery
    feature_names = get_feature_names(preprocessor, features)
    
    # Pad fallback feature names if count mismatches due to drops/hashes
    if X_transformed.shape[1] != len(feature_names):
        feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]
        
    df_clean_numeric = pd.DataFrame(X_transformed, columns=feature_names, index=X_train.index)
    
    # --- 11. Imbalance Resolution via SMOTE (Only for Classification Training Sets) ---
    if audit.get('smote_recommended') and not args.no_smote and y_train is not None and problem_type == 'classification':
        try:
            narrate(f"  -> Applying SMOTE to heavily imbalanced training set (Target: {target})")
            smote = SMOTE(random_state=args.random_state)
            X_res, y_res = smote.fit_resample(df_clean_numeric, y_train.loc[X_train.index])
            df_clean_numeric = pd.DataFrame(X_res, columns=feature_names)
            y_train = pd.Series(y_res, name=target)
        except Exception as e:
            narrate(f"  [!] SMOTE failed ({str(e)}). Proceeding with original imbalanced distribution.")
    
    if y_train is not None:
        df_clean_numeric[target] = y_train.values  # Align length if SMOTE generated new records
        
    narrate(f"  [+] Preprocessing Complete: {df_clean_numeric.shape[1]} resulting dimensions.")
    
    return df_clean_numeric, preprocessor
