# contracts/preprocessing_contract.md

## run_preprocessor() Interface Contract
**Version:** 1.0  
**Agreed by:** Person 1, Person 2  
**Date:** 2026-03-09

### Input
- df     : pd.DataFrame  — raw loaded CSV
- audit  : dict          — full audit object from run_audit()

### Output
Must return a tuple of exactly two elements:

1. df_clean      : pd.DataFrame
   - All drop candidates removed
   - Missing values imputed
   - Outliers capped
   - Skewed columns log-transformed
   - Datetime features extracted
   - Categorical columns encoded
   - Numerical columns scaled via fitted_scaler
   - SMOTE applied if audit['smote_recommended'] == True
   - Shape: (n_rows_after_dedup, n_features_after_preprocessing)

2. fitted_scaler : sklearn.preprocessing.StandardScaler
   - Fitted on training data only
   - Must be the exact scaler used to produce df_clean
   - Required for inference pipeline stitching in evaluator.py

### Contract Violation
If this function returns anything other than a 2-tuple,
main.py will raise ContractViolationError immediately.
No silent fallbacks. No workarounds.

### Change Protocol
Any change to this interface requires agreement from
Person 1 AND Person 2 before implementation.
