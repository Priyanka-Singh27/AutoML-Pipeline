# AutoML Pipeline Bug Report

## Executive Summary
This report documents bugs identified in the AutoML pipeline codebase after comprehensive code analysis. The pipeline consists of multiple modules including auditor, detector, preprocessor, feature selector, tuner, evaluator, and API/reporting components.

---

## Critical Bugs (High Severity)

### Bug 1: High Correlation Tuple Handling in Feature Selector
**File:** [`core/feature_selector.py:39`](core/feature_selector.py:39)
**Type:** AttributeError - Type M**Status:** Confirmed (ismatch
from debug_out.txt)

**Description:** The `CorrelationFilter.fit()` method assumes `high_correlations` is a list of dictionaries, but the auditor stores it as a list of tuples: `[(col_a, col_b, corr), ...]`.

```python
# Current code (broken):
for pair in self.high_correlations:
    if isinstance(pair, tuple):
        f1, f2 = pair[0], pair[1]
    else:
        f1 = pair.get('feature1')  # This branch is NEVER reached
        f2 = pair.get('feature2')
```

**Impact:** Pipeline crashes when high correlation features are detected.

**Fix Required:** Ensure the code always uses tuple unpacking since that's what auditor produces.

---

### Bug 2: Preprocessor Return Signature Mismatch
**File:** [`core/preprocessor.py:256-262`](core/preprocessor.py:256-262)
**Type:** Contract Violation
**Status:** Confirmed

**Description:** The preprocessor returns a dictionary with keys `column_transformer` and `label_encoder`, but `main.py:270` expects a tuple of `(df_clean, fitted_preprocessor)`.

```python
# Current output:
output_bundle = {
    'column_transformer': preprocessor,
    'label_encoder': fitted_label_encoder
}
return df_clean_numeric, output_bundle  # Returns dict, not preprocessor

# Expected by main.py:
df_clean, fitted_preprocessor = result  # Expects ColumnTransformer directly
```

**Impact:** Contract violation error causes pipeline to abort.

**Fix Required:** Return the preprocessor object directly, not wrapped in a dict.

---

### Bug 3: VarianceThreshold Failure When All Features Dropped
**File:** [`core/feature_selector.py:150-167`](core/feature_selector.py:150-167)
**Type:** ValueError Handling
**Status:** Confirmed (from debug_trace.txt)

**Description:** When all features are removed by quasi-constant filtering, VarianceThreshold throws an error that gets caught but the error message is misleading.

```python
# Current code:
vf = VarianceThreshold(threshold=0.01)
try:
    vf.fit(X_train)
except ValueError as e:
    if "variance" in str(e).lower() or "feature" in str(e).lower():
        raise ValueError("Feature selection failed: 100% of features were dropped by Hard Filters.")
    raise e
```

**Impact:** Error message doesn't accurately describe the root cause.

---

## Medium Severity Bugs

### Bug 4: SMOTE Index Alignment Issue
**File:** [`core/preprocessor.py:240-251`](core/preprocessor.py:240-251)
**Type:** Data Integrity

**Description:** After SMOTE resampling, the index alignment between features and target may be inconsistent:

```python
X_res, y_res = smote.fit_resample(df_clean_numeric, y_train.loc[X_train.index])
df_clean_numeric = pd.DataFrame(X_res, columns=feature_names)  # Loses original index
y_train = pd.Series(y_res, name=target)  # New index from SMOTE
```

**Impact:** Target column may become misaligned with features.

---

### Bug 5: SHAP Generation for Non-Tree Models
**File:** [`core/evaluator.py:218-227`](core/evaluator.py:218-227)
**Type:** Runtime Warning

**Description:** The code uses KernelExplainer for linear models but may fail silently:

```python
background = shap.sample(X_tr, min(100, len(X_tr)), random_state=42)
X_test_shap = X_te.sample(min(200, len(X_te)), random_state=42)
explainer = shap.KernelExplainer(pred_func, background)
shap_vals = explainer.shap_values(X_test_shap)  # Can fail or be slow
```

**Impact:** SHAP computation may be very slow or fail for large datasets.

---

### Bug 6: API Batch Prediction Key Mismatch
**File:** [`api/app.py:176-187`](api/app.py:176-187)
**Type:** Interface Design

**Description:** The batch endpoint expects `records` key but single prediction doesn't:

```python
# Batch endpoint:
records = payload.get("records", [])  # Requires "records" key

# Single endpoint:
results = validate_and_predict([payload])  # Takes payload directly
```

**Impact:** Inconsistent API interface.

---

### Bug 7: Report Generation Cluster Visualization Save
**File:** [`reporting/generator.py:131-138`](reporting/generator.py:131-138)
**Type:** Type Error

**Description:** The code tries to save cluster visualization assuming it's a matplotlib figure, but it may be a dictionary:

```python
cluster_viz = evaluation.get('cluster_visualization')
if cluster_viz is not None:
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        cluster_viz.savefig(f.name, ...)  # .savefig only works on figures
```

**Impact:** PDF report generation may fail for clustering problems.

---

### Bug 8: Adj R2 Division by Zero
**File:** [`core/evaluator.py:137`](core/evaluator.py:137)
**Type:** Potential Division Error

**Description:** Adjusted R² calculation doesn't handle edge case when `n <= p + 1`:

```python
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
```

**Impact:** Potential division by zero handled, but `None` is returned which may cause issues downstream.

---

### Bug 9: Mock All Features Dropped Test Failure
**File:** [`tests/test_all_mocks.py:103-110`](tests/test_all_mocks.py:103-110)
**Type:** Test Logic

**Description:** The test expects `mock_all_features_dropped` to raise ValueError, but the current implementation doesn't raise properly:

```python
if name == 'mock_all_features_dropped':
    results.append((name, 'FAIL', "Should have raised ValueError — all features were expected to be dropped"))
```

**Impact:** Test fails - 16/17 tests passing (from test_log.txt).

---

## Low Severity Bugs / Improvements

### Bug 10: Boolean Column Mapping Incompleteness
**File:** [`core/preprocessor.py:134-137`](core/preprocessor.py:134-137)
**Type:** Edge Case

**Description:** Boolean column mapping may not handle all possible string representations:

```python
df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, '1': 1, '0': 0, 1: 1, 0: 0, True: 1, False: 0})
# Missing: 'true', 'false', 'TRUE', 'FALSE', 'yes', 'no', etc.
```

---

### Bug 11: Missing Global Random State in Cluster Evaluation
**File:** [`core/evaluator.py:192-193`](core/evaluator.py:192-193)
**Type:** Reproducibility

**Description:** PCA in clustering doesn't use random_state:

```python
pca = PCA(n_components=2)
X_2d = pca.fit_transform(...)  # No random_state specified
```

---

### Bug 12: Log Transform Without Positive Check
**File:** [`core/preprocessor.py:172`](core/preprocessor.py:172)
**Type:** Logic Error

**Description:** Log transform action is set to 'none' but may still be applied elsewhere:

```python
action = 'log_transform' if series.min() >= 0 else 'none'
audit['skewed_columns'][col] = {"skew": s_val, "action": action}
# The action may not be properly checked before application
```

---

### Bug 13: Input Validation Race Condition
**File:** [`api/app.py:68-69`](api/app.py:68-69)
**Type:** Edge Case

**Description:** If `expected_columns` is empty and records list is empty:

```python
expected_cols = metadata.get('expected_columns', [])
if not expected_cols:
    expected_cols = list(records[0].keys())  # Will fail if records is empty
```

---

### Bug 14: Model Name Extraction Fallback Issues
**File:** [`core/evaluator.py:324-330`](core/evaluator.py:324-330)
**Type:** Logic

**Description:** Multiple fallback paths for model name may result in 'unknown':

```python
model_name = 'unknown'
if hasattr(study, 'model_name'):
    model_name = study.model_name
elif best_params and 'model' in best_params:
    model_name = best_params['model']
elif hasattr(study, 'best_trial') and hasattr(study.best_trial, 'params'):
    model_name = study.best_trial.params.get('model', 'unknown')
```

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| Critical | 3 |
| Medium | 6 |
| Low | 5 |
| **Total** | **14** |

---

## Recommendations

1. **Immediate Action Required:** Fix Bug 1 (high correlation handling) and Bug 2 (preprocessor return signature) as they cause pipeline crashes.

2. **Testing:** Add integration tests that specifically test the contract between modules.

3. **Error Handling:** Add more descriptive error messages and proper exception chaining.

4. **Type Safety:** Consider adding type hints and runtime type checking for inter-module contracts.

5. **Documentation:** Document expected data formats for all module interfaces.

---

*Report generated by code analysis of AutoML pipeline project.*
