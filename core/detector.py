"""
detector.py
Infers problem type (classification, regression, clustering) from audit data and statistical tests on the target distribution.
"""

from core.narrator import narrate
from core.headers import Section

import numpy as np
import pandas as pd
from scipy import stats

def analyze_target_distribution(target_values):
    """
    Runs statistical tests on the target column to gather evidence for classification vs regression.
    """
    # Filter out NaNs
    target_values = np.asarray(target_values).flatten()
    try:
        if pd.api.types.is_numeric_dtype(target_values):
            target_values = target_values[~np.isnan(target_values)]
        else:
            target_values = target_values[pd.notna(target_values)]
    except:
        pass

    if len(target_values) == 0:
        return {'is_normal': False, 'p_normal': 0.0, 'is_integer_valued': False, 'entropy': 0.0, 'gap_variance': 0.0, 'is_numeric': False}

    # Test 1 - Normality test
    # A normally distributed target strongly suggests regression
    try:
        # Require target values to be numeric for normality tests
        if np.issubdtype(target_values.dtype, np.number):
            if len(target_values) < 3:
                is_normal = False
                p_normal = 0.0
            elif len(target_values) <= 5000:
                _, p_normal = stats.shapiro(target_values)
                is_normal = p_normal > 0.05
            else:
                _, p_normal = stats.normaltest(target_values)
                is_normal = p_normal > 0.05
        else:
            is_normal = False
            p_normal = 0.0
    except Exception:
        is_normal = False
        p_normal = 0.0

    # Test 2 - Are values integers disguised as floats?
    try:
        if np.issubdtype(target_values.dtype, np.number):
            is_integer_valued = np.all(target_values == np.floor(target_values))
        else:
            is_integer_valued = False
    except Exception:
        is_integer_valued = False

    # Test 3 - Entropy of value distribution
    # High entropy = spread out = regression signal
    # Low entropy = concentrated in few values = classification signal
    try:
        value_counts = pd.Series(target_values).value_counts(normalize=True).values
        entropy = stats.entropy(value_counts)
    except Exception:
        entropy = 0.0
        
    # Test 4 - Gap analysis
    # Sort unique values and check gaps between them
    # Large irregular gaps = continuous = regression
    # Small regular gaps = discrete = classification
    try:
        if np.issubdtype(target_values.dtype, np.number):
            unique_sorted = np.sort(np.unique(target_values))
            gaps = np.diff(unique_sorted)
            gap_variance = np.var(gaps) if len(gaps) > 0 else 0.0
            is_numeric = True
        else:
            gap_variance = 0.0
            is_numeric = False
    except Exception:
        gap_variance = 0.0
        is_numeric = False

    return {
        'is_normal': is_normal,
        'p_normal': p_normal,
        'is_integer_valued': is_integer_valued,
        'entropy': entropy,
        'gap_variance': gap_variance,
        'is_numeric': is_numeric
    }

def feature_type_signal(audit):
    """
    Uses the proportion of continuous vs categorical features as a weak supporting signal.
    """
    col_types = audit.get('column_types', {})
    n_continuous = len(col_types.get('numerical_continuous', []))
    n_categorical = len(col_types.get('categorical_nominal', []) + 
                        col_types.get('categorical_ordinal', []))
    total = n_continuous + n_categorical
    if total == 0:
        return 'neutral', 0.0
    continuous_ratio = n_continuous / total
    if continuous_ratio > 0.75:
        return 'regression', 0.5
    elif continuous_ratio < 0.25:
        return 'classification', 0.5
    return 'neutral', 0.0

def is_genuinely_ambiguous(n_unique, target_dtype, dist_analysis):
    """
    Determines if the problem type genuinely needs user confirmation.
    """
    # String/Boolean/Categorical is definitely classification
    if target_dtype in ['object', 'bool', 'category', 'string']:
        return False
    # Truely continuous floats are definitely regression
    if target_dtype in ['float64', 'float32'] and not dist_analysis.get('is_integer_valued', False):
        return False
    # <= 5 unique values is clearly classification (even if numeric)
    if n_unique <= 5:
        return False
    # > 100 unique values is clearly regression
    if n_unique > 100:
        return False
    
    # The grey zone - integer-like target with 6-100 unique values
    return True

def sanity_check(detection, audit):
    problem = detection['problem_type']
    target = audit.get('target_column')
    n_unique = audit.get('target_unique_values', 0)
    target_dtype = audit.get('target_dtype', '')

    warnings = []
    # Regression chosen but target has only <= 2 unique values
    # (Sometimes users want probability prediction on binary, but it's rare)
    if problem == 'regression' and n_unique <= 2:
        warnings.append(
            f"Regression chosen but target has only {n_unique} unique values. "
            f"This is highly unusual - ensure this isn't meant to be classification."
        )

    # Classification chosen but target is float with hundreds of unique values
    if problem == 'classification' and target_dtype in ['float64', 'float32'] and n_unique > 50:
        warnings.append(
            f"Classification chosen but target is float with {n_unique} unique values. "
            f"This is highly unusual - ensure this isn't meant to be regression."
        )

    for w in warnings:
        narrate(f"  [!] Sanity Check Warning: {w}")

    return warnings


def run_detector(audit, df=None, force_type=None, _auto_input=None):
    """
    Analyzes the audit object and statistically tests the target distribution
    to determine the ML problem type (Classification, Regression, Clustering).
    """
    narrate("\n[DETECTION]")

    target = audit.get('target_column')
    target_dtype = audit.get('target_dtype')
    n_unique = audit.get('target_unique_values', 0)
    shape = audit.get('shape', (0, 0))
    n_rows = shape[0]

    detection = {
        'problem_type': 'unknown',
        'detection_method': 'inferred',
        'confidence': 'low',
        'classification_subtype': None,
        'num_classes': None,
        'class_labels': None,
        'metrics_averaging': None,
        'regression_subtype': 'standard',
        'target_log_transform': False,
        'signals': {}
    }

    # 1. Handle user override
    if force_type is not None:
        if force_type not in ['classification', 'regression', 'clustering']:
            raise ValueError(f"Invalid force_type: {force_type}")
        detection['problem_type'] = force_type
        detection['detection_method'] = 'user_flag'
        detection['confidence'] = 'high'
        narrate(f"  -> Problem type inferred as: {force_type.capitalize()} (User override)")
        sanity_check(detection, audit)
        return detection

    # 2. Check for Clustering (No target)
    if target is None:
        detection['problem_type'] = 'clustering'
        detection['confidence'] = 'high'
        detection['signals']['target_column'] = {'value': None, 'vote': 'clustering', 'weight': 5.0}
        narrate("  -> Problem type inferred as: Clustering (No target column found)")
        return detection

    # 3. Analyze Statistical Distribution
    if df is not None and target in df.columns:
        dist_analysis = analyze_target_distribution(df[target])
    else:
        # Fallback if raw dataframe not provided
        dist_analysis = {'is_normal': False, 'p_normal': 0.0, 'is_integer_valued': (target_dtype in ['int64', 'int32']), 'entropy': 0.0, 'gap_variance': 0.0, 'is_numeric': (target_dtype in ['int64', 'int32', 'float64', 'float32'])}

    cls_score = 0.0
    reg_score = 0.0

    # Signal A: Statistical Distribution
    # Entropy: Low entropy favors classification (concentrated classes). High entropy favors regression.
    if dist_analysis['entropy'] > 0:
        if dist_analysis['entropy'] < 1.0:
            detection['signals']['entropy'] = {'value': dist_analysis['entropy'], 'vote': 'classification', 'weight': 2.0}
            cls_score += 2.0
            narrate(f"  -> Signal 1 - Distribution  : Low entropy ({dist_analysis['entropy']:.2f}) -> Classification (+2.0)")
        elif dist_analysis['entropy'] > 2.5:
            detection['signals']['entropy'] = {'value': dist_analysis['entropy'], 'vote': 'regression', 'weight': 2.0}
            reg_score += 2.0
            narrate(f"  -> Signal 1 - Distribution  : High entropy ({dist_analysis['entropy']:.2f}) -> Regression (+2.0)")

    # Normality: Normally distributed target almost guarantees regression
    if dist_analysis['is_normal']:
        detection['signals']['normality'] = {'value': dist_analysis['p_normal'], 'vote': 'regression', 'weight': 2.5}
        reg_score += 2.5
        narrate(f"  -> Signal 2 - Normality     : Normally distributed (p={dist_analysis['p_normal']:.3f}) -> Regression (+2.5)")

    # Gap Variance: Regular spacing = classification ordinal. Irregular spacing = continuous regression.
    # We only apply this safely if there are some gaps to measure.
    if dist_analysis['is_numeric'] and n_unique > 2:
        if dist_analysis['gap_variance'] < 0.1:
            detection['signals']['gap_variance'] = {'value': dist_analysis['gap_variance'], 'vote': 'classification', 'weight': 1.0}
            cls_score += 1.0
            narrate(f"  -> Signal 3 - Gap variance  : Small/regular gaps ({dist_analysis['gap_variance']:.2f}) -> Classification (+1.0)")
        elif dist_analysis['gap_variance'] > 5.0:
            detection['signals']['gap_variance'] = {'value': dist_analysis['gap_variance'], 'vote': 'regression', 'weight': 1.0}
            reg_score += 1.0
            narrate(f"  -> Signal 3 - Gap variance  : Large/irregular gaps ({dist_analysis['gap_variance']:.2f}) -> Regression (+1.0)")

    # Signal B: Dtype + Unique Values combo
    if target_dtype in ['object', 'category', 'bool', 'string']:
        # Pure categorical
        detection['signals']['dtype'] = {'value': target_dtype, 'vote': 'classification', 'weight': 3.0}
        cls_score += 3.0
        narrate(f"  -> Signal 4 - Target Dtype  : {target_dtype} -> Classification (+3.0)")
    elif target_dtype in ['float64', 'float32'] and not dist_analysis['is_integer_valued']:
        # Pure continuous float
        detection['signals']['dtype'] = {'value': target_dtype, 'vote': 'regression', 'weight': 3.0}
        reg_score += 3.0
        narrate(f"  -> Signal 4 - Target Dtype  : {target_dtype} (Continuous) -> Regression (+3.0)")
    else:
        # Integer or Float-acting-as-integer
        if n_unique <= 10:
            detection['signals']['dtype'] = {'value': target_dtype, 'vote': 'classification', 'weight': 2.0}
            cls_score += 2.0
            narrate(f"  -> Signal 4 - Target Dtype  : {target_dtype} (Low cardinality: {n_unique}) -> Classification (+2.0)")
        elif n_unique > 100:
            detection['signals']['dtype'] = {'value': target_dtype, 'vote': 'regression', 'weight': 2.0}
            reg_score += 2.0
            narrate(f"  -> Signal 4 - Target Dtype  : {target_dtype} (High cardinality: {n_unique}) -> Regression (+2.0)")
        else:
            # Grey zone for dtype
            narrate(f"  -> Signal 4 - Target Dtype  : {target_dtype} ({n_unique} unique) -> Ambiguous (+0.0)")

    # Signal C: Column name heuristics
    target_lower = target.lower()
    class_keywords = ['is_', 'has_', 'was_', 'category', 'class', 'label', 'type', 'status', 'churn', 'fraud', 'survived', 'diagnosis', 'default', 'outcome', 'flag']
    reg_keywords = ['price', 'cost', 'revenue', 'amount', 'score', 'rate', 'value', 'age', 'count', 'quantity', 'duration', 'temperature', 'salary', 'weight', 'height']
    
    if any(p in target_lower for p in class_keywords):
        detection['signals']['name_heuristic'] = {'value': target, 'vote': 'classification', 'weight': 1.0}
        cls_score += 1.0
        narrate(f"  -> Signal 5 - Name heuristic: '{target}' matches discrete keywords -> Classification (+1.0)")
    elif any(p in target_lower for p in reg_keywords):
        detection['signals']['name_heuristic'] = {'value': target, 'vote': 'regression', 'weight': 1.0}
        reg_score += 1.0
        narrate(f"  -> Signal 5 - Name heuristic: '{target}' matches continuous keywords -> Regression (+1.0)")

    # Signal D: Feature types (Multivariate signal)
    vote, weight = feature_type_signal(audit)
    if weight > 0:
        if vote == 'classification':
            cls_score += weight
        else:
            reg_score += weight
        detection['signals']['feature_types'] = {'value': vote, 'vote': vote, 'weight': weight}
        narrate(f"  -> Signal 6 - Feature types : Majority {vote} -> {vote.capitalize()} (+{weight})")

    narrate(f"  -> Total Classification score : {cls_score}")
    narrate(f"  -> Total Regression score     : {reg_score}")

    # Decision Logic
    ambiguous = is_genuinely_ambiguous(n_unique, target_dtype, dist_analysis)
    
    if ambiguous or (cls_score == reg_score):
        narrate(f"\n  [!] Ambiguous Target Detected: '{target}'")
        narrate(f"  -> Evidence for Classification:")
        if n_unique <= 20:
             narrate(f"     [+] Low cardinality ({n_unique} unique values)")
        if dist_analysis['gap_variance'] < 2.0:
             narrate(f"     [+] Regular intervals between values (gap var: {dist_analysis['gap_variance']:.2f})")
        if dist_analysis['entropy'] < 1.5:
             narrate(f"     [+] Concentrated value distribution (entropy: {dist_analysis['entropy']:.2f})")
             
        narrate(f"  -> Evidence for Regression:")
        if reg_score > cls_score:
             narrate(f"     [+] Regression heuristic score is higher ({reg_score} vs {cls_score})")
        if dist_analysis['entropy'] > 2.0:
             narrate(f"     [+] Wide spread of target values (entropy: {dist_analysis['entropy']:.2f})")

        narrate(f"\n  -> Is '{target}' a categorical label or a numeric metric?")
        narrate(f"     (1) Classification - predict which category/label")
        narrate(f"     (2) Regression     - predict the exact numeric scale")

        if _auto_input:
            choice = _auto_input
            narrate(f"  -> Your choice (auto): {choice}")
        else:
            try:
                choice = input("  -> Your choice: ").strip()
            except EOFError:
                # Test runner bypass
                choice = '1' if cls_score >= reg_score else '2'
                narrate(f"  -> Your choice (auto-fallback): {choice}")

        if choice == '1':
            detection['problem_type'] = 'classification'
            detection['detection_method'] = 'user_confirmed'
            detection['confidence'] = 'high'
        elif choice == '2':
            detection['problem_type'] = 'regression'
            detection['detection_method'] = 'user_confirmed'
            detection['confidence'] = 'high'
        else:
            # Fallback if nonsense
            detection['problem_type'] = 'classification' if cls_score >= reg_score else 'regression'
            detection['detection_method'] = 'inferred_fallback'
            detection['confidence'] = 'low'
    else:
        # Not genuinely ambiguous - safe to pick max
        detection['problem_type'] = 'classification' if cls_score > reg_score else 'regression'
        
        # Calculate confidence
        total_score = cls_score + reg_score
        if total_score > 0:
            confidence_ratio = max(cls_score, reg_score) / total_score
            detection['confidence'] = 'high' if confidence_ratio >= 0.75 else ('medium' if confidence_ratio >= 0.6 else 'low')
        else:
            detection['confidence'] = 'low'

    # Compute Subtypes
    if detection['problem_type'] == 'classification':
        dist = audit.get('target_distribution', {})
        num_classes = len(dist) if dist else (n_unique if n_unique > 0 else 2) 
        
        detection['num_classes'] = num_classes
        detection['class_labels'] = [str(k) for k in dist.keys()] if dist else [str(i) for i in range(num_classes)]
        
        if num_classes == 2:
             detection['classification_subtype'] = 'binary'
             detection['metrics_averaging'] = 'binary'
        elif 3 <= num_classes <= 15:
             detection['classification_subtype'] = 'multiclass'
             detection['metrics_averaging'] = 'weighted'
        else:
             detection['classification_subtype'] = 'many_class'
             detection['metrics_averaging'] = 'weighted'
             narrate(f"  -> Warning: many_class subtype detected ({num_classes} classes). Will use LightGBM for speed.")
    else:
        # Regression boundary check
        target_stats = audit.get('stats', {}).get(target, {})
        t_min = target_stats.get('min', None)
        t_max = target_stats.get('max', None)
        
        if t_min is not None and t_max is not None:
            if 0.0 <= t_min and t_max <= 1.0:
                detection['regression_subtype'] = 'probability'
                narrate("  -> Target appears to be a probability/rate (0-1 domain)")
            elif t_min >= 0.0:
                detection['regression_subtype'] = 'bounded_positive'
            else:
                detection['regression_subtype'] = 'unbounded'
        else:
            detection['regression_subtype'] = 'standard'
            
        target_skews = audit.get('skewed_columns', {}).get(target, {})
        if target_skews.get('action') == 'log_transform':
             detection['target_log_transform'] = True
             narrate("  -> Detection: Target was previously log-transformed. Evaluator will back-transform.")

    narrate(f"  -> Problem type finalized as: {detection['problem_type'].capitalize()} ({detection['confidence']} confidence)\n")

    # Final check
    sanity_check(detection, audit)

    return detection
