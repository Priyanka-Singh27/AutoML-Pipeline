import sys
from pathlib import Path

# Add project root to path so we can import 'mocks' and 'core' directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocks import (
    mock_binary_classification,
    mock_regression,
    mock_clustering,
    mock_severe_imbalance,
    mock_very_small,
    mock_large,
    mock_high_correlation,
    mock_zero_variance,
    mock_weak_features,
    mock_leakage,
    mock_multiclass,
    mock_all_features_dropped,
    mock_single_feature,
    mock_many_classes,
    mock_heteroscedastic,
    mock_skewed_target,
    mock_low_optuna_budget,
)

try:
    from core.feature_selector import run_feature_selection
    from core.tuner import run_optuna_study
    from core.evaluator import run_evaluation
    CORE_IMPORTED = True
except ImportError:
    CORE_IMPORTED = False

DEFAULT_TIME_BUDGET = 30  # seconds per mock during testing

mocks = [
    mock_binary_classification,
    mock_regression,
    mock_clustering,
    mock_severe_imbalance,
    mock_very_small,
    mock_large,
    mock_high_correlation,
    mock_zero_variance,
    mock_weak_features,
    mock_leakage,
    mock_multiclass,
    mock_all_features_dropped,
    mock_single_feature,
    mock_many_classes,
    mock_heteroscedastic,
    mock_skewed_target,
    mock_low_optuna_budget,
]

def main():
    if not CORE_IMPORTED:
        print("⚠️  Core modules (feature_selector, tuner, evaluator) not found yet.")
        print("   Test runner will execute when Phase 3-6 implementation is complete.")
        print(f"   Loaded {len(mocks)} mocks successfully:\n")
        for m in mocks:
            print(f"     • {m.__name__.replace('mocks.', '')}")
        print()
        return

    results = []

    for mock in mocks:
        name = mock.__name__.replace('mocks.', '')
        time_budget = getattr(mock, 'TIME_BUDGET_OVERRIDE', DEFAULT_TIME_BUDGET)

        try:
            # Clustering has no target column
            if mock.mock_detection['problem_type'] == 'clustering':
                X = mock.mock_df
                y = None
            else:
                X = mock.mock_df.drop('target', axis=1)
                y = mock.mock_df['target']

            # 1. Feature Selection
            X_sel, dropped, remaining = run_feature_selection(X, y, mock.mock_audit)

            # 2. Optuna Tuning
            study, model = run_optuna_study(
                X_sel, y, mock.mock_detection, mock.mock_audit, time_budget=time_budget
            )

            # 3. Evaluation
            evaluation = run_evaluation(
                model, X_sel, y, mock.mock_detection, mock.mock_audit, study
            )

            # Edge case: all features dropped should raise, not pass silently
            if name == 'mock_all_features_dropped':
                results.append((name, 'FAIL', "Should have raised ValueError — all features were expected to be dropped"))
            else:
                results.append((name, 'PASS', None))

        except ValueError as e:
            if name == 'mock_all_features_dropped' and 'feature' in str(e).lower():
                results.append((name, 'PASS', None))  # correctly caught edge case
            else:
                results.append((name, 'FAIL', str(e)))
        except Exception as e:
            results.append((name, 'FAIL', str(e)))

    # ── Print results ─────────────────────────────────────────────────────────
    print('\n' + '=' * 62)
    print('  MOCK TEST RESULTS')
    print('=' * 62)

    for name, status, error in results:
        icon = '[+]' if status == 'PASS' else '[-]'
        print(f'  {icon}  {name:<38} {status}')
        if error is not None:
            print(f'       Error: {error}')

    print('=' * 62)
    passed = sum(1 for _, s, _ in results if s == 'PASS')
    print(f'  {passed}/{len(results)} passed\n')

if __name__ == '__main__':
    main()
