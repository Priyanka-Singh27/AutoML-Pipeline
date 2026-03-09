import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocks import mock_high_correlation

try:
    from core.detector import run_detector
    from core.feature_selector import run_feature_selection
    from core.tuner import run_optuna_study
    from core.evaluator import run_evaluation
except ImportError as e:
    print(f"FAILED IMPORT: {e}")
    sys.exit(1)

def main():
    try:
        mock = mock_high_correlation
        print(f"Running debug for {mock.__name__}")
        
        detection = run_detector(mock.mock_audit, df=mock.mock_df, _auto_input='1')

        if detection['problem_type'] == 'clustering':
            X = mock.mock_df
            y = None
        else:
            X = mock.mock_df.drop('target', axis=1)
            y = mock.mock_df['target']

        fs_results = run_feature_selection(X, y, mock.mock_audit, detection)
        
        X_train = fs_results['X_train']
        y_train = fs_results['y_train']

        study, model = run_optuna_study(
            X_train, y_train, detection, mock.mock_audit, time_budget=3, _auto_input='1'
        )

        evaluation = run_evaluation(
            model, X_train, y_train, detection, mock.mock_audit, study, run_shap=False
        )
        print("PASSED")
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}")
        print(f"Message: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
