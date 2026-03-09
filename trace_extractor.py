import sys
import traceback
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mocks import mock_multiclass, mock_many_classes

try:
    from core.detector import run_detector
    from core.feature_selector import run_feature_selection
    from core.tuner import run_optuna_study
    from core.evaluator import run_evaluation
except ImportError as e:
    print(f"FAILED IMPORT: {e}")
    sys.exit(1)

def run_debug(mock):
    print(f"--- Running {mock.__name__} ---")
    try:
        detection = run_detector(mock.mock_audit, df=mock.mock_df, _auto_input='1')
        X = mock.mock_df.drop('target', axis=1)
        y = mock.mock_df['target']

        fs_results = run_feature_selection(X, y, mock.mock_audit, detection)
        
        X_train = fs_results['X_train']
        y_train = fs_results['y_train']

        study, model = run_optuna_study(
            X_train, y_train, detection, mock.mock_audit, time_budget=1, _auto_input='1'
        )

        evaluation = run_evaluation(
            model, X_train, y_train, detection, mock.mock_audit, study, run_shap=False
        )
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {type(e).__name__} - {e}")
        traceback.print_exc()
        
def main():
    with open('multi_trace.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        sys.stderr = f
        run_debug(mock_multiclass)
        run_debug(mock_many_classes)
        
if __name__ == '__main__':
    main()
