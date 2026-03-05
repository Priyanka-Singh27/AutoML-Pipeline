**AutoML Pipeline**

Task 4 & 5 --- Implementation Document

*Optuna Tuning \| Evaluation & Explainability*

+-----------------------------------------------------------------------+
| **Person 2 --- ML Engine Layer**                                      |
|                                                                       |
| Receives: Preprocessed DataFrame + Audit Object + Detection Object    |
|                                                                       |
| Produces: Trained Model + Evaluation Object                           |
+-----------------------------------------------------------------------+

Version 1.0 --- AutoML Pipeline Project

**1. Overview**

You are Person 2 on the AutoML Pipeline project. Your responsibility is
the ML Engine Layer --- specifically Task 4 (Optuna Tuning) and Task 5
(Evaluation & Explainability). You receive clean, preprocessed data and
produce a fully trained, evaluated, and deployable model along with a
comprehensive evaluation object consumed by Person 3 for reporting and
deployment.

**1.1 Your Position in the Pipeline**

+-----------------------------------------------------------------------+
| Person 1 Output Your Input                                            |
|                                                                       |
| ─────────────────────────────────────────────────────────             |
|                                                                       |
| preprocessed_df ──────────► feature_selector.py                       |
|                                                                       |
| audit_object ──────────► detector.py                                  |
|                                                                       |
| detection_object ──────────► tuner.py                                 |
|                                                                       |
| evaluator.py                                                          |
|                                                                       |
| │                                                                     |
|                                                                       |
| ▼                                                                     |
|                                                                       |
| evaluation_object ──► Person 3                                        |
|                                                                       |
| trained_pipeline ──► Person 3                                         |
+-----------------------------------------------------------------------+

**1.2 Files You Own**

  ---------------------- ------------------------------------------------
  **File**               **Responsibility**

  detector.py            Problem type detection --- classification /
                         regression / clustering

  feature_selector.py    3-level feature selection before Optuna runs

  tuner.py               Optuna study --- model type + hyperparams
                         searched simultaneously

  evaluator.py           Metrics, SHAP explainability, limitations
                         generation
  ---------------------- ------------------------------------------------

**2. Your Input**

You receive three things from Person 1. Never re-analyze the raw CSV ---
all data intelligence is already captured in the audit object.

**2.1 Preprocessed DataFrame**

**What it is:** A clean pandas DataFrame ready for modeling. Person 1
has already handled all of the following:

-   Missing value imputation --- median for numerical, mode for
    categorical

-   Encoding --- one-hot for nominal, ordinal encoding for ordinal,
    target encoding for high cardinality

-   Scaling --- StandardScaler applied to all numerical columns

-   SMOTE --- applied if imbalance was detected and \--no-smote was not
    passed

-   Datetime extraction --- year, month, day_of_week, days_since
    extracted from datetime columns

-   Outlier capping --- capped at 1st and 99th percentile for flagged
    columns

-   Log transform --- applied to highly skewed columns

-   Drop candidates removed --- ID columns, free text, constant columns
    already gone

**What you must NOT assume:** Do not assume any specific column names,
dtypes, or shapes. Always read from the audit object.

**2.2 Audit Object**

The complete data intelligence report produced by Person 1\'s auditor.
The fields most relevant to your tasks are:

+-----------------------------------------------------------------------+
| audit = {                                                             |
|                                                                       |
| \# Core info                                                          |
|                                                                       |
| \'shape\': (5000, 14),                                                |
|                                                                       |
| \'dataset_size_class\': \'medium\', \#                                |
| very_small/small/medium/large/very_large                              |
|                                                                       |
| \'rows_to_features_ratio\': 357,                                      |
|                                                                       |
| \'dimensionality_risk\': False,                                       |
|                                                                       |
| \# Target info                                                        |
|                                                                       |
| \'target_column\': \'churn\',                                         |
|                                                                       |
| \'target_dtype\': \'int64\',                                          |
|                                                                       |
| \'target_unique_values\': 2,                                          |
|                                                                       |
| \'target_distribution\': {0: 0.84, 1: 0.16},                          |
|                                                                       |
| \# Feature quality                                                    |
|                                                                       |
| \'high_correlations\': \[(\'total_spend\', \'monthly_spend\',         |
| 0.97)\],                                                              |
|                                                                       |
| \'feature_target_correlation\': {                                     |
|                                                                       |
| \'monthly_spend\': {\'score\': 0.61, \'signal\': \'strong\'},         |
|                                                                       |
| \'signup_month\': {\'score\': 0.02, \'signal\': \'weak\'},            |
|                                                                       |
| \'random_col\': {\'score\': 0.00, \'signal\': \'none\'},              |
|                                                                       |
| },                                                                    |
|                                                                       |
| \'leakage_candidates\': {\'churn_reason\': {\'score\': 0.98}},        |
|                                                                       |
| \'skewed_columns\': {\'salary\': {\'skew\': 2.3, \'action\':          |
| \'log_transform\'}},                                                  |
|                                                                       |
| \'outlier_columns\': {\'salary\': {\'iqr_pct\': 0.031}},              |
|                                                                       |
| \# Imbalance                                                          |
|                                                                       |
| \'imbalance_detected\': True,                                         |
|                                                                       |
| \'imbalance_ratio\': 0.16,                                            |
|                                                                       |
| \'imbalance_severity\': \'moderate\',                                 |
|                                                                       |
| \'smote_recommended\': True,                                          |
|                                                                       |
| \# Sampling                                                           |
|                                                                       |
| \'sampling_recommended\': False,                                      |
|                                                                       |
| \'sampling_fraction\': None,                                          |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**2.3 Detection Object**

Produced by your detector.py (or received from Person 1 depending on
team agreement). Contains the problem type decision and all supporting
signals.

+-----------------------------------------------------------------------+
| detection = {                                                         |
|                                                                       |
| \'problem_type\': \'classification\', \#                              |
| classification/regression/clustering                                  |
|                                                                       |
| \'detection_method\': \'inferred\', \#                                |
| inferred/user_flag/unsupervised_flag                                  |
|                                                                       |
| \'confidence\': \'high\',                                             |
|                                                                       |
| \'classification_subtype\': \'binary\', \#                            |
| binary/multiclass/many_class                                          |
|                                                                       |
| \'num_classes\': 2,                                                   |
|                                                                       |
| \'class_labels\': \[0, 1\],                                           |
|                                                                       |
| \'metrics_averaging\': \'binary\',                                    |
|                                                                       |
| \'regression_subtype\': None,                                         |
|                                                                       |
| \'target_log_transform\': False,                                      |
|                                                                       |
| \'signals\': {                                                        |
|                                                                       |
| \'unique_values\': {\'value\': 2, \'vote\': \'classification\',       |
| \'weight\': 2},                                                       |
|                                                                       |
| \'dtype\': {\'value\': \'int64\', \'vote\': \'classification\',       |
| \'weight\': 2},                                                       |
|                                                                       |
| \'distribution\': {\'value\': \'discrete\',\'vote\':                  |
| \'classification\',\'weight\': 1},                                    |
|                                                                       |
| \'column_name\': {\'value\': \'churn\', \'vote\': \'classification\', |
| \'weight\': 1},                                                       |
|                                                                       |
| }                                                                     |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**3. Task 4A --- Feature Selection (feature_selector.py)**

Before Optuna runs, the feature set must be cleaned. Feature selection
removes noise, reduces dimensionality, and improves Optuna\'s results.
All selection must be fitted on training data only --- never on the full
dataset --- to prevent data leakage.

**3.1 Three-Level Selection Strategy**

**Level 1 --- Variance Threshold (Filter Method)**

Fast, model-agnostic. Drops features that carry no information.

-   **Tool:** sklearn.feature_selection.VarianceThreshold

-   Drop features where a single value appears in 95%+ of rows

-   Drop features with zero variance --- constant columns

-   Read quasi_constant list from audit object --- these are already
    flagged, drop them directly

+-----------------------------------------------------------------------+
| from sklearn.feature_selection import VarianceThreshold               |
|                                                                       |
| selector = VarianceThreshold(threshold=0.01)                          |
|                                                                       |
| X_train_filtered = selector.fit_transform(X_train)                    |
|                                                                       |
| dropped = \[col for col, keep in zip(X_train.columns,                 |
| selector.get_support()) if not keep\]                                 |
|                                                                       |
| narrate(f\'Variance threshold: dropped {dropped}\')                   |
+-----------------------------------------------------------------------+

**Level 2 --- Correlation Filter**

Removes redundant features that say the same thing. Read directly from
audit object --- no recomputation needed.

-   For each highly correlated pair (\>0.90) from
    audit\[\'high_correlations\'\]

-   Keep the feature with the higher feature_target_correlation score

-   Drop the weaker one --- it adds no new information

-   Also drop features where feature_target_correlation score is 0.00
    --- zero signal

+-----------------------------------------------------------------------+
| for feat_a, feat_b, corr in audit\[\'high_correlations\'\]:           |
|                                                                       |
| score_a =                                                             |
| audit\[\'feature_target_correlation\'\]\[feat_a\]\[\'score\'\]        |
|                                                                       |
| score_b =                                                             |
| audit\[\'feature_target_correlation\'\]\[feat_b\]\[\'score\'\]        |
|                                                                       |
| drop = feat_a if score_a \< score_b else feat_b                       |
|                                                                       |
| narrate(f\'Correlation filter: dropping {drop} ({corr:.2f} corr,      |
| lower target signal)\')                                               |
|                                                                       |
| X_train = X_train.drop(columns=\[drop\])                              |
+-----------------------------------------------------------------------+

**Level 3 --- SHAP-Based Selection (Embedded Method)**

Model-aware selection using a quick LightGBM probe. Most powerful of the
three levels.

-   Train a lightweight LightGBM model on X_train (50 estimators, fast)

-   **Tool:** shap.TreeExplainer

-   Compute SHAP values for all features on X_train

-   Calculate mean absolute SHAP value per feature across all training
    samples

-   Drop features where mean absolute SHAP \< threshold (0.001)

-   These features contribute nothing to model decisions

+-----------------------------------------------------------------------+
| import shap, lightgbm as lgb                                          |
|                                                                       |
| probe = lgb.LGBMClassifier(n_estimators=50, verbosity=-1)             |
|                                                                       |
| probe.fit(X_train, y_train)                                           |
|                                                                       |
| explainer = shap.TreeExplainer(probe)                                 |
|                                                                       |
| shap_vals = explainer.shap_values(X_train)                            |
|                                                                       |
| mean_abs_shap = np.abs(shap_vals).mean(axis=0)                        |
|                                                                       |
| weak = \[col for col, s in zip(X_train.columns, mean_abs_shap) if s   |
| \< 0.001\]                                                            |
|                                                                       |
| narrate(f\'SHAP-based: dropping {weak} (near-zero importance)\')      |
|                                                                       |
| X_train = X_train.drop(columns=weak)                                  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Important --- No Leakage Rule**                                     |
|                                                                       |
| All three levels of feature selection must be fitted ONLY on X_train. |
| Apply the fitted selectors to X_test separately. Wrap inside          |
| scikit-learn Pipeline to guarantee this automatically. This is one of |
| the most common mistakes in ML pipelines.                             |
+-----------------------------------------------------------------------+

**3.2 Narration Output**

+-----------------------------------------------------------------------+
| \[FEATURE SELECTION\]                                                 |
|                                                                       |
| → Level 1 - Variance threshold : dropped \'active\' (zero variance)   |
|                                                                       |
| → Level 2 - Correlation filter : dropped \'total_spend\' (0.97 corr,  |
|                                                                       |
| lower target signal than \'monthly_spend\')                           |
|                                                                       |
| → Level 3 - SHAP-based : dropped \'signup_month\', \'random_col\'     |
|                                                                       |
| (near-zero SHAP importance)                                           |
|                                                                       |
| → Features remaining: 9 of 14                                         |
|                                                                       |
| → Proceeding to Optuna with 9 features                                |
+-----------------------------------------------------------------------+

**4. Task 4B --- Optuna Tuning (tuner.py)**

The core ML task. Optuna searches over both model type and
hyperparameters simultaneously in one unified study. This is smarter and
faster than running a model competition followed by separate tuning.

**4.1 Technology Stack**

  ---------------------- ------------------------------------------------
  **Technology**         **Purpose & Reason**

  Optuna 3.x             Hyperparameter optimization framework --- uses
                         TPE sampler (smarter than random/grid search)

  XGBoost                Best overall performer on tabular data ---
                         handles missing values, imbalance, fast

  LightGBM               Fastest on large datasets, excellent on high
                         cardinality categoricals

  scikit-learn           Logistic Regression, Random Forest, SVM, KNN ---
                         baseline and comparison models

  StratifiedKFold        Cross-validation for classification ---
                         preserves class distribution in each fold

  KFold                  Cross-validation for regression --- standard
                         k-fold

  Optuna Pruning         MedianPruner cuts bad trials early --- saves
                         compute budget for promising trials

  joblib                 Save full scikit-learn Pipeline including
                         preprocessing + model as single object
  ---------------------- ------------------------------------------------

**4.2 Model Pools Per Problem Type**

**Classification Model Pool**

  ------------------ ---------------------- -----------------------------
  **Model**          **Primary Strength**   **When Optuna Favors It**

  XGBoost            Best overall tabular   Medium-large datasets,
                     accuracy, handles      imbalance present
                     imbalance well         

  LightGBM           Fastest training,      Large datasets (50k+ rows)
                     excellent on large     
                     data                   

  Random Forest      Robust, low            Small-medium datasets, noisy
                     overfitting, no        data
                     scaling needed         

  Logistic           Fast baseline, highly  Linearly separable data,
  Regression         interpretable          small datasets

  SVM (SVC)          Strong on small        Small datasets with clear
                     high-dimensional data  margins
  ------------------ ---------------------- -----------------------------

**Regression Model Pool**

  ------------------ ---------------------- -----------------------------
  **Model**          **Primary Strength**   **When Optuna Favors It**

  XGBoost Regressor  Best overall           Medium-large datasets
                     regression accuracy    

  LightGBM Regressor Fastest regression on  Large datasets
                     large data             

  Random Forest      Robust, handles        Noisy data, complex
  Regressor          non-linearity          relationships

  Ridge Regression   Fast linear baseline   Linear relationships, many
                     with regularization    features

  SVR                Strong on small        Small, clean datasets
                     datasets               
  ------------------ ---------------------- -----------------------------

**Clustering --- No Optuna**

Clustering sits outside the Optuna flow. There is no target to optimize
against. Instead run three algorithms and evaluate with silhouette score
and Davies-Bouldin index.

  ------------------ ---------------------- -----------------------------
  **Algorithm**      **Best For**           **Key Parameter**

  K-Means            Compact, spherical     n_clusters --- auto via
                     clusters               elbow + silhouette

  DBSCAN             Arbitrary shape        eps, min_samples --- grid
                     clusters, noise        search over small range
                     detection              

  Agglomerative      Hierarchical structure n_clusters, linkage method
                     in data                
  ------------------ ---------------------- -----------------------------

**4.3 The Optuna Objective Function**

The objective function is the heart of Task 4. It samples a model type
and its hyperparameters together, trains using cross-validation, and
returns a score for Optuna to optimize.

+-----------------------------------------------------------------------+
| import optuna                                                         |
|                                                                       |
| from sklearn.model_selection import StratifiedKFold, cross_val_score  |
|                                                                       |
| from xgboost import XGBClassifier                                     |
|                                                                       |
| from lightgbm import LGBMClassifier                                   |
|                                                                       |
| from sklearn.ensemble import RandomForestClassifier                   |
|                                                                       |
| from sklearn.linear_model import LogisticRegression                   |
|                                                                       |
| def build_objective(X_train, y_train, problem_type, detection,        |
| audit):                                                               |
|                                                                       |
| def objective(trial):                                                 |
|                                                                       |
| \# ── Step 1: Sample model type ──────────────────────────            |
|                                                                       |
| model_name = trial.suggest_categorical(\'model\',                     |
|                                                                       |
| \[\'xgboost\', \'lightgbm\', \'random_forest\',                       |
| \'logistic_regression\'\]                                             |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# ── Step 2: Sample hyperparams for chosen model ─────────           |
|                                                                       |
| if model_name == \'xgboost\':                                         |
|                                                                       |
| model = XGBClassifier(                                                |
|                                                                       |
| learning_rate = trial.suggest_float(\'xgb_lr\', 0.01, 0.3, log=True), |
|                                                                       |
| max_depth = trial.suggest_int(\'xgb_depth\', 3, 10),                  |
|                                                                       |
| n_estimators = trial.suggest_int(\'xgb_n\', 100, 500),                |
|                                                                       |
| subsample = trial.suggest_float(\'xgb_sub\', 0.6, 1.0),               |
|                                                                       |
| colsample_bytree= trial.suggest_float(\'xgb_col\', 0.6, 1.0),         |
|                                                                       |
| eval_metric = \'logloss\',                                            |
|                                                                       |
| verbosity = 0                                                         |
|                                                                       |
| )                                                                     |
|                                                                       |
| elif model_name == \'lightgbm\':                                      |
|                                                                       |
| model = LGBMClassifier(                                               |
|                                                                       |
| learning_rate = trial.suggest_float(\'lgbm_lr\', 0.01, 0.3,           |
| log=True),                                                            |
|                                                                       |
| num_leaves = trial.suggest_int(\'lgbm_leaves\', 20, 150),             |
|                                                                       |
| n_estimators = trial.suggest_int(\'lgbm_n\', 100, 500),               |
|                                                                       |
| min_child_samples= trial.suggest_int(\'lgbm_min\', 5, 50),            |
|                                                                       |
| verbosity = -1                                                        |
|                                                                       |
| )                                                                     |
|                                                                       |
| elif model_name == \'random_forest\':                                 |
|                                                                       |
| model = RandomForestClassifier(                                       |
|                                                                       |
| n_estimators = trial.suggest_int(\'rf_n\', 100, 500),                 |
|                                                                       |
| max_depth = trial.suggest_int(\'rf_depth\', 3, 20),                   |
|                                                                       |
| min_samples_split= trial.suggest_int(\'rf_split\', 2, 10),            |
|                                                                       |
| max_features = trial.suggest_categorical(\'rf_feat\', \[\'sqrt\',     |
| \'log2\'\])                                                           |
|                                                                       |
| )                                                                     |
|                                                                       |
| elif model_name == \'logistic_regression\':                           |
|                                                                       |
| model = LogisticRegression(                                           |
|                                                                       |
| C = trial.suggest_float(\'lr_C\', 0.001, 10.0, log=True),             |
|                                                                       |
| solver = \'lbfgs\',                                                   |
|                                                                       |
| max_iter = 1000                                                       |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# ── Step 3: Cross-validate ──────────────────────────────           |
|                                                                       |
| cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)       |
|                                                                       |
| scores = cross_val_score(model, X_train, y_train,                     |
|                                                                       |
| cv=cv, scoring=\'f1_weighted\', n_jobs=-1)                            |
|                                                                       |
| return scores.mean()                                                  |
|                                                                       |
| return objective                                                      |
+-----------------------------------------------------------------------+

**4.4 Running the Optuna Study**

+-----------------------------------------------------------------------+
| def run_optuna_study(X_train, y_train, detection, audit,              |
| time_budget=120):                                                     |
|                                                                       |
| \# Pruner cuts bad trials early --- saves budget for good ones        |
|                                                                       |
| pruner = optuna.pruners.MedianPruner(n_startup_trials=5,              |
| n_warmup_steps=3)                                                     |
|                                                                       |
| sampler = optuna.samplers.TPESampler(seed=42)                         |
|                                                                       |
| study = optuna.create_study(                                          |
|                                                                       |
| direction=\'maximize\',                                               |
|                                                                       |
| sampler=sampler,                                                      |
|                                                                       |
| pruner=pruner                                                         |
|                                                                       |
| )                                                                     |
|                                                                       |
| objective = build_objective(X_train, y_train,                         |
| detection\[\'problem_type\'\],                                        |
|                                                                       |
| detection, audit)                                                     |
|                                                                       |
| \# Time-bounded optimization                                          |
|                                                                       |
| study.optimize(                                                       |
|                                                                       |
| objective,                                                            |
|                                                                       |
| timeout=time_budget,                                                  |
|                                                                       |
| callbacks=\[narration_callback\], \# prints progress each trial       |
|                                                                       |
| show_progress_bar=False                                               |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Retrain best model on full training set                            |
|                                                                       |
| best_model = rebuild_model(study.best_params)                         |
|                                                                       |
| best_model.fit(X_train, y_train)                                      |
|                                                                       |
| return study, best_model                                              |
+-----------------------------------------------------------------------+

**4.5 Narration Callback**

+-----------------------------------------------------------------------+
| def narration_callback(study, trial):                                 |
|                                                                       |
| is_best = trial.value == study.best_value                             |
|                                                                       |
| marker = \' \<- new best\' if is_best else \'\'                       |
|                                                                       |
| model = trial.params.get(\'model\', \'unknown\')                      |
|                                                                       |
| score = trial.value                                                   |
|                                                                       |
| duration= trial.duration.total_seconds()                              |
|                                                                       |
| print(f\' Trial {trial.number:\>3} \| {model:\<20} \| F1={score:.4f}  |
| \|                                                                    |
|                                                                       |
| \[{duration:.1f}s\]{marker}\')                                        |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| \[OPTUNA TUNING\]                                                     |
|                                                                       |
| → Problem type : Classification (binary)                              |
|                                                                       |
| → Model pool : XGBoost, LightGBM, Random Forest, Logistic Regression  |
|                                                                       |
| → Time budget : 120 seconds                                           |
|                                                                       |
| → CV strategy : StratifiedKFold (5 folds)                             |
|                                                                       |
| → Primary metric : F1 Weighted                                        |
|                                                                       |
| → Pruner : MedianPruner (cuts bad trials early)                       |
|                                                                       |
| Trial 1 \| xgboost \| F1=0.7401 \| \[12.3s\]                          |
|                                                                       |
| Trial 2 \| logistic_regression \| F1=0.7089 \| \[2.1s\]               |
|                                                                       |
| Trial 3 \| xgboost \| F1=0.7812 \| \[14.2s\] \<- new best             |
|                                                                       |
| Trial 4 \| random_forest \| F1=0.7634 \| \[8.9s\]                     |
|                                                                       |
| Trial 5 \| lightgbm \| F1=0.7923 \| \[9.8s\] \<- new best             |
|                                                                       |
| \...                                                                  |
|                                                                       |
| Trial 23 \| xgboost \| F1=0.8134 \| \[13.1s\] \<- new best            |
|                                                                       |
| Study complete: 23 trials in 120 seconds                              |
|                                                                       |
| Best model : XGBoost (F1=0.8134)                                      |
|                                                                       |
| Best params : learning_rate=0.047, max_depth=6, n_estimators=312      |
|                                                                       |
| Retraining best model on full training set\...                        |
+-----------------------------------------------------------------------+

**4.6 Dataset-Aware Search Space Constraints**

The reasoning layer from the architecture discussion is implemented
here. Rather than hard-filtering models, constrain their hyperparameter
search spaces based on dataset characteristics from the audit object.

  ---------------------- ------------------------------------------------
  **Condition from       **Search Space Adjustment**
  Audit**                

  dataset_size_class ==  XGBoost/LightGBM max_depth capped at 4,
  \'very_small\'         n_estimators capped at 200

  dataset_size_class ==  SVM removed from pool (too slow), LightGBM given
  \'large\'              wider search space

  imbalance_detected ==  XGBoost/LightGBM scale_pos_weight added as
  True                   parameter

  dimensionality_risk == LogisticRegression C range widened for stronger
  True                   regularization

  sampling_recommended   Use audit\[\'sampling_fraction\'\] to sample
  == True                X_train before each trial
  ---------------------- ------------------------------------------------

**5. Task 5 --- Evaluation (evaluator.py)**

After the best model is trained, run full evaluation. The evaluator
produces all metrics, explainability plots, and auto-generated
limitations. Everything is stored in the evaluation object for Person 3.

**5.1 Technology Stack**

  --------------------------- ------------------------------------------------
  **Technology**              **Purpose**

  SHAP (TreeExplainer)        Feature importance --- global summary + local
                              waterfall plots

  matplotlib + seaborn        All plots --- confusion matrix, residuals,
                              cluster visualization

  scikit-learn metrics        Confusion matrix, classification report,
                              ROC-AUC, RMSE, MAE, R2

  sklearn.decomposition.PCA   2D cluster visualization for clustering problems

  sklearn.manifold.TSNE       Alternative 2D visualization for complex cluster
                              shapes

  scipy.stats                 Statistical tests for regression residual
                              analysis
  --------------------------- ------------------------------------------------

**5.2 Classification Evaluation**

**Confusion Matrix + Auto-Inference**

Compute the confusion matrix and automatically generate a plain English
interpretation. This is one of the most impressive parts of the
pipeline.

+-----------------------------------------------------------------------+
| from sklearn.metrics import confusion_matrix                          |
|                                                                       |
| import numpy as np                                                    |
|                                                                       |
| def evaluate_classification(model, X_test, y_test, detection, audit): |
|                                                                       |
| y_pred = model.predict(X_test)                                        |
|                                                                       |
| y_prob = model.predict_proba(X_test)\[:, 1\] \# for binary            |
|                                                                       |
| cm = confusion_matrix(y_test, y_pred)                                 |
|                                                                       |
| cm_inference = interpret_confusion_matrix(cm,                         |
| detection\[\'class_labels\'\], audit)                                 |
|                                                                       |
| return cm, cm_inference                                               |
|                                                                       |
| def interpret_confusion_matrix(cm, class_labels, audit):              |
|                                                                       |
| \# Find which class has highest false negative rate                   |
|                                                                       |
| fn_rates = \[\]                                                       |
|                                                                       |
| for i in range(len(class_labels)):                                    |
|                                                                       |
| total_actual = cm\[i\].sum()                                          |
|                                                                       |
| false_negatives = total_actual - cm\[i\]\[i\]                         |
|                                                                       |
| fn_rates.append(false_negatives / total_actual if total_actual \> 0   |
| else 0)                                                               |
|                                                                       |
| worst_class = class_labels\[np.argmax(fn_rates)\]                     |
|                                                                       |
| worst_rate = max(fn_rates) \* 100                                     |
|                                                                       |
| reason = \'\'                                                         |
|                                                                       |
| if audit\[\'imbalance_detected\'\]:                                   |
|                                                                       |
| reason = \' This is likely due to class imbalance in the training     |
| data.\'                                                               |
|                                                                       |
| return (f\'The model struggles most with class {worst_class},         |
| misclassifying\'                                                      |
|                                                                       |
| f\' {worst_rate:.1f}% of actual instances.{reason}\')                 |
+-----------------------------------------------------------------------+

**Classification Report**

+-----------------------------------------------------------------------+
| from sklearn.metrics import classification_report, roc_auc_score      |
|                                                                       |
| report = classification_report(y_test, y_pred, output_dict=True)      |
|                                                                       |
| roc_auc = roc_auc_score(y_test, y_prob)                               |
+-----------------------------------------------------------------------+

**SHAP Explainability --- Classification**

+-----------------------------------------------------------------------+
| import shap                                                           |
|                                                                       |
| explainer = shap.TreeExplainer(model)                                 |
|                                                                       |
| shap_vals = explainer.shap_values(X_test)                             |
|                                                                       |
| \# Global --- which features matter most overall                      |
|                                                                       |
| fig_summary = plt.figure()                                            |
|                                                                       |
| shap.summary_plot(shap_vals, X_test, show=False)                      |
|                                                                       |
| plt.tight_layout()                                                    |
|                                                                       |
| \# Local --- why did the model predict THIS for row 0                 |
|                                                                       |
| fig_waterfall = plt.figure()                                          |
|                                                                       |
| shap.waterfall_plot(shap.Explanation(                                 |
|                                                                       |
| values=shap_vals\[0\],                                                |
|                                                                       |
| base_values=explainer.expected_value,                                 |
|                                                                       |
| data=X_test.iloc\[0\]                                                 |
|                                                                       |
| ), show=False)                                                        |
|                                                                       |
| plt.tight_layout()                                                    |
+-----------------------------------------------------------------------+

**5.3 Regression Evaluation**

+-----------------------------------------------------------------------+
| from sklearn.metrics import mean_squared_error, mean_absolute_error,  |
| r2_score                                                              |
|                                                                       |
| import scipy.stats as stats                                           |
|                                                                       |
| def evaluate_regression(model, X_test, y_test):                       |
|                                                                       |
| y_pred = model.predict(X_test)                                        |
|                                                                       |
| residuals = y_test - y_pred                                           |
|                                                                       |
| rmse = np.sqrt(mean_squared_error(y_test, y_pred))                    |
|                                                                       |
| mae = mean_absolute_error(y_test, y_pred)                             |
|                                                                       |
| r2 = r2_score(y_test, y_pred)                                         |
|                                                                       |
| \# Residual plot --- predicted vs residuals                           |
|                                                                       |
| fig_residual, ax = plt.subplots(figsize=(8, 5))                       |
|                                                                       |
| ax.scatter(y_pred, residuals, alpha=0.4)                              |
|                                                                       |
| ax.axhline(0, color=\'red\', linestyle=\'\--\')                       |
|                                                                       |
| ax.set_xlabel(\'Predicted Values\')                                   |
|                                                                       |
| ax.set_ylabel(\'Residuals\')                                          |
|                                                                       |
| ax.set_title(\'Residual Plot\')                                       |
|                                                                       |
| \# Check heteroscedasticity --- if residuals fan out, flag it         |
|                                                                       |
| \_, p_value = stats.spearmanr(y_pred, np.abs(residuals))              |
|                                                                       |
| heteroscedastic = p_value \< 0.05                                     |
|                                                                       |
| return rmse, mae, r2, fig_residual, heteroscedastic                   |
+-----------------------------------------------------------------------+

**5.4 Clustering Evaluation**

+-----------------------------------------------------------------------+
| from sklearn.metrics import silhouette_score, davies_bouldin_score    |
|                                                                       |
| from sklearn.decomposition import PCA                                 |
|                                                                       |
| import pandas as pd                                                   |
|                                                                       |
| def evaluate_clustering(model, X, labels):                            |
|                                                                       |
| silhouette = silhouette_score(X, labels)                              |
|                                                                       |
| davies_bouldin = davies_bouldin_score(X, labels)                      |
|                                                                       |
| \# PCA 2D visualization                                               |
|                                                                       |
| pca = PCA(n_components=2)                                             |
|                                                                       |
| X_2d = pca.fit_transform(X)                                           |
|                                                                       |
| fig_clusters, ax = plt.subplots(figsize=(8, 6))                       |
|                                                                       |
| scatter = ax.scatter(X_2d\[:, 0\], X_2d\[:, 1\], c=labels,            |
| cmap=\'tab10\', alpha=0.6)                                            |
|                                                                       |
| ax.set_title(\'Cluster Visualization (PCA 2D)\')                      |
|                                                                       |
| plt.colorbar(scatter)                                                 |
|                                                                       |
| \# Cluster profiles --- mean feature values per cluster               |
|                                                                       |
| df = pd.DataFrame(X, columns=feature_names)                           |
|                                                                       |
| df\[\'cluster\'\] = labels                                            |
|                                                                       |
| cluster_profiles = df.groupby(\'cluster\').mean().round(3)            |
|                                                                       |
| return silhouette, davies_bouldin, fig_clusters, cluster_profiles     |
+-----------------------------------------------------------------------+

**5.5 Auto-Generated Limitations**

Limitations are generated dynamically from the audit object --- not
generic boilerplate. Each limitation is specific to what was actually
found in the data.

+-----------------------------------------------------------------------+
| def generate_limitations(audit, detection, study):                    |
|                                                                       |
| limitations = \[\]                                                    |
|                                                                       |
| \# Dataset size                                                       |
|                                                                       |
| n_rows = audit\[\'shape\'\]\[0\]                                      |
|                                                                       |
| if n_rows \< 500:                                                     |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| f\'Very small dataset ({n_rows} rows). Model may overfit and is       |
| unlikely\'                                                            |
|                                                                       |
| f\' to generalize to unseen distributions.\'                          |
|                                                                       |
| )                                                                     |
|                                                                       |
| elif n_rows \< 2000:                                                  |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| f\'Small dataset ({n_rows} rows). Results may vary significantly\'    |
|                                                                       |
| f\' on different train/test splits.\'                                 |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Imbalance                                                          |
|                                                                       |
| if audit\[\'imbalance_detected\'\]:                                   |
|                                                                       |
| ratio = audit\[\'imbalance_ratio\'\] \* 100                           |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| f\'Class imbalance present (minority class: {ratio:.0f}%). SMOTE      |
| was\'                                                                 |
|                                                                       |
| f\' applied --- synthetic samples may not represent real minority     |
| instances.\'                                                          |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Dropped columns                                                    |
|                                                                       |
| n_dropped = len(audit.get(\'drop_candidates\', {}))                   |
|                                                                       |
| if n_dropped \> 0:                                                    |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| f\'{n_dropped} columns were dropped during preprocessing. Model       |
| quality\'                                                             |
|                                                                       |
| f\' depends entirely on the remaining {audit\[\"shape\"\]\[1\] -      |
| n_dropped} features.\'                                                |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Leakage candidates that were kept                                  |
|                                                                       |
| kept_leaky = \[k for k in audit.get(\'leakage_candidates\', {})       |
|                                                                       |
| if k not in audit.get(\'drop_candidates\', {})\]                      |
|                                                                       |
| if kept_leaky:                                                        |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| f\'Potential leakage features kept by user override: {kept_leaky}.\'  |
|                                                                       |
| f\' Real-world performance may be significantly lower than            |
| reported.\'                                                           |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# High correlations remaining                                        |
|                                                                       |
| if audit.get(\'high_correlations\'):                                  |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| \'Some highly correlated features remain. Coefficient                 |
| interpretations\'                                                     |
|                                                                       |
| \' from linear models may be unreliable.\'                            |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Regression heteroscedasticity                                      |
|                                                                       |
| if detection\[\'problem_type\'\] == \'regression\':                   |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| \'Residual analysis should be reviewed. Heteroscedasticity or\'       |
|                                                                       |
| \' non-random patterns in residuals indicate model                    |
| misspecification.\'                                                   |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# Optuna budget                                                      |
|                                                                       |
| n_trials = len(study.trials)                                          |
|                                                                       |
| if n_trials \< 20:                                                    |
|                                                                       |
| limitations.append(                                                   |
|                                                                       |
| f\'Only {n_trials} Optuna trials completed. Increasing                |
| \--time-budget\'                                                      |
|                                                                       |
| f\' may yield better hyperparameters.\'                               |
|                                                                       |
| )                                                                     |
|                                                                       |
| return limitations                                                    |
+-----------------------------------------------------------------------+

**6. Your Output --- Evaluation Object**

Everything you produce gets packaged into this single evaluation object.
Person 3 reads this for terminal narration, PDF report generation, and
deployment. Never pass raw model objects directly --- always go through
this contract.

+-----------------------------------------------------------------------+
| evaluation = {                                                        |
|                                                                       |
| \# ── OPTUNA STUDY ────────────────────────────────────────────       |
|                                                                       |
| \'best_model\': trained_pipeline_object, \# full Pipeline not just    |
| model                                                                 |
|                                                                       |
| \'best_model_name\': \'XGBoost\',                                     |
|                                                                       |
| \'best_params\': {\'learning_rate\': 0.047, \'max_depth\': 6},        |
|                                                                       |
| \'training_time\': 120.3, \# seconds                                  |
|                                                                       |
| \'n_trials\': 23,                                                     |
|                                                                       |
| \'all_trials\': \[                                                    |
|                                                                       |
| {\'trial\': 1, \'model\': \'xgboost\', \'score\': 0.74, \'duration\': |
| 12.3},                                                                |
|                                                                       |
| {\'trial\': 2, \'model\': \'logistic_regression\', \'score\': 0.71,   |
| \'duration\': 2.1},                                                   |
|                                                                       |
| \# \... all trials                                                    |
|                                                                       |
| \],                                                                   |
|                                                                       |
| \# ── FEATURE SELECTION ───────────────────────────────────────       |
|                                                                       |
| \'features_original\': 14,                                            |
|                                                                       |
| \'features_remaining\': 9,                                            |
|                                                                       |
| \'features_dropped\': {                                               |
|                                                                       |
| \'active\': \'zero variance\',                                        |
|                                                                       |
| \'total_spend\': \'high correlation with monthly_spend\',             |
|                                                                       |
| \'signup_month\': \'near-zero SHAP importance\',                      |
|                                                                       |
| },                                                                    |
|                                                                       |
| \# ── CLASSIFICATION METRICS ──────────────────────────────────       |
|                                                                       |
| \'confusion_matrix\': np.ndarray, \# shape (n_classes, n_classes)     |
|                                                                       |
| \'confusion_matrix_inference\': \'The model struggles most with class |
| 1\...\',                                                              |
|                                                                       |
| \'classification_report\': dict, \# sklearn output_dict=True          |
|                                                                       |
| \'roc_auc\': 0.87,                                                    |
|                                                                       |
| \'f1_weighted\': 0.81,                                                |
|                                                                       |
| \'precision_weighted\': 0.82,                                         |
|                                                                       |
| \'recall_weighted\': 0.79,                                            |
|                                                                       |
| \# ── REGRESSION METRICS ──────────────────────────────────────       |
|                                                                       |
| \'rmse\': None,                                                       |
|                                                                       |
| \'mae\': None,                                                        |
|                                                                       |
| \'r2\': None,                                                         |
|                                                                       |
| \'heteroscedastic\': None,                                            |
|                                                                       |
| \'residual_plot\': None, \# matplotlib figure                         |
|                                                                       |
| \# ── CLUSTERING METRICS ──────────────────────────────────────       |
|                                                                       |
| \'silhouette_score\': None,                                           |
|                                                                       |
| \'davies_bouldin\': None,                                             |
|                                                                       |
| \'cluster_profiles\': None, \# pandas DataFrame                       |
|                                                                       |
| \'cluster_visualization\': None, \# matplotlib figure                 |
|                                                                       |
| \'n_clusters_final\': None,                                           |
|                                                                       |
| \# ── EXPLAINABILITY ──────────────────────────────────────────       |
|                                                                       |
| \'shap_values\': np.ndarray,                                          |
|                                                                       |
| \'feature_importance\': {                                             |
|                                                                       |
| \'monthly_spend\': 0.61,                                              |
|                                                                       |
| \'age\': 0.29,                                                        |
|                                                                       |
| \'country\': 0.08,                                                    |
|                                                                       |
| \# \... all remaining features                                        |
|                                                                       |
| },                                                                    |
|                                                                       |
| \'shap_summary_plot\': matplotlib_figure,                             |
|                                                                       |
| \'shap_waterfall_plot\': matplotlib_figure,                           |
|                                                                       |
| \# ── LIMITATIONS ─────────────────────────────────────────────       |
|                                                                       |
| \'limitations\': \[                                                   |
|                                                                       |
| \'Class imbalance present (16%). SMOTE applied\...\',                 |
|                                                                       |
| \'Only 23 Optuna trials completed. Increasing \--time-budget\...\',   |
|                                                                       |
| \# \... all auto-generated limitations                                |
|                                                                       |
| \],                                                                   |
|                                                                       |
| \# ── MODEL PATH ──────────────────────────────────────────────       |
|                                                                       |
| \'model_path\': \'./outputs/model.joblib\',                           |
|                                                                       |
| \'model_saved\': True,                                                |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**7. Critical Rules --- Do Not Forget**

+-----------------------------------------------------------------------+
| **Rule 1 --- Save the Full Pipeline, Not Just the Model**             |
|                                                                       |
| Always wrap preprocessing + feature selection + model inside a        |
| scikit-learn Pipeline and save the entire Pipeline as one joblib      |
| file. If you save only the model, inference will fail on raw input    |
| data because the preprocessing steps are missing.                     |
|                                                                       |
| +------------------------------------------------------------------+  |
| | from sklearn.pipeline import Pipeline                            |  |
| |                                                                  |  |
| | import joblib                                                    |  |
| |                                                                  |  |
| | full_pipeline = Pipeline(\[                                      |  |
| |                                                                  |  |
| | (\'scaler\', StandardScaler()),                                  |  |
| |                                                                  |  |
| | (\'selector\', feature_selector),                                |  |
| |                                                                  |  |
| | (\'model\', best_model),                                         |  |
| |                                                                  |  |
| | \])                                                              |  |
| |                                                                  |  |
| | full_pipeline.fit(X_train, y_train)                              |  |
| |                                                                  |  |
| | joblib.dump(full_pipeline, \'./outputs/model.joblib\')           |  |
| +------------------------------------------------------------------+  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Rule 2 --- No Data Leakage in Feature Selection**                   |
|                                                                       |
| All three levels of feature selection (VarianceThreshold, Correlation |
| Filter, SHAP-based) must be fitted ONLY on X_train. Apply the fitted  |
| selectors to X_test separately. Never fit on the full dataset before  |
| splitting.                                                            |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Rule 3 --- Use StratifiedKFold for Classification**                 |
|                                                                       |
| Always use StratifiedKFold for classification cross-validation inside |
| Optuna. Plain KFold can create folds with no minority class samples,  |
| especially on imbalanced datasets, causing misleading scores.         |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Rule 4 --- Optuna Verbosity**                                       |
|                                                                       |
| Set optuna.logging.set_verbosity(optuna.logging.WARNING) before       |
| creating the study. Otherwise Optuna prints its own logs that clash   |
| with your narration system.                                           |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Rule 5 --- SHAP for Non-Tree Models**                               |
|                                                                       |
| TreeExplainer only works for tree-based models (XGBoost, LightGBM,    |
| Random Forest). For Logistic Regression and SVM, use                  |
| shap.LinearExplainer or shap.KernelExplainer instead. KernelExplainer |
| is slow --- use a small background sample (100 rows).                 |
|                                                                       |
| +------------------------------------------------------------------+  |
| | if model_name in \[\'xgboost\', \'lightgbm\',                    |  |
| | \'random_forest\'\]:                                             |  |
| |                                                                  |  |
| | explainer = shap.TreeExplainer(model)                            |  |
| |                                                                  |  |
| | elif model_name == \'logistic_regression\':                      |  |
| |                                                                  |  |
| | explainer = shap.LinearExplainer(model, X_train)                 |  |
| |                                                                  |  |
| | else: \# SVM                                                     |  |
| |                                                                  |  |
| | background = shap.sample(X_train, 100)                           |  |
| |                                                                  |  |
| | explainer = shap.KernelExplainer(model.predict_proba,            |  |
| | background)                                                      |  |
| +------------------------------------------------------------------+  |
+-----------------------------------------------------------------------+

**8. Dependencies**

+-----------------------------------------------------------------------+
| \# requirements.txt --- your section                                  |
|                                                                       |
| optuna==3.6.1                                                         |
|                                                                       |
| xgboost==2.0.3                                                        |
|                                                                       |
| lightgbm==4.3.0                                                       |
|                                                                       |
| scikit-learn==1.4.2                                                   |
|                                                                       |
| shap==0.45.0                                                          |
|                                                                       |
| imbalanced-learn==0.12.2                                              |
|                                                                       |
| pandas==2.2.1                                                         |
|                                                                       |
| numpy==1.26.4                                                         |
|                                                                       |
| matplotlib==3.8.3                                                     |
|                                                                       |
| seaborn==0.13.2                                                       |
|                                                                       |
| scipy==1.13.0                                                         |
|                                                                       |
| joblib==1.4.0                                                         |
+-----------------------------------------------------------------------+

**9. Independent Development --- Mock Strategy**

Person 1 is working on the auditor and preprocessor in parallel. You do
not need to wait for them. Using mock objects and mock DataFrames, you
can build, run, and test your entire pipeline independently from Day 1.
When integration day comes, swapping mocks for real objects is a single
line change per file.

**9.1 The Golden Rule --- One Line Swap**

Structure every file so the data source is a single import at the top.
During development, import from mocks. On integration day, swap that one
import. Nothing else in your code changes.

+-----------------------------------------------------------------------+
| \# ── During development ──────────────────────────────────────────   |
|                                                                       |
| from mocks.mock_binary_classification import mock_df as df            |
|                                                                       |
| from mocks.mock_binary_classification import mock_audit as audit      |
|                                                                       |
| from mocks.mock_binary_classification import mock_detection as        |
| detection                                                             |
|                                                                       |
| \# ── After integration --- swap ONLY these three lines ───────────── |
|                                                                       |
| from core.auditor import run_audit                                    |
|                                                                       |
| from core.detector import run_detector                                |
|                                                                       |
| audit = run_audit(df)                                                 |
|                                                                       |
| detection = run_detector(audit)                                       |
+-----------------------------------------------------------------------+

**9.2 Folder Structure for Mocks**

+-----------------------------------------------------------------------+
| automl_pipeline/                                                      |
|                                                                       |
| │                                                                     |
|                                                                       |
| ├── mocks/                                                            |
|                                                                       |
| │ ├── mock_generator.py \# base generator --- all mocks built from    |
| this                                                                  |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ ├── \# Problem Type Mocks                                           |
|                                                                       |
| │ ├── mock_binary_classification.py \# main dev mock                  |
|                                                                       |
| │ ├── mock_multiclass.py \# 3-5 class target                          |
|                                                                       |
| │ ├── mock_regression.py \# continuous float target                   |
|                                                                       |
| │ ├── mock_clustering.py \# no target column                          |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ ├── \# Dataset Size Mocks                                           |
|                                                                       |
| │ ├── mock_very_small.py \# 200-300 rows                              |
|                                                                       |
| │ ├── mock_medium.py \# 5,000 rows (default)                          |
|                                                                       |
| │ ├── mock_large.py \# 100,000 rows                                   |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ ├── \# Imbalance Mocks                                              |
|                                                                       |
| │ ├── mock_balanced.py \# 50/50 split                                 |
|                                                                       |
| │ ├── mock_moderate_imbalance.py \# 80/20 split                       |
|                                                                       |
| │ ├── mock_severe_imbalance.py \# 95/5 split                          |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ ├── \# Feature Quality Mocks                                        |
|                                                                       |
| │ ├── mock_high_correlation.py \# correlated feature pairs            |
|                                                                       |
| │ ├── mock_zero_variance.py \# constant columns                       |
|                                                                       |
| │ ├── mock_weak_features.py \# near-zero SHAP features                |
|                                                                       |
| │ ├── mock_leakage.py \# leakage feature present                      |
|                                                                       |
| │ │                                                                   |
|                                                                       |
| │ └── \# Edge Case Mocks                                              |
|                                                                       |
| │ ├── mock_all_features_dropped.py                                    |
|                                                                       |
| │ ├── mock_single_feature.py                                          |
|                                                                       |
| │ ├── mock_many_classes.py \# 15+ classes                             |
|                                                                       |
| │ └── mock_low_optuna_budget.py                                       |
+-----------------------------------------------------------------------+

**9.3 The Mock Generator --- Build Once, Use Everywhere**

Never build each mock from scratch. One parameterized generator function
produces all of them. This also guarantees all mocks have the same
schema --- critical for consistent testing.

+-----------------------------------------------------------------------+
| \# mocks/mock_generator.py                                            |
|                                                                       |
| from sklearn.datasets import make_classification, make_regression,    |
| make_blobs                                                            |
|                                                                       |
| import pandas as pd                                                   |
|                                                                       |
| import numpy as np                                                    |
|                                                                       |
| def make_classification_mock(                                         |
|                                                                       |
| n_samples=5000, n_features=10, n_classes=2,                           |
|                                                                       |
| weights=None, n_informative=6, n_redundant=2, random_state=42         |
|                                                                       |
| ):                                                                    |
|                                                                       |
| X, y = make_classification(                                           |
|                                                                       |
| n_samples=n_samples, n_features=n_features,                           |
|                                                                       |
| n_classes=n_classes, weights=weights,                                 |
|                                                                       |
| n_informative=n_informative, n_redundant=n_redundant,                 |
|                                                                       |
| random_state=random_state                                             |
|                                                                       |
| )                                                                     |
|                                                                       |
| df = pd.DataFrame(X, columns=\[f\'feature\_{i}\' for i in             |
| range(n_features)\])                                                  |
|                                                                       |
| df\[\'target\'\] = y                                                  |
|                                                                       |
| return df                                                             |
|                                                                       |
| def make_regression_mock(n_samples=5000, n_features=10, noise=0.1):   |
|                                                                       |
| X, y = make_regression(                                               |
|                                                                       |
| n_samples=n_samples, n_features=n_features,                           |
|                                                                       |
| noise=noise, random_state=42                                          |
|                                                                       |
| )                                                                     |
|                                                                       |
| df = pd.DataFrame(X, columns=\[f\'feature\_{i}\' for i in             |
| range(n_features)\])                                                  |
|                                                                       |
| df\[\'target\'\] = y                                                  |
|                                                                       |
| return df                                                             |
|                                                                       |
| def make_clustering_mock(n_samples=5000, n_clusters=4,                |
| n_features=10):                                                       |
|                                                                       |
| X, \_ = make_blobs(                                                   |
|                                                                       |
| n_samples=n_samples, n_features=n_features,                           |
|                                                                       |
| centers=n_clusters, random_state=42                                   |
|                                                                       |
| )                                                                     |
|                                                                       |
| \# No target column --- clustering problem                            |
|                                                                       |
| return pd.DataFrame(X, columns=\[f\'feature\_{i}\' for i in           |
| range(n_features)\])                                                  |
|                                                                       |
| \# ── Feature injection helpers ───────────────────────────────────── |
|                                                                       |
| def add_correlated_features(df, base_col, n=2, noise=0.05):           |
|                                                                       |
| for i in range(n):                                                    |
|                                                                       |
| df\[f\'{base_col}\_corr\_{i}\'\] = df\[base_col\] +                   |
| np.random.normal(0, noise, len(df))                                   |
|                                                                       |
| return df                                                             |
|                                                                       |
| def add_zero_variance_feature(df, col_name=\'constant_col\',          |
| value=1):                                                             |
|                                                                       |
| df\[col_name\] = value                                                |
|                                                                       |
| return df                                                             |
|                                                                       |
| def add_weak_features(df, n_weak=3):                                  |
|                                                                       |
| for i in range(n_weak):                                               |
|                                                                       |
| df\[f\'noise\_{i}\'\] = np.random.normal(0, 1, len(df))               |
|                                                                       |
| return df                                                             |
|                                                                       |
| def add_leakage_feature(df, target_col, noise=0.01):                  |
|                                                                       |
| df\[\'leakage_col\'\] = df\[target_col\] + np.random.normal(0, noise, |
| len(df))                                                              |
|                                                                       |
| return df                                                             |
+-----------------------------------------------------------------------+

**9.4 Example Mock Files**

Each mock file is just a few lines --- it uses the generator and defines
matching audit and detection objects.

**mock_binary_classification.py --- Your Main Dev Mock**

+-----------------------------------------------------------------------+
| from mock_generator import make_classification_mock                   |
|                                                                       |
| mock_df = make_classification_mock(n_samples=5000, weights=\[0.84,    |
| 0.16\])                                                               |
|                                                                       |
| mock_audit = {                                                        |
|                                                                       |
| \'shape\': (5000, 10), \'dataset_size_class\': \'medium\',            |
|                                                                       |
| \'rows_to_features_ratio\': 500, \'dimensionality_risk\': False,      |
|                                                                       |
| \'target_column\': \'target\', \'target_dtype\': \'int64\',           |
|                                                                       |
| \'target_unique_values\': 2, \'target_distribution\': {0: 0.84, 1:    |
| 0.16},                                                                |
|                                                                       |
| \'target_valid\': True, \'target_min_class_size\': 800,               |
|                                                                       |
| \'column_types\': {\'numerical_continuous\': \[f\'feature\_{i}\' for  |
| i in range(10)\]},                                                    |
|                                                                       |
| \'missing\': {}, \'drop_candidates\': {}, \'high_cardinality\': \[\], |
|                                                                       |
| \'high_correlations\': \[\], \'leakage_candidates\': {},              |
|                                                                       |
| \'feature_target_correlation\': {f\'feature\_{i}\': {\'score\':       |
| round(0.6 - i\*0.05, 2),                                              |
|                                                                       |
| \'signal\': \'strong\' if i \< 3 else \'moderate\'} for i in          |
| range(10)},                                                           |
|                                                                       |
| \'skewed_columns\': {}, \'outlier_columns\': {}, \'quasi_constant\':  |
| \[\],                                                                 |
|                                                                       |
| \'imbalance_detected\': True, \'imbalance_ratio\': 0.16,              |
|                                                                       |
| \'imbalance_severity\': \'moderate\', \'smote_recommended\': True,    |
|                                                                       |
| \'sampling_recommended\': False, \'sampling_fraction\': None,         |
|                                                                       |
| }                                                                     |
|                                                                       |
| mock_detection = {                                                    |
|                                                                       |
| \'problem_type\': \'classification\', \'detection_method\':           |
| \'inferred\',                                                         |
|                                                                       |
| \'confidence\': \'high\', \'classification_subtype\': \'binary\',     |
|                                                                       |
| \'num_classes\': 2, \'class_labels\': \[0, 1\],                       |
|                                                                       |
| \'metrics_averaging\': \'binary\', \'regression_subtype\': None,      |
|                                                                       |
| \'target_log_transform\': False,                                      |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**mock_severe_imbalance.py**

+-----------------------------------------------------------------------+
| from mock_generator import make_classification_mock                   |
|                                                                       |
| mock_df = make_classification_mock(n_samples=5000, weights=\[0.95,    |
| 0.05\])                                                               |
|                                                                       |
| mock_audit = {                                                        |
|                                                                       |
| \# \... same as binary classification mock \...                       |
|                                                                       |
| \'imbalance_detected\': True,                                         |
|                                                                       |
| \'imbalance_ratio\': 0.05, \# \<- only change                         |
|                                                                       |
| \'imbalance_severity\': \'severe\', \# \<- only change                |
|                                                                       |
| \'smote_recommended\': True,                                          |
|                                                                       |
| }                                                                     |
|                                                                       |
| mock_detection = {                                                    |
|                                                                       |
| \'problem_type\': \'classification\',                                 |
|                                                                       |
| \'classification_subtype\': \'binary\',                               |
|                                                                       |
| \# \... rest same \...                                                |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**mock_large.py**

+-----------------------------------------------------------------------+
| from mock_generator import make_classification_mock                   |
|                                                                       |
| mock_df = make_classification_mock(n_samples=100000, weights=\[0.7,   |
| 0.3\])                                                                |
|                                                                       |
| mock_audit = {                                                        |
|                                                                       |
| \'shape\': (100000, 10),                                              |
|                                                                       |
| \'dataset_size_class\': \'large\', \# \<- triggers LightGBM           |
| preference                                                            |
|                                                                       |
| \'sampling_recommended\': True, \# \<- triggers 20% sampling in       |
| Optuna                                                                |
|                                                                       |
| \'sampling_fraction\': 0.2,                                           |
|                                                                       |
| \# \... rest same \...                                                |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**mock_high_correlation.py**

+-----------------------------------------------------------------------+
| from mock_generator import make_classification_mock,                  |
| add_correlated_features                                               |
|                                                                       |
| mock_df = make_classification_mock(n_samples=5000)                    |
|                                                                       |
| mock_df = add_correlated_features(mock_df, base_col=\'feature_0\',    |
| n=2, noise=0.02)                                                      |
|                                                                       |
| mock_audit = {                                                        |
|                                                                       |
| \# \...                                                               |
|                                                                       |
| \'high_correlations\': \[                                             |
|                                                                       |
| (\'feature_0\', \'feature_0_corr_0\', 0.97),                          |
|                                                                       |
| (\'feature_0\', \'feature_0_corr_1\', 0.95),                          |
|                                                                       |
| \],                                                                   |
|                                                                       |
| \'feature_target_correlation\': {                                     |
|                                                                       |
| \'feature_0\': {\'score\': 0.61, \'signal\': \'strong\'},             |
|                                                                       |
| \'feature_0_corr_0\': {\'score\': 0.12, \'signal\': \'weak\'}, \#     |
| lower --- will be dropped                                             |
|                                                                       |
| \'feature_0_corr_1\': {\'score\': 0.09, \'signal\': \'weak\'}, \#     |
| lower --- will be dropped                                             |
|                                                                       |
| \# \...                                                               |
|                                                                       |
| },                                                                    |
|                                                                       |
| }                                                                     |
+-----------------------------------------------------------------------+

**mock_all_features_dropped.py --- Edge Case**

+-----------------------------------------------------------------------+
| from mock_generator import make_classification_mock,                  |
| add_zero_variance_feature                                             |
|                                                                       |
| import pandas as pd                                                   |
|                                                                       |
| import numpy as np                                                    |
|                                                                       |
| \# Only constant and noise columns --- everything should get dropped  |
|                                                                       |
| mock_df = pd.DataFrame({                                              |
|                                                                       |
| \'constant_1\': \[1\] \* 5000,                                        |
|                                                                       |
| \'constant_2\': \[0\] \* 5000,                                        |
|                                                                       |
| \'noise_1\': np.random.normal(0, 0.00001, 5000), \# near-zero         |
| variance                                                              |
|                                                                       |
| \'target\': np.random.randint(0, 2, 5000),                            |
|                                                                       |
| })                                                                    |
|                                                                       |
| \# Pipeline should catch this and exit with clean error:              |
|                                                                       |
| \# \'Feature selection removed all features. Please review data       |
| quality.\'                                                            |
+-----------------------------------------------------------------------+

**9.5 Complete Mock Coverage Table**

One mock per distinct scenario. Every case your pipeline will ever
encounter should have a corresponding mock to test against.

  ---------------------------- ---------------------- -----------------------------
  **Mock File**                **What It Tests**      **Priority**

  mock_binary_classification   Core pipeline --- main 1 --- Build Day 1
                               dev mock               

  mock_regression              Completely different   1 --- Build Day 1
                               metrics + plots        

  mock_clustering              No target, different   1 --- Build Day 1
                               flow entirely          

  mock_severe_imbalance        SMOTE behavior +       2 --- Build Day 2
                               strong limitation      

  mock_very_small              Depth constraints +    2 --- Build Day 2
                               small dataset          
                               limitation             

  mock_large                   Sampling + LightGBM    2 --- Build Day 2
                               preference             

  mock_high_correlation        Feature selection      3 --- Build Day 3
                               Level 2                

  mock_zero_variance           Feature selection      3 --- Build Day 3
                               Level 1                

  mock_weak_features           Feature selection      3 --- Build Day 3
                               Level 3 (SHAP)         

  mock_leakage                 Leakage limitation     3 --- Build Day 3
                               generation             

  mock_multiclass              Weighted F1,           3 --- Build Day 3
                               multiclass SHAP        

  mock_all_features_dropped    Graceful failure       4 --- Build as needed
                               handling               

  mock_single_feature          Edge case --- one      4 --- Build as needed
                               feature remains        

  mock_many_classes            many_class subtype     4 --- Build as needed
                               warning                

  mock_heteroscedastic         Regression             4 --- Build as needed
                               heteroscedasticity     
                               flag                   

  mock_skewed_target           Log transform          4 --- Build as needed
                               recommendation         
  ---------------------------- ---------------------- -----------------------------

**9.6 Test Runner --- Run All Mocks in One Command**

Build this on Day 1 alongside your first mock. Before integration you
should be able to run one command and see all cases pass or fail
clearly.

+-----------------------------------------------------------------------+
| \# test_all_mocks.py                                                  |
|                                                                       |
| import traceback                                                      |
|                                                                       |
| from mocks import (                                                   |
|                                                                       |
| mock_binary_classification,                                           |
|                                                                       |
| mock_regression,                                                      |
|                                                                       |
| mock_clustering,                                                      |
|                                                                       |
| mock_severe_imbalance,                                                |
|                                                                       |
| mock_very_small,                                                      |
|                                                                       |
| mock_large,                                                           |
|                                                                       |
| mock_high_correlation,                                                |
|                                                                       |
| mock_all_features_dropped,                                            |
|                                                                       |
| )                                                                     |
|                                                                       |
| from core.feature_selector import run_feature_selection               |
|                                                                       |
| from core.tuner import run_optuna_study                               |
|                                                                       |
| from core.evaluator import run_evaluation                             |
|                                                                       |
| mocks = \[                                                            |
|                                                                       |
| mock_binary_classification,                                           |
|                                                                       |
| mock_regression,                                                      |
|                                                                       |
| mock_clustering,                                                      |
|                                                                       |
| mock_severe_imbalance,                                                |
|                                                                       |
| mock_very_small,                                                      |
|                                                                       |
| mock_large,                                                           |
|                                                                       |
| mock_high_correlation,                                                |
|                                                                       |
| mock_all_features_dropped,                                            |
|                                                                       |
| \]                                                                    |
|                                                                       |
| results = \[\]                                                        |
|                                                                       |
| for mock in mocks:                                                    |
|                                                                       |
| name = mock.\_\_name\_\_.replace(\'mocks.\', \'\')                    |
|                                                                       |
| try:                                                                  |
|                                                                       |
| X, y = mock.mock_df.drop(\'target\', axis=1),                         |
| mock.mock_df\[\'target\'\]                                            |
|                                                                       |
| X_sel = run_feature_selection(X, y, mock.mock_audit)                  |
|                                                                       |
| study, model = run_optuna_study(X_sel, y, mock.mock_detection,        |
|                                                                       |
| mock.mock_audit, time_budget=30)                                      |
|                                                                       |
| evaluation = run_evaluation(model, X_sel, y, mock.mock_detection,     |
|                                                                       |
| mock.mock_audit, study)                                               |
|                                                                       |
| results.append((name, \'PASS\', None))                                |
|                                                                       |
| except Exception as e:                                                |
|                                                                       |
| results.append((name, \'FAIL\', str(e)))                              |
|                                                                       |
| print()                                                               |
|                                                                       |
| print(\'━\' \* 60)                                                    |
|                                                                       |
| print(\' MOCK TEST RESULTS\')                                         |
|                                                                       |
| print(\'━\' \* 60)                                                    |
|                                                                       |
| for name, status, error in results:                                   |
|                                                                       |
| icon = \'✔\' if status == \'PASS\' else \'✘\'                         |
|                                                                       |
| print(f\' {icon} {name:\<35} {status}\')                              |
|                                                                       |
| if error:                                                             |
|                                                                       |
| print(f\' Error: {error}\')                                           |
|                                                                       |
| print(\'━\' \* 60)                                                    |
|                                                                       |
| passed = sum(1 for \_, s, \_ in results if s == \'PASS\')             |
|                                                                       |
| print(f\' {passed}/{len(results)} passed\')                           |
|                                                                       |
| print()                                                               |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Expected Output When Everything Passes**                            |
|                                                                       |
| +------------------------------------------------------------------+  |
| | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     |  |
| |                                                                  |  |
| | MOCK TEST RESULTS                                                |  |
| |                                                                  |  |
| | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     |  |
| |                                                                  |  |
| | ✔ mock_binary_classification PASS                                |  |
| |                                                                  |  |
| | ✔ mock_regression PASS                                           |  |
| |                                                                  |  |
| | ✔ mock_clustering PASS                                           |  |
| |                                                                  |  |
| | ✔ mock_severe_imbalance PASS                                     |  |
| |                                                                  |  |
| | ✔ mock_very_small PASS                                           |  |
| |                                                                  |  |
| | ✔ mock_large PASS                                                |  |
| |                                                                  |  |
| | ✔ mock_high_correlation PASS                                     |  |
| |                                                                  |  |
| | ✔ mock_all_features_dropped PASS                                 |  |
| |                                                                  |  |
| | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     |  |
| |                                                                  |  |
| | 8/8 passed                                                       |  |
| +------------------------------------------------------------------+  |
+-----------------------------------------------------------------------+

**9.7 Development Timeline**

  ---------------------- ------------------------------------------------
  **Day**                **Task**

  Day 1                  Build mock_generator.py +
                         mock_binary_classification + mock_regression +
                         mock_clustering. Set up test_all_mocks.py.
                         Confirm data flows end-to-end even if results
                         are wrong.

  Day 2                  Build feature_selector.py. Test against
                         mock_binary_classification,
                         mock_high_correlation, mock_zero_variance,
                         mock_weak_features. All Level 1-3 selection
                         working.

  Day 3-4                Build tuner.py. Optuna objective function for
                         classification + regression. Narration callback.
                         Test against mock_binary_classification,
                         mock_very_small, mock_large.

  Day 5                  Extend tuner.py for clustering. Elbow +
                         silhouette auto-detection. Test against
                         mock_clustering.

  Day 6-7                Build evaluator.py. All metrics, SHAP plots,
                         confusion matrix inference, limitations
                         generator. Test all mocks.

  Day 8                  Run full test suite against all mocks. Fix
                         failures. Clean up code, add docstrings.

  Integration Day        Swap mock imports for real Person 1 outputs. Run
                         test suite again on real data. Fix interface
                         mismatches.
  ---------------------- ------------------------------------------------

**10. Testing Checklist**

Run this checklist before declaring your module ready for integration.
Every row should pass on both mock data and real data.

  ---------------------- ------------------------------------------------
  **Test Case**          **Expected Behaviour**

  Binary classification  F1 metric, confusion matrix + inference, binary
  mock                   SHAP, limitations generated

  Multiclass             Weighted F1, per-class metrics, multiclass SHAP
  classification mock    summary

  Regression mock        RMSE/MAE/R2, residual plot, heteroscedasticity
                         flag if applicable

  Clustering mock        Silhouette score, cluster profiles table, PCA
                         visualization

  Very small dataset     Small dataset limitation auto-generated, XGBoost
  mock                   depth capped

  Large dataset mock     Sampling applied, LightGBM favored, SVM excluded
                         from pool

  Severe imbalance mock  Imbalance limitation generated, scale_pos_weight
                         in XGBoost/LightGBM

  High correlation mock  Weaker of each correlated pair dropped in Level
                         2

  Zero variance mock     Constant column dropped in Level 1

  Weak features mock     Near-zero SHAP features dropped in Level 3

  All features dropped   Clean error message, no stack trace
  mock                   

  Optuna time_budget=10s Completes gracefully, low trial count limitation
                         added

  Full pipeline joblib   Load saved pipeline, run predict on raw input,
  save                   correct output

  Test runner passes 8/8 All mocks green before integration
  ---------------------- ------------------------------------------------

+-----------------------------------------------------------------------+
| **Ready for Integration When:**                                       |
|                                                                       |
| -   test_all_mocks.py passes 8/8 (or all mocks you have built)        |
|                                                                       |
| -   Saved joblib pipeline loads and predicts correctly on new raw     |
|     data                                                              |
|                                                                       |
| -   Evaluation object matches the agreed schema exactly --- Person 3  |
|     depends on this                                                   |
|                                                                       |
| -   No raw stack traces anywhere --- all failures produce clean       |
|     narrated error messages                                           |
|                                                                       |
| -   Code is readable with docstrings on every function --- teammates  |
|     will need to understand it                                        |
+-----------------------------------------------------------------------+
