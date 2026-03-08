# AutoML Pipeline: From Spreadsheet to Explained Predictions

Turn any spreadsheet into a working, explained, and deployable prediction model — automatically, transparently, and without requiring you to know anything about machine learning.

**You give it a spreadsheet. It does the rest.**

Run one command, point it at your file, and the tool walks through the entire data science process on its own. Best of all, it **narrates every decision it makes in plain English**, just like a smart assistant explaining its thinking out loud.

---

## 🏗️ Design Decisions & Architecture

This pipeline is built on the philosophy of **Single Responsibility** and **Structural Decoupling**. The math is secondary to the structure. By keeping components entirely isolated, the pipeline fails loudly and locally, making it genuinely production-ready and extensible.

### Why do we use an Audit Object contract?
The `preprocessor` and the `feature_selector` never talk to each other directly. Instead, they communicate strictly through a JSON-like `audit_object` generated in Phase 1. This decoupling ensures that if we completely swap out our preprocessing math tomorrow, the feature selector won't break, because the interface contract remains identical.

### Why do we use Statistical Distribution Testing for Target Detection?
Instead of relying on naive rules (e.g., "if unique values < 20, it's classification"), `detector.py` uses `scipy.stats` to prove its conclusions mathematically. It measures Shannon entropy, gap variance, and D'Agostino normality. A float column can be classified as categorical if its entropy and gap variance mathematically prove it behaves like a discrete ordinal variable. 

### Why do we swap the traditional order of Correlation and SHAP filtering?
Most pipelines run correlation filters first to drop redundant features before running SHAP. We do the opposite. If you drop correlated features first, you might blindly drop the feature the model actually prefers. 
We train a multi-seed LightGBM probe *first* to generate a stable **Consensus Importance Array**. Then, when we resolve correlated pairs, we use that array to intelligently keep the feature that actually drives model performance.

### Why do we use a Relative SHAP Threshold instead of Absolute?
An absolute threshold (e.g., `mean(|SHAP|) < 0.001`) is fragile. On datasets with sparse signals, you might accidentally drop 100% of your features. By using a relative threshold (`< 1% of the Maximum Importance`), the feature selector scales dynamically to the unique signal distribution of every dataset.

### Why `StratifiedKFold` vs `KFold`?
Classification tasks strictly use `StratifiedKFold` to ensure minority classes are represented equally across all train/test splits. Regression uses standard `KFold`. This explicit separation (enforced by `detector.py`'s output) prevents target leakage and crash-loops on highly imbalanced data.

---

## 🌟 Why is this impressive?

Most machine learning tools are black boxes — they give you a result, but no explanation. 
This tool narrates every single decision: Why it chose one model over another. Why it handled data imbalance a certain way. What its limitations are. This transparency is rare and genuinely useful for real-world decision-making.

## ✨ Features

- **Automated Health Check**: Scans your data like a doctor reading a checkup report. Finds missing data, extreme outliers, and columns that don't make sense.
- **Smart Problem Detection**: Automatically figures out whether your data needs Classification (yes/no), Regression (predicting a number), or Clustering (finding natural groups).
- **Intelligent Data Preprocessing**: Cleans gaps, converts text to machine-readable formats, and fixes imbalanced categories automatically.
- **Model Competition**: Runs several prediction approaches head-to-head. Uses a smart search strategy to find the absolute best combination of technique and settings.
- **Plain-English Explanations**: Tells you exactly which factors in your data matter most, which ones barely matter, and where the model is likely to make mistakes.
- **Deployment Ready**: Saves a comprehensive PDF report and packages the finished model so you can use it later or serve it instantly via an API.

## 🛠️ Tech Stack

Under the hood, this pipeline utilizes industry-standard technologies to deliver robust performance:

*   **Core Machine Learning**: `scikit-learn`, `XGBoost`, `LightGBM`
*   **Hyperparameter Optimization**: `Optuna` (Smart TPE sampling)
*   **Data Processing**: `pandas`, `numpy`, `imbalanced-learn`
*   **Explainability (XAI)**: `SHAP` (SHapley Additive exPlanations)
*   **Visualizations**: `matplotlib`, `seaborn`
*   **Deployment & Serialization**: `joblib`

## ⚙️ How It Works (In plain English)

1. **Load**: You provide a CSV file full of data.
2. **Audit**: The system looks for problems (missing data, useless columns).
3. **Detect**: It identifies the target question (e.g., predicting churn, estimating price, grouping customers).
4. **Clean**: Data is polished, encoded, and balanced.
5. **Select Features**: The system rigorously filters out noise, keeping only the data points that matter.
6. **Tune & Train**: It searches thousands of combinations intelligently to find the most accurate model.
7. **Evaluate**: It evaluates the winner, generating insights on accuracy and limitations.
8. **Report & Deploy**: You get a fully explained PDF and a ready-to-use model.

## 🎯 Who is this for?

- **Small Business Owners**: Predict customer churn without hiring a data scientist.
- **Researchers**: Quickly explore a new dataset without spending days on boilerplate setup.
- **Startups**: Get a working ML prototype without the cost of a full ML engineer.
- **Students & Analysts**: People who understand their domain deeply, but don't want to get bogged down in ML machinery.

---

## 💻 Setup Intructions

```bash
# Clone the repository
git clone <repo-url>
cd AutoML

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install all required dependencies
pip install -r requirements.txt
```
