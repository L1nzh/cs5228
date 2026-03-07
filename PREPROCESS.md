# Preprocessing Pipeline (Train-Fit, Test-Transform)

This document explains what the preprocessing stage does, why each step is used, and what artifacts are produced. The implementation lives in [preprocess.py](file:///Users/bytedance/Documents/trae_projects/cs5228/src/preprocess.py) and is invoked by [run_preprocess_eda.py](file:///Users/bytedance/Documents/trae_projects/cs5228/run_preprocess_eda.py).

## Inputs
- Training set: `churn-bigml-80.csv` (2666 rows)
- Test set: `churn-bigml-20.csv` (667 rows)
- Target column: `Churn` (binary)

## Step 1 — Load data and encode the target
**What**
- Read each CSV into a DataFrame.
- Convert `Churn` to a numeric label:
  - `TRUE → 1`
  - `FALSE → 0`
- Split into features `X` (all columns except `Churn`) and label `y`.

**Why**
- Most ML libraries (e.g., scikit-learn) expect labels as numbers.
- A strict mapping catches unexpected label values early to prevent silent data issues.

**Where**
- [load_csv](file:///Users/bytedance/Documents/trae_projects/cs5228/src/data.py)

## Step 2 — Validate train/test schema consistency
**What**
- Ensure training features and test features have the exact same columns (names and order).

**Why**
- Prevents feature misalignment (missing/extra columns) that can break transforms or invalidate evaluation.

**Where**
- [assert_same_schema](file:///Users/bytedance/Documents/trae_projects/cs5228/src/data.py)

## Step 3 — Infer categorical vs numeric feature columns
**What**
- Treat `object` dtype columns as categorical.
- Treat non-`object` columns as numeric.
- Additionally, treat `Area code` as categorical by default (configurable).

**Why**
- Categorical variables require encoding (e.g., One-Hot) before feeding into most models.
- `Area code` is numeric-looking but semantically an enum-like identifier; treating it as categorical avoids imposing an artificial “distance” between codes (e.g., 408 vs 415).

**Where**
- [infer_feature_types](file:///Users/bytedance/Documents/trae_projects/cs5228/src/preprocess.py)

## Step 4 — Build a train-fitted preprocessing transformer (ColumnTransformer)
All preprocessing is **fit only on the training set** and then applied to both train and test.

### 4.1 Missing-value imputation
**What**
- Categorical columns: impute with most frequent value.
- Numeric columns: impute with median.

**Why**
- Simple, robust defaults that keep the pipeline reproducible.
- Median is less sensitive to outliers than mean for numeric features.

### 4.2 Categorical encoding (One-Hot)
**What**
- One-Hot encode categorical columns.
- Use `handle_unknown="ignore"` so unseen test categories do not crash the pipeline.

**Why**
- One-Hot is a widely compatible encoding for linear models, tree models, clustering, etc.
- Ignoring unknown categories makes the test-time transform robust.

### 4.3 Numeric scaling (StandardScaler)
**What**
- Standardize numeric columns to zero mean and unit variance.

**Why**
- Scale-sensitive algorithms (e.g., KNN, K-Means, Logistic Regression) benefit from standardized numeric inputs.
- Makes coefficients/distances easier to interpret and optimization more stable.

**Where (all of Step 4)**
- [build_preprocessor](file:///Users/bytedance/Documents/trae_projects/cs5228/src/preprocess.py)

## Step 5 — Fit on train, transform train and test, and create feature names
**What**
- Fit the preprocessor on `X_train` (and `y_train` for compatibility).
- Transform both `X_train` and `X_test` into numeric matrices.
- Obtain transformed feature names (One-Hot expanded columns) when available.

**Why**
- Training-only fitting avoids data leakage.
- Feature names are critical for debugging, interpretation, and later reporting.

**Where**
- [fit_transform](file:///Users/bytedance/Documents/trae_projects/cs5228/src/preprocess.py)

## Step 6 — Export preprocessing artifacts
Artifacts are written to: [artifacts/preprocess/](file:///Users/bytedance/Documents/trae_projects/cs5228/artifacts/preprocess)

**What is exported**
- `preprocessor.joblib`: the fitted preprocessing transformer (for reuse later)
- `feature_names.json`: list of transformed feature names (column order is important)
- `X_train_processed.csv`, `X_test_processed.csv`: processed feature matrices (no label)
- `y_train.csv`, `y_test.csv`: labels (0/1)
- `train_processed_with_y.csv`, `test_processed_with_y.csv`: convenience files (X + y)
- `meta.json`: row counts and the final feature dimension

**Why**
- Reproducibility: later stages (clustering/classification) can reuse the exact same preprocessing.
- Debuggability: you can inspect the processed matrices and the feature mapping directly.

**Where**
- [save_artifacts](file:///Users/bytedance/Documents/trae_projects/cs5228/src/preprocess.py)

## Results (this run)
- Train size: 2666
- Test size: 667
- Final processed feature dimension: 73 (after One-Hot expansion)  
  Source: [meta.json](file:///Users/bytedance/Documents/trae_projects/cs5228/artifacts/preprocess/meta.json)

