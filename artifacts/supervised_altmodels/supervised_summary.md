# Supervised Learning Summary (Churn Prediction)

## Data

- Target: Churn (1 = churn, 0 = non-churn).
- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.

## Feature set used in this run

- Feature filter: {'drop_state': False, 'drop_area_code': False, 'drop_charges': False, 'n_features_before': 73, 'n_features_after': 73, 'removed_features_count': 0}

## Models

- ExtraTrees (randomized tree ensemble)
- HistGradientBoosting (boosted trees)
- Random Forest (tree ensemble baseline)

## Hyperparameter tuning

- Tuning uses training data only via stratified 5-fold cross-validation.
- Optimization metric: F1-score.

- hgb: best_cv_f1=0.8450, best_params={'min_samples_leaf': 10, 'max_iter': 200, 'max_depth': 7, 'learning_rate': 0.2, 'l2_regularization': 1.0}
- et: best_cv_f1=0.8133, best_params={'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': None, 'class_weight': None}
- rf: best_cv_f1=0.8016, best_params={'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': None, 'class_weight': None}

## Results (train vs test)

```
model  variant split  accuracy  precision  recall     f1  roc_auc
  hgb baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
  hgb baseline  test    0.9520     0.8987  0.7474 0.8161   0.9051
   et baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   et baseline  test    0.9205     0.9200  0.4842 0.6345   0.9097
   rf baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   rf baseline  test    0.9385     0.9355  0.6105 0.7389   0.9113
  hgb    tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
  hgb    tuned  test    0.9580     0.9036  0.7895 0.8427   0.9098
   et    tuned train    0.9996     0.9974  1.0000 0.9987   1.0000
   et    tuned  test    0.9505     0.8780  0.7579 0.8136   0.9239
   rf    tuned train    0.9985     0.9898  1.0000 0.9949   0.9999
   rf    tuned  test    0.9595     0.9048  0.8000 0.8492   0.9144
```

## Key observations

- Best tuned model on the test set (by F1): rf (F1=0.8492, precision=0.9048, recall=0.8000, accuracy=0.9595).
- Best model confusion matrix (test): TN=564, FP=8, FN=19, TP=76.
- et baseline: train-test F1 gap = 0.3655.
- et tuned: train-test F1 gap = 0.1852.
- hgb baseline: train-test F1 gap = 0.1839.
- hgb tuned: train-test F1 gap = 0.1573.
- rf baseline: train-test F1 gap = 0.2611.
- rf tuned: train-test F1 gap = 0.1457.

## Interpretation and diagnostics

- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).
- If accuracy is high but recall is low, the model may miss churners due to class imbalance.
- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.

## Connection to EDA (feature considerations)

- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).
- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).
- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.
