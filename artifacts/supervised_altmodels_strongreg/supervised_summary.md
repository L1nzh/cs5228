# Supervised Learning Summary (Churn Prediction)

## Data

- Target: Churn (1 = churn, 0 = non-churn).
- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.

## Feature set used in this run

- Feature filter: {'drop_state': True, 'drop_area_code': True, 'drop_charges': True, 'n_features_before': 73, 'n_features_after': 15, 'removed_features_count': 58}

## Models

- ExtraTrees (randomized tree ensemble)
- HistGradientBoosting (boosted trees)
- Random Forest (tree ensemble baseline)

## Hyperparameter tuning

- Tuning uses training data only via stratified 5-fold cross-validation.
- Optimization metric: F1-score.

- hgb: best_cv_f1=0.8259, best_params={'min_samples_leaf': 30, 'max_iter': 1000, 'max_depth': 5, 'learning_rate': 0.05, 'l2_regularization': 5.0}
- et: best_cv_f1=0.6473, best_params={'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2', 'max_depth': 12, 'class_weight': None}
- rf: best_cv_f1=0.7885, best_params={'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2', 'max_depth': 12, 'class_weight': None}

## Results (train vs test)

```
model  variant split  accuracy  precision  recall     f1  roc_auc
  hgb baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
  hgb baseline  test    0.9505     0.8875  0.7474 0.8114   0.9062
   et baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   et baseline  test    0.9085     0.9722  0.3684 0.5344   0.9138
   rf baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   rf baseline  test    0.9415     0.9828  0.6000 0.7451   0.9196
  hgb    tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
  hgb    tuned  test    0.9520     0.8539  0.8000 0.8261   0.9139
   et    tuned train    0.8897     0.5813  0.8660 0.6957   0.9370
   et    tuned  test    0.8801     0.5532  0.8211 0.6610   0.8764
   rf    tuned train    0.9644     0.8786  0.8763 0.8774   0.9949
   rf    tuned  test    0.9370     0.7524  0.8316 0.7900   0.9247
```

## Key observations

- Best tuned model on the test set (by F1): hgb (F1=0.8261, precision=0.8539, recall=0.8000, accuracy=0.9520).
- Best model confusion matrix (test): TN=559, FP=13, FN=19, TP=76.
- et baseline: train-test F1 gap = 0.4656.
- et tuned: train-test F1 gap = 0.0346.
- hgb baseline: train-test F1 gap = 0.1886.
- hgb tuned: train-test F1 gap = 0.1739.
- rf baseline: train-test F1 gap = 0.2549.
- rf tuned: train-test F1 gap = 0.0874.

## Interpretation and diagnostics

- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).
- If accuracy is high but recall is low, the model may miss churners due to class imbalance.
- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.

## Connection to EDA (feature considerations)

- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).
- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).
- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.
