# Supervised Learning Summary (Churn Prediction)

## Data

- Target: Churn (1 = churn, 0 = non-churn).
- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.

## Models

- Logistic Regression (linear baseline)
- K-Nearest Neighbors (distance-based baseline)
- Random Forest (tree ensemble baseline)

## Hyperparameter tuning

- Tuning uses training data only via stratified 5-fold cross-validation.
- Optimization metric: F1-score.

- logreg: best_cv_f1=0.4850, best_params={'penalty': 'l2', 'class_weight': 'balanced', 'C': 0.2}
- knn: best_cv_f1=0.4137, best_params={'weights': 'uniform', 'p': 1, 'n_neighbors': 3}
- rf: best_cv_f1=0.8096, best_params={'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 20, 'class_weight': None}

## Results (train vs test)

```
 model  variant split  accuracy  precision  recall     f1  roc_auc
logreg baseline train    0.7776     0.3730  0.7758 0.5038   0.8455
logreg baseline  test    0.7766     0.3650  0.7684 0.4949   0.8188
   knn baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   knn baseline  test    0.8861     0.9524  0.2105 0.3448   0.8731
    rf baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
    rf baseline  test    0.9445     1.0000  0.6105 0.7582   0.9293
logreg    tuned train    0.7802     0.3753  0.7680 0.5042   0.8427
logreg    tuned  test    0.7781     0.3695  0.7895 0.5034   0.8259
   knn    tuned train    0.9269     0.9447  0.5284 0.6777   0.9706
   knn    tuned  test    0.8966     0.7708  0.3895 0.5175   0.7668
    rf    tuned train    1.0000     1.0000  1.0000 1.0000   1.0000
    rf    tuned  test    0.9595     0.9359  0.7684 0.8439   0.9199
```

## Key observations

- Best tuned model on the test set (by F1): rf (F1=0.8439, precision=0.9359, recall=0.7684, accuracy=0.9595).
- Best model confusion matrix (test): TN=567, FP=5, FN=22, TP=73.
- knn baseline: train-test F1 gap = 0.6552.
- knn tuned: train-test F1 gap = 0.1602.
- logreg baseline: train-test F1 gap = 0.0089.
- logreg tuned: train-test F1 gap = 0.0009.
- rf baseline: train-test F1 gap = 0.2418.
- rf tuned: train-test F1 gap = 0.1561.

## Interpretation and diagnostics

- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).
- If accuracy is high but recall is low, the model may miss churners due to class imbalance.
- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.

## Connection to EDA (feature considerations)

- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).
- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).
- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.
