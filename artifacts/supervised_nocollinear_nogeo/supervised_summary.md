# Supervised Learning Summary (Churn Prediction)

## Data

- Target: Churn (1 = churn, 0 = non-churn).
- Note: The dataset is class-imbalanced (churn is a minority class), so recall/F1 are emphasized.

## Feature set used in this run

- Feature filter: {'drop_state': True, 'drop_area_code': True, 'drop_charges': True, 'n_features_before': 73, 'n_features_after': 15, 'removed_features_count': 58}

## Models

- K-Nearest Neighbors (distance-based baseline)
- Logistic Regression (linear baseline)
- Random Forest (tree ensemble baseline)

## Hyperparameter tuning

- Tuning uses training data only via stratified 5-fold cross-validation.
- Optimization metric: F1-score.

- logreg: best_cv_f1=0.4773, best_params={'penalty': 'l1', 'class_weight': None, 'C': 5.0}
- knn: best_cv_f1=0.4457, best_params={'weights': 'uniform', 'p': 1, 'n_neighbors': 3}
- rf: best_cv_f1=0.8126, best_params={'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': None, 'class_weight': None}

## Threshold optimization

- For selected models, the decision threshold is optimized using training data only (out-of-fold probabilities).
- Variants containing `threshold` in the table reflect this post-processing step.

## Results (train vs test)

```
 model         variant split  accuracy  precision  recall     f1  roc_auc
logreg        baseline train    0.7697     0.3605  0.7526 0.4875   0.8248
logreg        baseline  test    0.7751     0.3618  0.7579 0.4898   0.8302
   knn        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
   knn        baseline  test    0.8771     0.8824  0.1579 0.2679   0.8741
    rf        baseline train    1.0000     1.0000  1.0000 1.0000   1.0000
    rf        baseline  test    0.9415     0.9828  0.6000 0.7451   0.9196
logreg           tuned train    0.7693     0.3604  0.7552 0.4879   0.8249
logreg           tuned  test    0.7766     0.3650  0.7684 0.4949   0.8308
logreg threshold_tuned train    0.8083     0.4041  0.6675 0.5034   0.8249
logreg threshold_tuned  test    0.8036     0.3929  0.6947 0.5019   0.8308
   knn           tuned train    0.9329     0.9300  0.5825 0.7163   0.9709
   knn           tuned  test    0.8966     0.7955  0.3684 0.5036   0.7836
   knn threshold_tuned train    0.8938     0.5782  1.0000 0.7328   0.9709
   knn threshold_tuned  test    0.8111     0.4037  0.6842 0.5078   0.7836
    rf           tuned train    0.9985     0.9898  1.0000 0.9949   0.9999
    rf           tuned  test    0.9595     0.9146  0.7895 0.8475   0.9151
```

## Key observations

- Best tuned model on the test set (by F1): rf (F1=0.8475, precision=0.9146, recall=0.7895, accuracy=0.9595).
- Best model confusion matrix (test): TN=565, FP=7, FN=20, TP=75.
- knn baseline: train-test F1 gap = 0.7321.
- knn tuned: train-test F1 gap = 0.2127.
- logreg baseline: train-test F1 gap = -0.0023.
- logreg tuned: train-test F1 gap = -0.0070.
- rf baseline: train-test F1 gap = 0.2549.
- rf tuned: train-test F1 gap = 0.1474.

## Interpretation and diagnostics

- Very high training scores with much lower test scores suggest overfitting (common for KNN/Random Forest if not regularized).
- If accuracy is high but recall is low, the model may miss churners due to class imbalance.
- Confusion matrices are exported for each model/variant (train and test) to inspect FP/FN trade-offs.

## Connection to EDA (feature considerations)

- EDA found near-perfect redundancy between minutes and charges (e.g., Total day minutes vs Total day charge).
- For interpretability and to reduce collinearity in linear models, consider keeping only one from each redundant pair (minutes *or* charges).
- Area/state features are one-hot encoded and can increase dimensionality; optionally exclude them if they are weakly associated with churn.
