# Stage 6 – Model Interpretation and Insights

## 6.1 Key Features Influencing Churn

Two methods applied on the same 15-feature set (nocollinear_nogeo filter):
- **Random Forest**: Gini importance (mean decrease in impurity)
- **Logistic Regression**: |coefficient| after feature standardisation

### RF Feature Importance (top 10)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Total day minutes | 0.244 |
| 2 | Customer service calls | 0.159 |
| 3 | International plan = Yes | 0.145 |
| 4 | Total eve minutes | 0.101 |
| 5 | Total intl minutes | 0.070 |
| 6 | Total intl calls | 0.048 |
| 7 | Total night minutes | 0.047 |
| 8–10 | Total night/day calls, Account length | < 0.035 |

Figure: `artifacts/genai/interpretation/rf_feature_importances.png`

### Logistic Regression |Coefficient| (top 8)

| Rank | Feature | \|Coef\| | Direction |
|------|---------|--------|-----------|
| 1 | International plan = Yes | 2.093 | ↑ churn |
| 2 | Voice mail plan = Yes | 1.899 | ↓ churn |
| 3 | Customer service calls | 0.759 | ↑ churn |
| 4 | Total day minutes | 0.668 | ↑ churn |
| 5 | Number vmail messages | 0.522 | ↓ churn |
| 6 | Total eve minutes | 0.275 | ↑ churn |
| 7 | Total intl minutes | 0.217 | ↑ churn |
| 8+ | Call counts, Account length | < 0.21 | weak |

Figure: `artifacts/genai/interpretation/logreg_coef_magnitudes.png`

### Cross-model consensus

| Priority | Feature | Key finding |
|----------|---------|-------------|
| Critical | **Total day minutes** | #1 RF, #4 LR — heavy daytime usage → bill shock → churn |
| Critical | **Customer service calls** | #2 RF, #3 LR — each call is a dissatisfaction signal |
| Critical | **International plan** | #3 RF, #1 LR — largest single linear predictor |
| High | **Voice mail plan** | #2 LR — voicemail adoption strongly reduces churn |
| High | **Eve/intl minutes** | #4–5 RF — secondary usage dimensions add to bill pressure |
| Low | **Account length, call counts** | Weak predictors across both models |

---

## 6.2 Actionable Retention Strategies

1. **High-usage alert**: Flag customers with Total day minutes above the 75th percentile (~225 min). Proactively offer an unlimited plan before their bill arrives.

2. **Service call escalation**: After a customer's 2nd service call in 30 days, route to a retention specialist. Customers with ≥4 calls have roughly 3× the churn odds of those with 0.

3. **International plan pricing review**: Customers on international plans churn at the highest rate. Introduce per-minute cost alerts or a transparent usage cap to reduce bill surprise.

4. **Voicemail adoption campaign**: Customers without a voicemail plan churn more. A free 3-month trial targeted at non-adopters increases stickiness at low cost.

5. **Usage dashboard**: Customers with high combined usage (day + eve + intl minutes) across tiers accumulate churn risk additively. A monthly bill-projection notification reduces surprise across all tiers simultaneously.

---

## 6.3 Gen-AI in a Data-Mining Study

| Use case | How it helps |
|----------|-------------|
| **Data augmentation** | Synthesise additional minority-class samples (churners) to improve recall without collecting more real data |
| **Class-imbalance handling** | CTGAN's conditional sampling oversamples rare classes while preserving feature correlations — unlike SMOTE, which interpolates without regard to joint distributions |
| **Privacy preservation** | Replace PII-containing records with statistically equivalent synthetic ones for safe sharing |
| **Stress-testing** | Generate edge-case profiles (e.g., extremely high usage) to test classifier robustness under distribution shift |
| **What-if simulation** | Conditionally generate customers under hypothetical plan changes to evaluate retention policy impact before deployment |

**Lesson from this project**: CTGAN reproduced marginal distributions and class imbalance well, but the performance gap on synthetic vs real test data shows that marginal fidelity does not guarantee joint fidelity. Gen-AI synthetic data is best used to supplement real data, not replace it as a test set.
