# Stage 5 – Generative AI: CTGAN Synthesis

**Script**: `run_genai.py` | **Training data**: `churn-bigml-80.csv` | **Output**: `churn-gen-ai-test-data.csv` (500 rows)

---

## 1. Algorithm: CTGAN

We used **CTGAN** (Conditional Tabular GAN) via the SDV library (v1.35), trained for **300 epochs** on the training set only.

Two design choices make CTGAN suitable here:
- **Conditional generation**: samples minority classes (Churn=TRUE, ~14%) proportionally during training, preventing mode collapse on the majority class.
- **VGM normalisation**: models each numerical column as a mixture of Gaussians, capturing non-Gaussian shapes without parametric assumptions.

Training config: State / International plan / Voice mail plan / Churn set as categorical; integer columns (call counts, Account length, etc.) rounded after sampling.

---

## 2. Histogram Comparison – Columns D, E, G, H, I

| Col | Feature | Synthetic | Original Test | Notes |
|-----|---------|-----------|---------------|-------|
| D | International plan (Yes%) | 11.4% | 9.7% | +1.7 pp — slight over-representation |
| E | Voice mail plan (Yes%) | 11.4% | 28.3% | Under-represented; CTGAN mode-drop |
| G | Total day minutes (mean) | 196.0 | 180.9 | +8.4% drift |
| H | Total day calls (mean) | 106.8 | 100.9 | +5.8% drift |
| I | Total day charge (mean) | 36.7 | 30.8 | +19.2%; CTGAN treats it semi-independently of col G |

Visual comparisons: `artifacts/genai/histograms/`

---

## 3. Model Evaluation on Synthetic Data

Models loaded from `artifacts/supervised_nocollinear_nogeo/` (15-feature filtered set). Same preprocessor and feature filter as Stage 4 applied to synthetic data before prediction.

| Model | Dataset | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|----|
| logreg | original_test | 0.777 | 0.365 | 0.768 | 0.495 |
| logreg | synthetic | 0.536 | 0.167 | 0.548 | 0.256 |
| knn | original_test | 0.897 | 0.795 | 0.368 | 0.504 |
| knn | synthetic | 0.766 | 0.156 | 0.137 | 0.146 |
| rf | original_test | 0.960 | 0.915 | 0.789 | 0.848 |
| rf | synthetic | 0.722 | 0.177 | 0.247 | 0.206 |

**Performance drops** because CTGAN preserves marginal distributions but not the joint P(Churn | features). Key feature means shifted upward (+9% day minutes, +12% service calls, +66% vmail messages), pushing samples across the classifiers' learned decision boundaries. Despite this, the synthetic data correctly reproduces the class imbalance (14.6% churn) and broad categorical proportions.

---

## 4. Output Files

### Root
| File | Description |
|------|-------------|
| `churn-gen-ai-test-data.csv` | 500 synthetic rows, same 20-column schema as originals |

### `artifacts/genai/`
| File | Description |
|------|-------------|
| `genai_report.md` | This file |
| `model_interpretation.md` | Stage 6: feature importance, retention strategies, Gen-AI in data mining |
| `model_evaluation_comparison.csv` | Accuracy / precision / recall / F1 for all 3 models × 2 datasets |
| `feature_stats_comparison.csv` | Mean / std / min / max per feature across train, test, synthetic |

### `artifacts/genai/histograms/`
| File | Description |
|------|-------------|
| `histogram_comparison_DEGHI.png` | 5×2 panel: cols D/E/G/H/I, synthetic (left) vs original test (right) |
| `hist_international_plan.png` | Col D: grouped bar chart, Yes/No proportions |
| `hist_voice_mail_plan.png` | Col E: grouped bar chart, Yes/No proportions |
| `hist_total_day_minutes.png` | Col G: overlaid density histograms |
| `hist_total_day_calls.png` | Col H: overlaid density histograms |
| `hist_total_day_charge.png` | Col I: overlaid density histograms |

### `artifacts/genai/confusion_matrices/`
| File | Description |
|------|-------------|
| `cm_{model}_original_test.png` | Confusion matrix on `churn-bigml-20.csv` (logreg / knn / rf) |
| `cm_{model}_synthetic.png` | Confusion matrix on `churn-gen-ai-test-data.csv` |

### `artifacts/genai/interpretation/`
| File | Description |
|------|-------------|
| `rf_feature_importances.png` | Horizontal bar chart, RF Gini importance (15 features) |
| `rf_feature_importances.csv` | feature / display_name / importance, sorted descending |
| `logreg_coef_magnitudes.png` | Horizontal bar chart, \|LogReg coefficient\| (15 features) |
| `logreg_coefficients.csv` | feature / display_name / abs_coef / coef, sorted descending |

---

## 5. Reproduce

```bash
python run_preprocess_eda.py          # regenerate preprocessor (once)
rm churn-gen-ai-test-data.csv         # optional: force CTGAN retrain (~5 min)
python run_genai.py
```
