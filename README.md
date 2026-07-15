# Capstone Three: Predicting Customer Conversion in Digital Marketing

## Problem Statement

Which customers are most likely to convert from a digital marketing campaign, and which demographic, behavioral, and campaign-level factors most strongly drive that conversion? The goal is a model that, given a customer's profile and the campaign targeting them, predicts conversion and surfaces the levers marketers can actually pull.

Marketing budgets face constant ROI scrutiny, and a large share of spend is wasted on customers who were never likely to convert. A reliable conversion model lets teams concentrate spend on high-probability segments and justify targeting decisions with evidence rather than intuition.

## Data

- **Source:** [Kaggle — Predict Conversion in Digital Marketing](https://www.kaggle.com) (Rabie El Kharoua)
- **Size:** 8,000 customer records, 20 columns (demographics, campaign attributes, engagement metrics, loyalty indicators)
- **Target:** `Conversion` (binary) — ~88% positive class, a significant imbalance
- **Notes:** No nulls or duplicates. `AdvertisingPlatform` and `AdvertisingTool` were fully redacted and dropped. The data is synthetic, and some behavioral fields show internal inconsistencies (e.g., 0 website visits paired with nonzero pages-per-visit) that wouldn't occur in real traffic data.

## Repository Structure

```
Capstone-Three/
├── project_proposal/       # Initial project proposal PDF
├── Data Wrangling/         # Cleaning, validation, and profiling
│   └── data_wrangling.ipynb
├── EDA/                    # Exploratory analysis, correlations, baseline model
│   └── EDA.ipynb
├── Modeling/                # Feature engineering, imbalance handling, final models
│   └── Modeling.ipynb
├── model_metrics.csv       # Features, parameters, hyperparameters & performance for every model
├── Capstone_Final_Report.pdf
└── Capstone_Three_Slide_Deck.pptx
```

## Approach

1. **Data Wrangling** — cleaned and validated the raw dataset; dropped uninformative/redacted columns; profiled distributions.
2. **EDA** — found weak collinearity between most features and the target; ran a baseline logistic regression (ROC-AUC 0.782) to confirm a predictive signal existed. `ClickThroughRate`, `TimeOnSite`, and `EmailClicks` emerged as the strongest predictors.
3. **Feature Engineering** — added `EngagementScore`, `ClickToOpenRatio`, `EngagementPerVisit`, `AdSpendPerClick`, and a binned `AgeGroup`.
4. **Class Imbalance** — compared `class_weight='balanced'` against SMOTE oversampling; results were nearly identical (ROC-AUC 0.783 vs. 0.780). Threshold tuning (0.70, from the PR-curve crossover) had a larger effect, lifting accuracy from 0.76 to 0.86.
5. **Modeling** — compared logistic regression, Gradient Boosting, and an RBF-kernel SVM, all trained on SMOTE-balanced data.

## Results

| Model | ROC-AUC | Accuracy | Class-0 F1 | Class-1 F1 |
|---|---|---|---|---|
| Log Reg (class-weighted) | 0.783 | 0.75 | 0.42 | 0.84 |
| Log Reg (SMOTE) | 0.780 | 0.76 | 0.43 | 0.85 |
| Log Reg (SMOTE, tuned threshold) | — | 0.86 | 0.42 | 0.92 |
| **Gradient Boosting (SMOTE)** | **0.789** | **0.89** | **0.52** | **0.94** |
| SVM — RBF (SMOTE) | 0.726 | 0.83 | 0.40 | 0.90 |

**Gradient Boosting** was selected as the final model — it led on every key metric, particularly class-0 F1, the harder and more valuable non-converter class to identify. That said, the gain over the class-weighted logistic regression baseline was modest (~0.006 ROC-AUC), a limitation this project documents honestly rather than overstating.

Full metrics for every model — features used, class-imbalance handling, hyperparameters, and precision/recall by class — are in [`model_metrics.csv`](./model_metrics.csv).

## Key Takeaway

Early engagement behavior — click-through rate, time on site, email interaction, prior purchases — predicts conversion far better than demographics or campaign channel. Targeting and budget decisions should weight observed engagement more heavily than audience segmentation alone.

## Limitations & Future Work

- Synthetic data; generalization to live campaign data is unproven.
- `ConversionRate` and `ClickThroughRate` carry some leakage risk as near-deterministic proxies for the target.
- No timeframe field, limiting temporal analysis (e.g., campaign fatigue).
- Future iterations could try XGBoost/LightGBM with a proper hyperparameter search, SHAP-based feature attribution, and cost-sensitive thresholds tied to real marketing spend.
