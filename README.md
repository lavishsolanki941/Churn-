# Telecom Customer Churn Prediction & Retention ROI

Predicts which telecom customers will churn, explains *why* (SHAP), and ranks
which high-value customers to retain — with an ROI estimate for the campaign.

## Problem
Retaining a customer is cheaper than acquiring one. This project flags at-risk
customers *before* they leave and prioritizes retention spend by revenue at risk.

## Dataset
Telco Customer Churn (7,043 customers, 26.5% churn — imbalanced).

## Approach
1. **EDA** — churn driven by contract type, tenure, monthly charges
2. **Preprocessing** — cleaned TotalCharges, one-hot encoding, scaling, stratified split (no leakage)
3. **Models** — Logistic Regression, Random Forest, XGBoost compared via 5-fold CV
4. **Tuning** — GridSearchCV on XGBoost (best model)
5. **Explainability** — SHAP (global + per-customer)
6. **Retention ROI** — rank by churn_prob × customer value, simulate campaign ROI

## Results
| Model | ROC-AUC | Recall (Churn) |
|-------|---------|----------------|
| Logistic Regression | 0.842 | 0.78 |
| Random Forest | 0.820 | 0.62 |
| **XGBoost (tuned)** | **0.845** | **0.79** |

- Evaluated on **recall / ROC-AUC** (not accuracy) due to class imbalance
- Top churn drivers: month-to-month contract, low tenure, high charges
- Retention ROI: targeting top 500 at-risk high-value customers → ~5x return (on stated assumptions)

## Tech
Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib/seaborn

## Files
- `churn_prediction.ipynb` — full analysis, phase by phase
- `at_risk_customers.csv` — ranked retention list (output)
- `churn_model.joblib`, `preprocessor.joblib` — saved model + preprocessor

## How to run
```
pip install -r requirements.txt
jupyter lab   # open churn_prediction.ipynb and run all
```
