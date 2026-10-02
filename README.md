# Telecom Customer Churn Prediction & Retention ROI

### 🔗 [Live Demo](https://lavish-churn-predictor.streamlit.app/)

![Churn prediction with SHAP explanation](screenshots/prediction.png)
![AI-generated retention plan](screenshots/ai_plan.png)


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

## Scope

- **Phase 1 (complete):** churn model, evaluation, SHAP explainability, retention-ROI ranking, Streamlit demo.
- **Phase 2A (complete):** LLM retention-action generator — Gemini turns the model's SHAP risk factors into a written retention plan with a specific recommended offer.
- **Phase 2B (not implemented):** RAG offer matcher — see Future scope.

## Phase 2A — LLM Retention-Action Generator ✅

Built on top of the Phase 1 model (XGBoost still does all prediction; the LLM only explains and recommends).

- Takes the customer's churn probability + SHAP risk factors → generates a structured retention plan
- Prompt engineered with explicit role, context, constraints, and output format
- Anti-hallucination constraint: the model may only reason from the provided risk factors
- Model-fallback chain with exponential backoff across four Gemini Flash models for reliability
- API key stored in `.env` (git-ignored), never committed

## Future scope (not implemented)

- **RAG offer matcher** — embed a real offer catalogue in a vector store, retrieve the best-fit offers, and have the LLM select from them instead of generating offers freely.
- Structured JSON output from the LLM.
- Uplift modeling / contextual bandit for offer selection.
- Decision-threshold tuning.
- Deployment as a Spring Boot API + Python model microservice.