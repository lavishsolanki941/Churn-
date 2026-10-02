import os, time
import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv
from google import genai
from google.genai import errors

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")

load_dotenv()

# --- Gemini setup ---
@st.cache_resource
def get_gemini_client():
    # Works locally (.env) AND on Streamlit Cloud (st.secrets)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        key = st.secrets.get("GEMINI_API_KEY")
    return genai.Client(api_key=key)
gemini = get_gemini_client()

MODEL_FALLBACKS = ["gemini-3.5-flash", "gemini-3.6-flash",
                   "gemini-3.7-flash", "gemini-flash-lite-latest"]

def ask_gemini(prompt, retries=2):
    """Try each model in turn with backoff until one responds."""
    for model in MODEL_FALLBACKS:
        for attempt in range(retries):
            try:
                return gemini.models.generate_content(model=model, contents=prompt).text
            except errors.ServerError:
                time.sleep(2 ** attempt)
    return "(All Gemini models are busy right now — try again in a few minutes.)"

def build_retention_prompt(churn_prob, tenure, monthly, contract, reasons):
    reasons_text = "\n".join(f"- {r}" for r in reasons)
    return f"""You are a senior customer-retention strategist at a telecom company.

CONTEXT — a churn model flagged this customer:
- Churn probability: {churn_prob:.0%}
- Tenure: {tenure} months
- Monthly bill: ₹{monthly:.0f}
- Contract: {contract}
- Key risk factors identified by the model (SHAP):
{reasons_text}

TASK:
Write a short retention plan for this specific customer.

CONSTRAINTS:
- Base your reasoning ONLY on the risk factors above. Do not invent data.
- Suggest ONE concrete retention offer that directly addresses the top risk factor.
- Keep it under 120 words. Be practical, not generic.

OUTPUT FORMAT:
Risk summary: <one sentence>
Recommended offer: <one specific offer>
Why it works: <one sentence tying it to the risk factors>
"""

# --- Load the saved model, preprocessor, and data ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    df = pd.read_csv("telco.csv")
    return model, preprocessor, df

model, preprocessor, df = load_artifacts()


st.title("📡 Telecom Churn Predictor")
st.caption("Enter a customer's details to get their churn risk, reasons, and a retention recommendation.")

# --- Input form: the few features that matter most (from SHAP) ---
st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    monthly = st.slider("Monthly charges (₹)", 0, 200, 90)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
with c2:
    tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    payment = st.selectbox("Payment method",
                           ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])

# --- Build a full customer row: use dataset defaults, override with the inputs above ---
feature_cols = [c for c in df.columns if c not in ["customerID", "Churn"]]
row = {col: df[col].mode()[0] for col in feature_cols}   # sensible defaults for untouched fields
row.update({
    "tenure": tenure, "MonthlyCharges": monthly,
    "TotalCharges": monthly * max(tenure, 1),
    "Contract": contract, "InternetService": internet,
    "TechSupport": tech, "OnlineSecurity": security,
    "PaymentMethod": payment, "SeniorCitizen": 1 if senior == "Yes" else 0,
})
X_one = pd.DataFrame([row])[feature_cols]

# --- Predict ---
want_plan = st.checkbox("🤖 Also generate an AI retention plan (Gemini)")

if st.button("Predict churn risk", type="primary"):
    X_t = preprocessor.transform(X_one)
    prob = float(model.predict_proba(X_t)[:, 1][0])     

    st.metric("Churn probability", f"{prob:.0%}")
        # 0.35 = business-optimal threshold from cost-based tuning (not the default 0.5)
    if prob >= 0.35:
        st.error("HIGH risk — prioritise for retention.")
    elif prob >= 0.20:
        st.warning("MEDIUM risk — worth watching.")
    else:
        st.success("LOW risk — no action needed.")

    # Retention recommendation (mirrors Phase 6 logic)
    value = monthly * 12
    st.write(f"**12-month value:** ₹{value:,.0f}  ·  **Priority score:** {prob * value:,.0f}")
    if prob >= 0.6 and value > df["MonthlyCharges"].median() * 12:
        st.info("💡 High-value AND high-risk → strong retention-offer candidate.")




    # Top reasons via SHAP
    # --- Explain the prediction in plain English ---

    raises, lowers = [], []
    try:
        import shap
        names = preprocessor.get_feature_names_out()
        sv = shap.TreeExplainer(model).shap_values(X_t)
        contrib = pd.Series(sv[0], index=names)

        ALLOWED = {"tenure", "MonthlyCharges", "TotalCharges", "Contract",
                   "InternetService", "TechSupport", "OnlineSecurity",
                   "PaymentMethod", "SeniorCitizen"}

        def base_col(feat):
            if feat.startswith("num__"): return feat[5:]
            if feat.startswith("cat__"): return feat[5:].split("_", 1)[0]
            return feat

        row_vals = pd.Series(
            X_t.toarray()[0] if hasattr(X_t, "toarray") else X_t[0], index=names
        )

        def keep(f):
            if base_col(f) not in ALLOWED:
                return False
            if f.startswith("cat__"):
                return row_vals[f] == 1
            return True

        contrib = contrib[[f for f in contrib.index if keep(f)]]

        def humanize(feat):
            f = feat.replace("num__", "").replace("cat__", "")
            tenure_val = int(X_one["tenure"].iloc[0])
            bill_val = X_one["MonthlyCharges"].iloc[0]
            mapping = {
                "tenure": (f"Newer customer — only {tenure_val} months with us"
                           if tenure_val < 24 else
                           f"Loyal customer — {tenure_val} months with us"),
                "MonthlyCharges": f"Monthly bill of ₹{bill_val:.0f}"
                                  + (" (on the higher side)" if bill_val > 70 else " (on the lower side)"),
                "TotalCharges": "Total amount billed so far",
                "SeniorCitizen": "Senior citizen",
                "Contract_Month-to-month": "On a flexible month-to-month plan (easy to leave)",
                "Contract_One year": "Locked into a one-year contract",
                "Contract_Two year": "Locked into a two-year contract",
                "InternetService_Fiber optic": "Uses fiber-optic internet",
                "InternetService_DSL": "Uses DSL internet",
                "InternetService_No": "Has no internet service",
                "TechSupport_No": "No tech-support add-on",
                "TechSupport_Yes": "Has a tech-support add-on",
                "OnlineSecurity_No": "No online-security add-on",
                "OnlineSecurity_Yes": "Has an online-security add-on",
                "PaymentMethod_Electronic check": "Pays by electronic check",
                "PaymentMethod_Mailed check": "Pays by mailed check",
                "PaymentMethod_Bank transfer (automatic)": "Pays by automatic bank transfer",
                "PaymentMethod_Credit card (automatic)": "Pays by automatic credit card",
            }
            return mapping.get(f, f.replace("_", ": "))

        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
        raises = [humanize(f) for f, v in top.items() if v > 0]
        lowers = [humanize(f) for f, v in top.items() if v < 0]

        st.subheader("Why this prediction?")
        top_feat, top_val = top.index[0], top.iloc[0]
        direction = "increasing" if top_val > 0 else "reducing"
        st.info(f"The biggest factor is **{humanize(top_feat).lower()}**, {direction} their churn risk.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔺 Raising churn risk**")
            for r in raises: st.markdown(f"- {r}")
            if not raises: st.caption("None significant")
        with col_b:
            st.markdown("**🟢 Helping retention**")
            for l in lowers: st.markdown(f"- {l}")
            if not lowers: st.caption("None significant")
    except Exception as e:
        st.caption(f"(Explanation unavailable: {e})")

    if want_plan:
        if not raises:
            st.warning("No risk factors available — skipping AI plan.")
        else:
            st.subheader("🤖 AI-Generated Retention Plan")
            with st.spinner("Generating plan with Gemini..."):
                prompt = build_retention_prompt(prob, tenure, monthly, contract, raises)
                plan = ask_gemini(prompt)
            st.markdown(plan)
            st.caption("Generated by Gemini from the model's SHAP risk factors — not a prediction.")






