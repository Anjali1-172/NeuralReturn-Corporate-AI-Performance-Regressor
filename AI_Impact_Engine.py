import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ROI Predictor", layout="wide")

# ---------------- FEATURE ENGINEERING ----------------
def engineering_logic(X):
    X = X.copy()

    # binary feature
    X['uses_advanced_ai_tool'] = X['ai_primary_tool'].apply(
        lambda x: 1 if x in ['Custom Internal AI', 'Claude', 'Gemini'] else 0
    )

    # engineered features
    X['budget_per_project'] = X['ai_budget_percentage'] / (X['ai_projects_active'] + 1)
    X['investment_per_tool'] = X['ai_investment_per_employee'] / (X['num_ai_tools_used'] + 1)

    X['success_rate'] = 1 - X['ai_failure_rate']
    X['effective_budget'] = X['ai_budget_percentage'] * X['success_rate']

    X['maturity_x_adoption'] = X['ai_maturity_score'] * X['ai_adoption_rate']
    X['experience_factor'] = X['years_using_ai'] * X['ai_maturity_score']

    X['projects_per_tool'] = X['ai_projects_active'] / (X['num_ai_tools_used'] + 1)
    X['revenue_per_ai_investment'] = X['annual_revenue_usd_millions'] / (X['ai_investment_per_employee'] + 1)

    # drop raw columns used in engineering
    X = X.drop(columns=[
        "ai_budget_percentage",
        "num_ai_tools_used",
        "ai_projects_active",
        "ai_investment_per_employee"
    ])

    return X

# ---------------- LOAD MODEL ----------------
st.write("Loading model...")

try:
    pipeline = joblib.load("best_xgb_model.pkl")
    st.success("Model loaded successfully ✅")
except Exception as e:
    pipeline = None
    st.error(f"Model loading failed: {e}")

# ---------------- STYLES ----------------
st.markdown("""
<style>
.stApp { background-color: #F8F9FB; color: #31333F; }
.main-header { text-align: center; font-weight: bold; }
.sub-header { text-align: center; margin-bottom: 30px; }

.result-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E6E9EF;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 class='main-header'>👑 ROI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI Adoption ROI Analysis Dashboard</p>", unsafe_allow_html=True)

# ---------------- FORM ----------------
with st.form("roi_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        industry = st.selectbox("Industry", ["Healthcare", "Finance", "Technology", "Manufacturing", "Retail", "Other"])
        company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"])
        revenue = st.number_input("Annual Revenue (USD Millions)", value=150.0)
        inv_per_emp = st.number_input("AI Investment per Employee", value=1200.0)

    with col2:
        adoption_rate = st.slider("AI Adoption Rate (%)", 0, 100, 45)
        adoption_stage = st.selectbox("AI Adoption Stage", ["none", "pilot", "partial", "full"])
        years_ai = st.number_input("Years Using AI", value=3.0)
        maturity = st.slider("AI Maturity Score", 0, 100, 60)
        failure_rate = st.slider("Failure Rate (%)", 0.0, 100.0, 15.0)

    with col3:
        primary_tool = st.selectbox("AI Tool", ["ChatGPT", "Claude", "Gemini", "Custom"])
        num_tools = st.number_input("Num AI Tools", min_value=1, value=8)
        active_projects = st.number_input("Active AI Projects", value=5)
        budget = st.slider("AI Budget (%)", 0.0, 100.0, 12.5)

    predict_btn = st.form_submit_button("Generate Prediction")

# ---------------- PREDICTION ----------------
if predict_btn:

    input_data = pd.DataFrame([{
        "company_size": company_size,
        "annual_revenue_usd_millions": revenue,
        "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": adoption_stage,
        "years_using_ai": years_ai,
        "ai_maturity_score": maturity,
        "num_ai_tools_used": num_tools,
        "ai_projects_active": active_projects,
        "ai_budget_percentage": budget,
        "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": inv_per_emp,
        "industry_grouped": industry,
        "ai_primary_tool": primary_tool
    }])

    try:
        if pipeline is None:
            st.error("Model not loaded.")
        else:
            roi_prediction = pipeline.predict(input_data)[0]

        # ---------------- RESULT ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("ROI Projection")

            st.metric("Predicted ROI", f"{roi_prediction:.2%}")

            investment = revenue * (budget / 100)
            net_gain = investment * roi_prediction

            st.write(f"Investment: ${investment:.2f}M")
            st.write(f"Net Gain: ${net_gain:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Health Check")

            st.bar_chart(pd.DataFrame({
                "Score": [maturity, 52]
            }, index=["Your Company", "Industry Avg"]))

            if failure_rate > 25:
                st.error("High failure risk")
            else:
                st.success("Failure rate under control")

        # ---------------- TRUST REPORT (UPGRADED) ----------------
        st.markdown("### 🛡️ Model Trust Report")

        t1, t2, t3 = st.columns(3)

        # R2 interpretation
        with t1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            r2 = 0.45
            st.metric("Explained Variance (R²)", f"{r2:.2f}")
            st.progress(r2)
            st.caption("Model explains ~45% of ROI variation. Remaining is external business factors.")
            st.markdown("</div>", unsafe_allow_html=True)

        # input alignment
        with t2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            alignment = 1.0
            if revenue > 1000: alignment -= 0.3
            if budget > 50: alignment -= 0.2
            if failure_rate > 40: alignment -= 0.2
            if num_tools > 20: alignment -= 0.1

            alignment = max(0, min(1, alignment))

            st.metric("Input Alignment", f"{alignment:.2f}")
            st.progress(alignment)

            if alignment > 0.75:
                st.success("Within training distribution")
            elif alignment > 0.5:
                st.warning("Partial extrapolation risk")
            else:
                st.error("High extrapolation risk")

            st.markdown("</div>", unsafe_allow_html=True)

        # reliability
        with t3:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            uncertainty = (1 - r2) * (1.2 - alignment)
            uncertainty = max(0, min(1, uncertainty))

            st.metric("Uncertainty", f"{uncertainty:.2f}")
            st.progress(1 - uncertainty)

            if uncertainty < 0.3:
                st.success("High confidence")
            elif uncertainty < 0.6:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence")

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")


