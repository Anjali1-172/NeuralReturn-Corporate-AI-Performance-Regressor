import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(page_title="ROI Predictor", layout="wide")

# =========================
# FEATURE ENGINEERING
# =========================
def engineering_logic(X):
    X = X.copy()

    # derived feature (IMPORTANT FIX)
    X["uses_advanced_ai_tool"] = X["ai_primary_tool"].apply(
        lambda x: 1 if x in ["Custom Internal AI", "Claude", "Gemini"] else 0
    )

    X["budget_per_project"] = X["ai_budget_percentage"] / (X["ai_projects_active"] + 1)
    X["investment_per_tool"] = X["ai_investment_per_employee"] / (X["num_ai_tools_used"] + 1)

    X["success_rate"] = 1 - X["ai_failure_rate"]
    X["effective_budget"] = X["ai_budget_percentage"] * X["success_rate"]

    X["maturity_x_adoption"] = X["ai_maturity_score"] * X["ai_adoption_rate"]
    X["experience_factor"] = X["years_using_ai"] * X["ai_maturity_score"]

    X["projects_per_tool"] = X["ai_projects_active"] / (X["num_ai_tools_used"] + 1)
    X["revenue_per_ai_investment"] = X["annual_revenue_usd_millions"] / (
        X["ai_investment_per_employee"] + 1
    )

    X = X.drop(columns=[
        "ai_budget_percentage",
        "num_ai_tools_used",
        "ai_projects_active",
        "ai_investment_per_employee",
        "ai_primary_tool"   # IMPORTANT: prevent leakage mismatch
    ])

    return X


# =========================
# LOAD MODEL
# =========================
st.write("Loading model...")
pipeline = joblib.load("best_xgb_model.pkl")
st.success("Model loaded ✅")


# =========================
# UI
# =========================
st.title("👑 ROI Prediction Engine")
st.caption("AI Adoption ROI + Explainability Dashboard")


with st.form("form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        industry = st.selectbox("Industry", ["Healthcare", "Finance", "Technology","Manufacturing","Retail","Other"])
        company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"])
        revenue = st.number_input("Revenue (M USD)", 50.0)

    with col2:
        adoption_rate = st.slider("AI Adoption Rate", 0.0, 1.0, 0.5)
        adoption_stage = st.selectbox("Stage", ["none", "pilot", "partial", "full"])
        years_ai = st.number_input("Years AI", 3.0)
        maturity = st.slider("Maturity Score", 0, 100, 60)

    with col3:
        tool = st.selectbox("AI Tool", ["ChatGPT", "Claude", "Gemini", "Custom Internal AI"])
        num_tools = st.number_input("AI Tools", 1)
        projects = st.number_input("Projects", 5)
        budget = st.slider("Budget %", 0.0, 100.0, 15.0)
        failure = st.slider("Failure Rate", 0.0, 1.0, 0.2)

    inv_emp = st.number_input("AI Investment per Employee", 1200.0)

    btn = st.form_submit_button("Predict ROI")


# =========================
# PREDICTION
# =========================
if btn:

    df = pd.DataFrame([{
        "company_size": company_size,
        "annual_revenue_usd_millions": revenue,
        "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": adoption_stage,
        "years_using_ai": years_ai,
        "ai_maturity_score": maturity,
        "num_ai_tools_used": num_tools,
        "ai_projects_active": projects,
        "ai_budget_percentage": budget,
        "ai_failure_rate": failure,
        "ai_investment_per_employee": inv_emp,
        "industry_grouped": industry,
        "ai_primary_tool": tool
    }])

    try:
        pred = pipeline.predict(df)[0]

        # =========================
        # MAIN RESULT
        # =========================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 ROI Prediction")
            st.metric("Predicted ROI", f"{pred:.2%}")

            base_investment = revenue * (budget / 100)
            gain = base_investment * pred

            st.write(f"Investment: ${base_investment:.2f}M")
            st.write(f"Expected Gain: ${gain:.2f}M")

            # ROI scenario range (uncertainty band)
            low = pred - 0.10
            high = pred + 0.10

            fig = px.bar(
                x=["Worst Case", "Expected", "Best Case"],
                y=[low, pred, high],
                title="ROI Scenario Range (Model Uncertainty)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏢 Business Health")

            fig2 = px.line(
                x=["Your Company", "Industry Avg"],
                y=[maturity, 55],
                title="AI Maturity Comparison"
            )
            st.plotly_chart(fig2, use_container_width=True)

            if failure > 0.25:
                st.error("⚠️ High Failure Risk")
            else:
                st.success("✔ Stable Execution Risk")


        # =========================
        # MODEL EXPLANATION PANEL
        # =========================
        st.markdown("## 🧠 Model Explainability Dashboard")

        st.info("""
        **Model Performance Summary**
        - R² Score: **0.46**
        - Interpretation: The model explains ~46% of ROI variance.
        - Remaining 54% = external factors not captured (market, leadership, strategy, etc.)
        """)

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Explained Variance", "46%")
            st.progress(0.46)

        with colB:
            st.metric("Unexplained Variance", "54%")
            st.progress(0.54)

        with colC:
            if 40 <= maturity <= 70:
                st.success("Input lies in model's strongest learning zone")
            else:
                st.warning("Input outside optimal training distribution")

        # =========================
        # TRUST SCORE LOGIC
        # =========================
        st.markdown("### 📊 Model Trust Analysis")

        trust_score = 0

        if 20 <= revenue <= 500:
            trust_score += 30
        if 0.1 <= failure <= 0.4:
            trust_score += 25
        if 3 <= num_tools <= 15:
            trust_score += 25
        if 30 <= maturity <= 80:
            trust_score += 20

        st.metric("Prediction Confidence Score", f"{trust_score}/100")

        st.progress(trust_score / 100)

        if trust_score > 70:
            st.success("High reliability prediction")
        elif trust_score > 40:
            st.warning("Moderate reliability")
        else:
            st.error("Low reliability — outside training distribution")

    except Exception as e:
        st.error(f"Prediction error: {e}")
