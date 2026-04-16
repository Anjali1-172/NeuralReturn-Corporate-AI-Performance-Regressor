import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="ROI Predictor", layout="wide")

# When using custom transformers inside a pipeline,
# they must be available in the runtime environment during inference, otherwise deserialization fails.
def engineering_logic(X):
    # working on a copy to avoid SettingWithCopy warnings
    X = X.copy()
    
    # Division by zero protection (+ 1)
    X['budget_per_project'] = X['ai_budget_percentage'] / (X['ai_projects_active'] + 1)
    X['investment_per_tool'] = X['ai_investment_per_employee'] / (X['num_ai_tools_used'] + 1)
    
    X['success_rate'] = 1 - X['ai_failure_rate']
    X['effective_budget'] = X['ai_budget_percentage'] * X['success_rate']
    
    X['maturity_x_adoption'] = X['ai_maturity_score'] * X['ai_adoption_rate']
    X['experience_factor'] = X['years_using_ai'] * X['ai_maturity_score']
    
    X['projects_per_tool'] = X['ai_projects_active'] / (X['num_ai_tools_used'] + 1)
    X['revenue_per_ai_investment'] = X['annual_revenue_usd_millions'] / (X['ai_investment_per_employee'] + 1)

    X = X.drop(columns =["ai_budget_percentage","num_ai_tools_used","ai_projects_active","ai_investment_per_employee"])
    
    return X

# Load PIPELINE (not just model)
st.write("Loading model...")

pipeline = None

try:
    pipeline = joblib.load("best_xgb_model.pkl")
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Model loading failed: {e}")

# --- STYLES ---
st.markdown("""
<style>
.stApp { background-color: #F8F9FB; color: #31333F; }
.main-header { text-align: center; padding-top: 20px; font-weight: bold; }
.sub-header { text-align: center; margin-bottom: 40px; }

.result-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E6E9EF;
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    border-radius: 8px;
    height: 3.5em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='main-header'>👑 ROI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI Adoption ROI Analysis Dashboard</p>", unsafe_allow_html=True)

# --- FORM ---
with st.form("roi_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Retail", "Consulting"])
        company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"])
        revenue = st.number_input("Annual Revenue (USD Millions)", value=150.0)
        industry_grouped = st.selectbox("Industry", ["Healthcare", "Finance", "Technology","Manufacturing","Retail","Other"])
        uses_advanced_ai_tool = 1 if ai_primary_tool in ["Custom Internal AI", "Claude", "Gemini"] else 0

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

# --- PREDICTION ---
if predict_btn:

    input_data = pd.DataFrame([{
        "industry": industry,
        "company_size": company_size,
        "annual_revenue_usd_millions": revenue,
        "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": adoption_stage,
        "years_using_ai": years_ai,
        "ai_maturity_score": maturity,
        "ai_primary_tool": primary_tool,
        "num_ai_tools_used": num_tools,
        "ai_projects_active": active_projects,
        "ai_budget_percentage": budget,
        "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": inv_per_emp
    }])

    try:
        # ✅ Use pipeline (handles encoding automatically)
        if pipeline is None:
            st.error("Model not loaded. Cannot predict.")
        else:
            roi_prediction = pipeline.predict(input_data)[0]

        col1, col2 = st.columns(2)

        # --- ROI CARD ---
        with col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("ROI Projection")
            st.metric("Predicted ROI", f"{roi_prediction:.2%}")

            total_cost = revenue * (budget / 100)
            net_gain = total_cost * roi_prediction

            st.write(f"Investment: ${total_cost:.2f}M")
            st.write(f"Net Gain: ${net_gain:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- HEALTH CARD ---
        with col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Health Check")

            chart = pd.DataFrame({
                "Score": [maturity, 52]
            }, index=["Your Company", "Industry Avg"])

            st.bar_chart(chart)

            if failure_rate > 25:
                st.error("High failure risk")
            else:
                st.success("Failure rate under control")

            if maturity > 70:
                st.info("AI Leader")
            else:
                st.info("Scaling Stage")

            st.markdown("</div>", unsafe_allow_html=True)

        # --- TRUST REPORT ---
        st.markdown("### 🛡️ Model Trust Report")
        t1, t2, t3 = st.columns(3)

        with t1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.write("Model Stability")
            st.info("R² ≈ 0.45")
            st.markdown("</div>", unsafe_allow_html=True)

        with t2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.write("Error Range")
            st.warning("±0.71 ROI")
            st.markdown("</div>", unsafe_allow_html=True)

        with t3:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            if revenue > 1000 or budget > 50:
                st.error("Out of training range")
            else:
                st.success("Valid input range")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
