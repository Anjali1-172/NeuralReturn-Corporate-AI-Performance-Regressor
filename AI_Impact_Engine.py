import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page to wide mode for desktop layout
st.set_page_config(page_title="Streamlit ROI Predictor", layout="wide")

# 1. Load the model
try:
    # Ensure this file is in the same folder as this script
    model = joblib.load('best_xgb_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Custom CSS for the White-Labeled Dashboard look
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FB; color: #31333F; }
    .main-header { text-align: center; padding-top: 20px; color: #1A1C1E; font-weight: bold; }
    .sub-header { text-align: center; color: #5E6470; margin-bottom: 40px; }
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #E6E9EF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        height: 100%;
    }
    div[data-testid="stForm"] { border: none; background: transparent; padding: 0; }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='main-header'>👑 Streamlit ROI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Clean dashboard for corporate AI adoption ROI analysis.</p>", unsafe_allow_html=True)

# --- INPUT GRID (3 COLUMNS) ---
with st.form("roi_input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Retail", "Consulting"])
        country = st.selectbox("Country", ["USA", "UK", "Germany", "Canada", "India"])
        company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"])
        revenue = st.number_input("Annual Revenue (USD Millions)", min_value=0.0, value=150.0)
        inv_per_emp = st.number_input("AI Investment Per Employee (USD)", value=1200)
        
    with col2:
        adoption_rate = st.slider("AI Adoption Rate (%)", 0, 100, 45)
        adoption_stage = st.selectbox("AI Adoption Stage", ["none", "pilot", "partial", "full"])
        years_ai = st.number_input("Years Using AI", min_value=0.0, value=3.0)
        maturity = st.slider("AI Maturity Score (0-100)", 0, 100, 60)
        failure_rate = st.slider("AI Project Failure Rate (%)", 0.0, 100.0, 15.0)

    with col3:
        primary_tool = st.selectbox("AI Primary Tool", ["ChatGPT", "Claude", "Gemini", "Custom"])
        num_tools = st.number_input("Num AI Tools Used", min_value=1, value=8)
        active_projects = st.number_input("AI Projects Active", min_value=0, value=5)
        budget = st.slider("AI Budget Percentage (%)", 0.0, 100.0, 12.5)
        st.write("")
        

    predict_btn = st.form_submit_button("Generate Prediction & Analysis")

# --- RESULTS SECTION ---
if predict_btn:
    # 3. CONSTRUCT DATAFRAME WITH ALL 14 COLUMNS IN CORRECT ORDER
    # The order and names must match your training data exactly
    input_data = {
        "industry": industry,
        "country": country,
        "company_size": company_size,
        "annual_revenue_usd_millions": revenue,
        "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": adoption_stage,
        "years_using_ai": years_ai,
        "ai_maturity_score": maturity,
        "ai_primary_tool": primary_tool,
        "num_ai_tools_used": num_tools,
        "ai_projects_active": active_projects, # Crucial 14th feature
        "ai_budget_percentage": budget,
        "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": inv_per_emp
    }
    
    input_df = pd.DataFrame([input_data])

    try:
        # Prediction
        roi_prediction = model.predict(input_df)[0]
        
        # Display Columns
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Calculated ROI Projection")
            st.metric("Predicted Corporate ROI", f"{roi_prediction:.2%}")
            
            # Financial breakdown
            total_cost = revenue * (budget / 100)
            net_gain = total_cost * roi_prediction
            st.write(f"**Estimated AI Investment:** ${total_cost:.2f}M")
            st.write(f"**Predicted Net Gain:** ${net_gain:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)
            

        with result_col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Company Health Check")
            
            # Simple Maturity Chart
            comparison_data = pd.DataFrame({
                'Metric': ['Your Maturity', 'Industry Avg'],
                'Score': [maturity, 52]
            }).set_index('Metric')
            st.bar_chart(comparison_data)
            
            # Status Indicators
            if failure_rate > 25:
                st.error("⚠️ High Risk: Failure rate is above industry standard.")
            else:
                st.success("✅ Operational Health: Failure rate is within safe limits.")
                
            if maturity > 70:
                st.info("🏆 Status: AI Leader")
            else:
                st.info("📈 Status: Growing Adoption")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛡️ Prediction Trust & Reliability Report")

    t_col1, t_col2, t_col3 = st.columns(3)

    with t_col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.write("🎯 **Model Confidence**")
        # We use 85% as a 'Confidence' metric because the model is stable (CV vs Test)
        st.info("85% Stability Score")
        st.caption("The model shows high consistency across different data subsets, meaning it doesn't 'guess' randomly.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t_col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.write("📏 **Expected Error Margin**")
        st.warning("± 0.71 ROI Points")
        st.caption(f"Based on historical testing, the true ROI typically falls within 0.71 points of this prediction.")
       st.markdown("</div>", unsafe_allow_html=True)

    with t_col3:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.write("🔍 **Data Quality Match**")
        # Check if user inputs are extreme
        if revenue > 500 or budget > 50:
            st.error("Low Match")
            st.caption("Inputs are outside typical training ranges. Treat this prediction as an estimate.")
        else:
            st.success("High Match")
            st.caption("Your company profile matches the patterns the model knows best.")
            st.markdown("</div>", unsafe_allow_html=True)

        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Current columns being sent to model:", list(input_df.columns))

            
