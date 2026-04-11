import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your pipeline
try:
    model = joblib.load('best_xgb_model.pkl')
except:
    st.error("Model file not found. Please ensure 'best_xgb_model.pkl' is in the directory.")

# Page Configuration
st.set_page_config(page_title="AI ROI Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #000000 25%, #1a0033 100%);
        color: #00ffcc;
    }
    .main {
        background: transparent;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.input-container) {
        background: rgba(0, 0, 0, 0.8);
        border: 2px solid #ff00ff;
        box-shadow: 0 0 15px #ff00ff, 0 0 5px #00ffff;
        padding: 30px;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
    }
    h1, h2, h3, p, label {
        color: #00ffcc !important;
        text-shadow: 0 0 5px #00ffcc;
    }
    .stButton>button {
        background-color: #ff00ff;
        color: white;
        border: none;
        box-shadow: 0 0 10px #ff00ff;
        width: 100%;
    }
    </style>
    """, unsafe_allow_stdio=True)

# App Content
st.title("🚀 AI ROI Predictor")
st.markdown("Enter your company's data below to predict the Return on Investment (ROI) of your AI initiatives.")

# Use columns to center the input area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.subheader("Company Profile")
        
        industry = st.selectbox("Industry", ["Consulting", "Healthcare", "Retail", "Technology", "Finance"], 
                               help="The primary sector your company operates in.")
        
        country = st.selectbox("Country", ["USA", "Japan", "Kenya", "Netherlands", "New Zealand", "Other"],
                              help="The headquarters or primary operating region.")
        
        company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"],
                                   help="Categorization based on employee count and revenue.")
        
        st.subheader("AI Strategy")
        
        ai_adoption_stage = st.selectbox("AI Adoption Stage", ["none", "pilot", "partial", "full"],
                                        help="How integrated AI is within your current workflows.")
        
        ai_primary_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Claude", "Gemini", "GitHub Copilot", "Custom Internal AI"],
                                      help="The main AI software or platform used by the company.")
        
        # Numerical Inputs
        revenue = st.number_input("Annual Revenue (USD Millions)", min_value=0.0, step=0.1,
                                 help="Total yearly revenue in millions of dollars.")
        
        adoption_rate = st.slider("AI Adoption Rate (%)", 0, 100, 50,
                                 help="Percentage of business processes currently involving AI.")
        
        years_ai = st.number_input("Years Using AI", min_value=0.0, step=0.5,
                                  help="Number of years since the company first started using AI tools.")
        
        maturity = st.slider("AI Maturity Score", 0, 100, 50,
                            help="A self-assessed score of how sophisticated your AI infrastructure is.")
        
        num_tools = st.number_input("Number of AI Tools Used", min_value=1, step=1,
                                   help="Count of different AI-powered software applications in use.")
        
        active_projects = st.number_input("Active AI Projects", min_value=0, step=1,
                                        help="Number of AI projects currently in development or production.")
        
        budget = st.slider("AI Budget Percentage (%)", 0.0, 100.0, 5.0,
                          help="Percentage of the total IT budget allocated specifically to AI.")
        
        failure_rate = st.slider("AI Project Failure Rate (%)", 0.0, 100.0, 10.0,
                                help="The percentage of AI projects that do not meet their defined objectives.")
        
        investment_per_emp = st.number_input("AI Investment Per Employee (USD)", min_value=0, step=100,
                                           help="Average dollar amount spent on AI resources per staff member.")

        submit = st.form_submit_button("Predict ROI")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if submit:
    # Prepare input data
    input_dict = {
        "industry": industry,
        "country": country,
        "company_size": company_size,
        "annual_revenue_usd_millions": revenue,
        "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": ai_adoption_stage,
        "years_using_ai": years_ai,
        "ai_maturity_score": maturity,
        "ai_primary_tool": ai_primary_tool,
        "num_ai_tools_used": num_tools,
        "ai_projects_active": active_projects,
        "ai_budget_percentage": budget,
        "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": investment_per_emp
    }
    
    input_df = pd.DataFrame([input_dict])
    
    try:
        prediction = model.predict(input_df)[0]
        st.balloons()
        st.success(f"### Predicted ROI: {prediction:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")