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
        
    with col2:
        adoption_rate = st.slider("AI Adoption Rate (%)", 0, 100, 45)
        adoption_stage = st.selectbox("AI Adoption Stage", ["none", "pilot", "partial", "full"])
        years_ai = st.number_input("Years Using AI", min_value=0.0, value=3.0)
        maturity = st.slider("AI Maturity Score (0-100)", 0, 100, 60)

    with col3:
        primary_tool = st.selectbox("AI Primary Tool", ["ChatGPT", "Claude", "Gemini", "Custom"])
        num_tools = st.number_input("Num AI Tools Used", min_value=1, value=8)
        # --- FIXED: ADDED MISSING COLUMN BELOW ---
        active_projects = st.number_input("AI Projects Active", min_value=0, value=5)
        budget = st.slider("AI Budget Percentage (%)", 0.0, 100.0, 12.5)
        failure_rate = st.slider("AI Project Failure Rate (%)", 0.0, 100.0, 15.0)
        inv_per_emp = st.number_input("AI Investment Per Employee (USD)", value=1200)

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
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Calculated ROI Projection")
            st.metric("Predicted Corporate ROI", f"{roi_prediction:.2%}")
            
            # Financial breakdown
            total_cost = revenue * (budget / 100)
            net_gain = total_cost * roi_prediction
            st.write(f"**Estimated AI Investment:** ${total_cost:.2f}M")
            st.write(f"**Predicted Net Gain:** ${net_gain:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
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
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Current columns being sent to model:", list(input_df.columns))

''''
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
    """, unsafe_allow_html=True)

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


'''
