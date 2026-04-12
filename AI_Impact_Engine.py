import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page to wide mode for desktop layout
st.set_page_config(page_title="Streamlit ROI Predictor", layout="wide")

# 1. Load the model
try:
    model = joblib.load('best_xgb_model.pkl')
except:
    st.error("Model file 'best_xgb_model.pkl' not found.")

# 2. Custom CSS to match the image (White background, centered titles, clean cards)
st.markdown("""
    <style>
    /* Background and global font */
    .stApp {
        background-color: #F8F9FB;
        color: #31333F;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding-top: 20px;
        color: #1A1C1E;
    }
    .sub-header {
        text-align: center;
        color: #5E6470;
        margin-bottom: 40px;
    }
    
    /* Card Styling for Results */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E6E9EF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
    }

    /* Input section background */
    [data-testid="stForm"] {
        border: none;
        background-color: transparent;
        padding: 0;
    }
    
    /* Style the Predict button */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='main-header'>👑 Streamlit ROI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>This app calculates your predicted ROI based on corporate AI adoption data. Input your data below to receive detailed projections and analysis.</p>", unsafe_allow_html=True)

# --- INPUT GRID ---
# Organizing inputs in a 3-column grid as seen in your image
with st.form("input_form"):
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        revenue = st.number_input("Annual Revenue (USD Millions)", min_value=0.0, value=100.0)
        maturity = st.slider("AI Maturity Score (0-100)", 0, 100, 50)
        industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Retail", "Consulting"])
        
    with row1_col2:
        num_tools = st.number_input("Num AI Tools Used (Count)", min_value=1, value=5)
        budget = st.slider("AI Budget Percentage (%)", 0.0, 100.0, 15.0)
        company_size = st.selectbox("Company Size", ["Enterprise", "SME", "Startup"])

    with row1_col3:
        years_ai = st.number_input("Years Using AI", min_value=0.0, value=2.0)
        failure_rate = st.slider("AI Project Failure Rate (%)", 0.0, 100.0, 10.0)
        adoption_stage = st.selectbox("AI Adoption Stage", ["full", "partial", "pilot", "none"])

    # Hidden fields needed for your specific model logic
    with st.expander("Additional Parameters"):
        c1, c2, c3 = st.columns(3)
        country = c1.text_input("Country", "USA")
        adoption_rate = c2.slider("AI Adoption Rate (%)", 0, 100, 40)
        inv_per_emp = c3.number_input("AI Investment Per Employee (USD)", value=1000)
        primary_tool = "ChatGPT" 
        active_projects = 3

    predict_pressed = st.form_submit_button("Generate Prediction & Analysis")

# --- RESULTS SECTION ---
if predict_pressed:
    # Prepare DataFrame for model
    input_df = pd.DataFrame([{
        "industry": industry, "country": country, "company_size": company_size,
        "annual_revenue_usd_millions": revenue, "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": adoption_stage, "years_using_ai": years_ai,
        "ai_maturity_score": maturity, "ai_primary_tool": primary_tool,
        "num_ai_tools_used": num_tools, "ai_projects_active": active_projects,
        "ai_budget_percentage": budget, "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": inv_per_emp
    }])

    # Get Prediction
    prediction_raw = model.predict(input_df)[0]
    
    # Visual Layout for Results (2 Columns)
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Calculated ROI Projection")
        st.markdown(f"### Predicted Corporate ROI: **{prediction_raw:.2%}**")
        st.caption("(Total Return on Investment as a percentage of total costs)")
        
        # Derived Metrics
        costs = revenue * (budget / 100)
        gain = costs * (1 + prediction_raw)
        
        st.write(f"**Projected Financial Gain:** ${gain:.2f} Millions")
        st.write(f"**Projected Total Costs:** ${costs:.2f} Millions")
        st.write(f"**Final Net ROI:** {prediction_raw*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with res_col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Company Data Insights & Health Check")
        
        # Maturity Chart
        st.write("📊 **Your Company Maturity vs Industry Average**")
        chart_data = pd.DataFrame({
            'Category': ['Your Company', 'Industry Avg'],
            'Score': [maturity, 55] # 55 is a static baseline for comparison
        })
        st.bar_chart(chart_data.set_index('Category'))
        
        st.write("➕ **Company Condition**")
        # Logic for health status
        if failure_rate > 20:
            st.error("Budget Health: High Risk (Project fail rate is too high)")
        else:
            st.success("Budget Health: Stable")
            
        if maturity < 40:
            st.warning("Maturity Status: Emerging AI User (Needs foundation)")
        else:
            st.info("Maturity Status: Established AI User")
        st.markdown("</div>", unsafe_allow_html=True)

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
