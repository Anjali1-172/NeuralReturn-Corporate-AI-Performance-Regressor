import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your best pipeline
try:
    model = joblib.load('best_xgb_model.pkl')
except:
    st.error("Model file 'best_xgb_model.pkl' not found. Please ensure it is in the directory.")

# Page Setup
st.set_page_config(page_title="AI Impact Engine", layout="wide")

# Custom CSS for Neon-Dashboard Hybrid
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #050505 0%, #1a0033 100%);
        color: #00ffcc;
    }
    
    /* Centered Header */
    .header-container {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Dashboard Cards */
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ff00ff;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
        margin-bottom: 20px;
    }

    h1, h2, h3, p, label {
        color: #00ffcc !important;
        text-shadow: 0 0 5px rgba(0, 255, 204, 0.3);
    }
    
    /* Button Style */
    .stButton>button {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px #ff00ff;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div class='header-container'>
        <h1>🚀 AI IMPACT ENGINE</h1>
        <p>Strategic ROI Analytics & Corporate Health Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# --- Input Area (Centered Grid) ---
col1, main_col, col3 = st.columns([1, 6, 1])

with main_col:
    with st.form("main_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📋 Company & Tool Parameters")
        
        # Grid Layout for Inputs
        row1_1, row1_2, row1_3 = st.columns(3)
        with row1_1:
            revenue = st.number_input("Annual Revenue (USD Millions)", min_value=0.0, value=100.0, step=1.0)
            industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Retail", "Consulting", "Manufacturing"])
        with row1_2:
            num_tools = st.number_input("Num AI Tools Used (Count)", min_value=1, value=5, step=1)
            country = st.selectbox("Country", ["USA", "Japan", "Kenya", "Netherlands", "New Zealand", "Other"])
        with row1_3:
            budget = st.slider("AI Budget Percentage (%)", 0.0, 100.0, 15.0)
            company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"])

        row2_1, row2_2, row2_3 = st.columns(3)
        with row2_1:
            maturity = st.slider("AI Maturity Score (0-100)", 0, 100, 50)
        with row2_2:
            failure_rate = st.slider("Project Failure Rate (%)", 0.0, 100.0, 12.0)
        with row2_3:
            years_ai = st.number_input("Years Using AI", min_value=0.0, value=2.0, step=0.5)

        # Advanced/hidden inputs included to match model requirements
        with st.expander("Show Advanced Strategy Inputs"):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                adoption_stage = st.selectbox("Adoption Stage", ["none", "pilot", "partial", "full"], index=2)
            with adv2:
                adoption_rate = st.slider("Internal Adoption Rate (%)", 0, 100, 45)
            with adv3:
                inv_per_emp = st.number_input("AI Investment Per Employee (USD)", value=1500)
                primary_tool = st.text_input("Primary AI Tool", "ChatGPT")
                active_projects = st.number_input("Active AI Projects", value=3)

        submit = st.form_submit_button("GENERATE DETAILED PROJECTION")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Analysis & Result Section ---
if submit:
    # 1. Create Dataframe for Prediction
    input_df = pd.DataFrame([{
        "industry": industry, "country": country, "company_size": company_size,
        "annual_revenue_usd_millions": revenue, "ai_adoption_rate": adoption_rate,
        "ai_adoption_stage": adoption_stage, "years_using_ai": years_ai,
        "ai_maturity_score": maturity, "ai_primary_tool": primary_tool,
        "num_ai_tools_used": num_tools, "ai_projects_active": active_projects,
        "ai_budget_percentage": budget, "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": inv_per_emp
    }])

    try:
        # 2. Prediction
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        res_left, res_right = st.columns(2)

        # LEFT COLUMN: ROI Metrics
        with res_left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 Financial ROI Projection")
            
            # Display result as a large metric
            st.metric(label="Predicted Net ROI", value=f"{prediction:.2%}")
            st.caption("Unit: Total Return on Investment as a % of total AI costs")
            
            # Calculation based on inputs
            est_cost = (revenue * (budget/100))
            est_gain = est_cost * (1 + prediction)
            
            st.write(f"**Est. Operational Cost:** ${est_cost:.2f}M")
            st.write(f"**Est. Value Created:** ${est_gain:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)

        # RIGHT COLUMN: Company Condition Analysis
        with res_right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🏥 Company Health Check")
            
            # Chart Comparison
            chart_data = pd.DataFrame({
                "Score": [maturity, 55], # 55 is a dummy industry average
                "Entity": ["Your Company", "Industry Avg"]
            })
            st.bar_chart(chart_data.set_index("Entity"))
            
            # Analysis Logic
            if maturity < 40:
                st.warning("⚠️ **Condition:** Emerging AI User. Your infrastructure needs foundational strengthening.")
            else:
                st.success("✅ **Condition:** Mature AI User. You are well-positioned for scaling.")

            if failure_rate > 20:
                st.error("❌ **Budget Risk:** High Failure Rate detected. Check project management protocols.")
            elif budget < 5:
                st.info("💡 **Opportunity:** Your budget is low relative to revenue. Increasing spend could accelerate ROI.")
            else:
                st.write("💎 **Budget Health:** Resource allocation is optimized.")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.balloons()

    except Exception as e:
        st.error(f"Prediction Error: {e}")


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
