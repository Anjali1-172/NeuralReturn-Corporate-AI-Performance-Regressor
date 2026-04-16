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
try:
    # Ensure this file is in the same folder as this script
    pipeline = joblib.load('best_xgb_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

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
        country = st.selectbox("Country", ["USA", "UK", "Germany", "Canada", "India"])
        company_size = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"])
        revenue = st.number_input("Annual Revenue (USD Millions)", value=150.0)
        inv_per_emp = st.number_input("AI Investment Per Employee", value=1200)

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
        "country": country,
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

'''
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
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E6E9EF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    /* FORCING DESKTOP LOOK ON MOBILE */
    @media (max-width: 768px) {
        /* This prevents the 2-column results from stacking into 1 column */
        [data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            display: flex !important;
            flex-wrap: wrap !important;
        }
        [data-testid="column"] {
            min-width: 45% !important; /* Forces two columns to sit side-by-side */
        }
    }

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
        st.write("") # Placeholder to balance layout

    predict_btn = st.form_submit_button("Generate Prediction & Analysis")

# --- RESULTS SECTION ---
if predict_btn:
    # 3. Construct Dataframe (Order must match training data)
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
        "ai_projects_active": active_projects,
        "ai_budget_percentage": budget,
        "ai_failure_rate": failure_rate,
        "ai_investment_per_employee": inv_per_emp
    }
    
    input_df = pd.DataFrame([input_data])

    try:
        # Prediction
        roi_prediction = model.predict(input_df)[0]
        
        # Row 1: ROI Metrics and Health Check
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Calculated ROI Projection")
            st.metric("Predicted Corporate ROI", f"{roi_prediction:.2%}")
            
            # Financial breakdown calculation
            total_cost = revenue * (budget / 100)
            net_gain = total_cost * roi_prediction
            
            st.write(f"**Estimated AI Investment:** ${total_cost:.2f}M")
            st.write(f"**Predicted Net Gain:** ${net_gain:.2f}M")
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Company Health Check")
            
            # Maturity Visual
            comparison_data = pd.DataFrame({
                'Metric': ['Your Maturity', 'Industry Avg'],
                'Score': [maturity, 52]
            }).set_index('Metric')
            st.bar_chart(comparison_data)
            
            # Condition Logic
            if failure_rate > 25:
                st.error("⚠️ High Risk: Failure rate is above industry standard.")
            else:
                st.success("✅ Operational Health: Failure rate is within safe limits.")
                
            if maturity > 70:
                st.info("🏆 Status: AI Leader")
            else:
                st.info("📈 Status: Growing Adoption")
            st.markdown("</div>", unsafe_allow_html=True)

        # Row 2: Trust & Reliability Report
        st.markdown("<h3 style='text-align: center; margin-top: 30px;'>🛡️ Prediction Trust & Reliability Report</h3>", unsafe_allow_html=True)
        t_col1, t_col2, t_col3 = st.columns(3)

        with t_col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.write("🎯 **Model Stability**")
            st.info("High Consistency")
            st.caption("The model explains ~45% of variance with minimal deviation between training and testing.")
            st.markdown("</div>", unsafe_allow_html=True)

        with t_col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.write("📏 **Margin of Error**")
            st.warning("± 0.71 ROI Points")
            st.caption("Based on Mean Absolute Error (MAE), actual results typically fall within this range.")
            st.markdown("</div>", unsafe_allow_html=True)

        with t_col3:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.write("🔍 **Input Validity**")
            # Logic to check if inputs are extreme
            if revenue > 1000 or budget > 50:
                st.error("Out of Bounds")
                st.caption("Inputs are outside typical training ranges. Prediction may be less accurate.")
            else:
                st.success("High Data Match")
                st.caption("Your company profile matches the core patterns in our training dataset.")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug: Column count provided is", len(input_df.columns))

'''
