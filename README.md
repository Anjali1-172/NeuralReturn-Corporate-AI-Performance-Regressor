# 📊 AI ROI Prediction System  
### Predicting Business Impact of AI Adoption using Machine Learning

🔗 **Live App:** http://neuralreturn-corporate-ai-performance.streamlit.app/

---

## Overview

This project builds a machine learning system to predict **Return on Investment (ROI)** for organizations based on AI adoption, maturity, and operational metrics.

The goal is to help businesses understand **how effectively AI investments translate into financial and operational outcomes**.

---

## 🎯 Problem Statement

Organizations invest heavily in AI but lack clear visibility into:
- Expected ROI from AI initiatives  
- Key drivers influencing AI success  
- Efficiency of AI investments  

This project addresses the problem by building a predictive model using enterprise-level data.

---

## 📊 Dataset

- 📌 ~110,000+ company records  
- 📌 Features include:
  - Industry, country, company size  
  - AI adoption rate & stage  
  - AI maturity score  
  - Budget allocation & investment  
  - AI failure rate & operational metrics  

---

## ⚙️ Data Processing

- Missing value handling & outlier removal  
- Encoding techniques:
  - Ordinal Encoding (company size, AI stage)  
  - One-Hot Encoding (industry)  
  - Frequency Encoding (country)  
- Feature scaling (for preprocessing consistency)  

---

## 🔧 Feature Engineering

Created 8+ engineered features to capture efficiency and interactions:

- Budget per project  
- Investment per tool  
- Success rate (1 - failure rate)  
- Effective budget  
- AI maturity × adoption interaction  
- Experience factor  
- Projects per tool  
- Revenue per AI investment  

---

## 📉 Target Variable (ROI)

ROI was constructed using **PCA (Principal Component Analysis)** on:

- Revenue Growth (%)  
- Productivity Change (%)  
- Cost Reduction (%)  

This creates a single latent variable representing overall business performance.

---

## 🤖 Machine Learning Models

- 🌲 Random Forest Regressor (Baseline)  
- ⚡ XGBoost Regressor (Primary Model)  

---

## 📈 Model Performance
Model performance plateaued across algorithms → **data-limited problem** 
| Metric | Value |
|------|------|
| R² Score | ~0.45 |
| MAE | ~0.71 |
| RMSE | ~0.89 |

---

## 🧠 Key Insights

- AI maturity and adoption rate are the **strongest drivers of ROI** (~0.60+ correlation)  
- Interaction features significantly improved model performance  
- Industry and country showed **low predictive impact**  
- Model performance plateaued across algorithms → **data-limited problem**  
- ROI is influenced by external factors not present in dataset:
  - Execution quality  
  - Market conditions  
  - Organizational readiness  

---

## 🚀 Live Application

👉 Try the model here:  
**http://neuralreturn-corporate-ai-performance.streamlit.app/**

Features:
- User-friendly input interface  
- Real-time ROI prediction  
- Interactive visual insights  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- Streamlit  

---

## 📌 Key Learnings

- Feature engineering is critical for tabular ML problems  
- Tree-based models do not require feature scaling  
- Real-world datasets often have **limited predictive signal**  
- Model performance can plateau due to **missing real-world variables**  

---

## 📎 Future Improvements

- Add time-series data (ROI trends over time)  
- Include external business factors (market growth, competition)  
- Improve target definition beyond PCA-based ROI  
- Use SHAP for model explainability  

---

## 👩‍💻 Author

**Anjali Patlan**  
Machine Learning & Data Science Enthusiast
