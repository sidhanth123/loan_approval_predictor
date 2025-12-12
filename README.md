🚀 Loan Approval Predictor (ML + Streamlit + SHAP)
An end-to-end Machine Learning Loan Approval Prediction System built using:
XGBoost Classifier
Feature Engineering
Full Preprocessing Pipeline
Model Explainability with SHAP
Interactive Streamlit Web App

📌 Features
✔ ML Pipeline
Data cleaning
Feature engineering
Preprocessing (OHE + scaling)
XGBoost model training
Hyperparameter tuning
Model + Preprocessor saved with Joblib

✔ Streamlit Dashboard
Clean UI for user inputs
Real-time prediction
Display of approval probability
SHAP explanation panel
Risk-category estimation

✔ SHAP Explainability
TreeExplainer when possible
KernelExplainer fallback
SHAP summary plot
Top feature contributions

📁 Project Structure
loan-approval-predictor/
│── app.py               # Streamlit UI
│── model.py             # ML training pipeline
│── Shap.py              # SHAP explainability script
│
├── data/
│   └── Enhanced_Loan_Dataset.csv
│
├── models/
│   ├── xgb_best.pkl
│   ├── preprocessor.pkl
│   └── feature_names.csv
│
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/loan-approval-predictor.git
cd loan-approval-predictor

2. Install dependencies
pip install -r requirements.txt

▶️ Run the Streamlit App
streamlit run app.py

The application will launch in your browser at:
http://localhost:8501

📊 Model Explainability
This project uses SHAP to understand how features influence the loan approval decision.
SHAP summary plot
Top feature importance
Per-sample explanation

🧠 Model Details
Model: XGBoost Classifier
Metric: Accuracy / ROC-AUC
Target: Loan_Status (Approved / Rejected)
