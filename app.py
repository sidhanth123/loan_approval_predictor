# app.py  (or streamlit_app.py)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time

# ---- UI / Page config ----
st.set_page_config(page_title="Loan Approval Predictor", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* subtle card look */
.stApp {
  background-color: #f8fafc;
}
h1 { color: #0f172a; }
.card {
  padding: 12px;
  border-radius: 8px;
  background: white;
  box-shadow: 0px 1px 6px rgba(15,23,42,0.06);
}
.small-muted { color: #6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---- Paths (robust, relative to this file) ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.joinpath("..", "models", "xgb_best.pkl").resolve()
PREPROCESSOR_PATH = BASE_DIR.joinpath("..", "models", "preprocessor.pkl").resolve()
FEATURES_CSV = BASE_DIR.joinpath("..", "models", "feature_names.csv").resolve()
DATA_PATH = BASE_DIR.joinpath("..", "data", "Enhanced_Loan_Dataset.csv").resolve()

# ---- Helper: load artifacts with clear errors ----
def load_artifact(path: Path, desc: str):
    if not path.exists():
        st.error(f"Missing {desc} at: {path}")
        raise FileNotFoundError(f"{desc} not found: {path}")
    return joblib.load(path)

# Load model + preprocessor (fail early and show message)
with st.sidebar:
    st.title("Model & Data")
    st.write("Model and preprocessor paths:")
    st.text(str(MODEL_PATH))
    st.text(str(PREPROCESSOR_PATH))
    if st.button("Reload artifacts"):
        st.experimental_rerun()

try:
    model = load_artifact(MODEL_PATH, "XGBoost pipeline (xgb_best.pkl)")
    preprocessor = load_artifact(PREPROCESSOR_PATH, "Preprocessor (preprocessor.pkl)")
except FileNotFoundError:
    st.stop()

# Attempt to load feature names (optional)
feature_names = None
if FEATURES_CSV.exists():
    try:
        feature_names = list(pd.read_csv(FEATURES_CSV, header=None).iloc[:,0])
    except Exception:
        feature_names = None

# ---- Sidebar: input form grouped ----
st.sidebar.header("Applicant Input")
def user_input():
    st.sidebar.subheader("Personal")
    Gender = st.sidebar.selectbox("Gender", ["Male","Female"])
    Married = st.sidebar.selectbox("Married", ["Yes","No"])
    Dependents = st.sidebar.selectbox("Dependents", ["0","1","2","3+"])
    Education = st.sidebar.selectbox("Education", ["Graduate","Not Graduate"])
    Self_Employed = st.sidebar.selectbox("Self Employed", ["No","Yes"])

    st.sidebar.subheader("Financial")
    ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=3000, step=500)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=500)
    LoanAmount = st.sidebar.number_input("Loan Amount (in 1000s)", min_value=10.0, value=100.0, step=5.0)
    Loan_Amount_Term = float(st.sidebar.selectbox("Loan Term (months)", [360.0, 180.0, 120.0]))
    Credit_History = float(st.sidebar.selectbox("Credit History (1=good,0=none)", [1.0, 0.0]))
    Property_Area = st.sidebar.selectbox("Property Area", ["Urban","Rural","Semiurban"])

    # Derived
    TotalIncome = ApplicantIncome + CoapplicantIncome
    Credit_Score = int(600 + (ApplicantIncome / max(1, ApplicantIncome)) * 200 - (1 - Credit_History) * 150)
    DTI = LoanAmount / (TotalIncome + 1)
    LTI = LoanAmount * 1000 / (TotalIncome + 1)
    Risk_Category = "Low" if Credit_Score > 700 else ("Medium" if Credit_Score > 600 else "High")
    Default_Probability = round(1 - (Credit_Score - 300) / 550, 2)

    input_dict = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
        "TotalIncome": TotalIncome,
        "Credit_Score": Credit_Score,
        "DTI": DTI,
        "LTI": LTI,
        "Risk_Category": Risk_Category,
        "Default_Probability": Default_Probability
    }
    return pd.DataFrame([input_dict])

input_df = user_input()

# ---- Main layout ----
st.title("Loan Approval Predictor — Demo")
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Applicant Summary")
    # display transposed input; convert to strings to avoid Arrow errors
    st.write(input_df.T.astype(str))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Actions")
    if st.button("Predict Approval Probability"):
        # Preprocess + predict
        try:
            # Preprocessor expects original columns used in training; ensure order
            X = input_df.copy()
            # Convert / align dtypes if necessary (safe)
            # transform and predict using pipeline preprocessor and model
            X_trans = preprocessor.transform(X)
            # model may be a pipeline; call predict_proba on pipeline if available
            try:
                proba = model.predict_proba(X)[0][1]  # prefer pipeline-level predict
                pred = model.predict(X)[0]
            except Exception:
                # if model is pipeline but expects transformed input, use classifier
                try:
                    classifier = model.named_steps['classifier']
                    proba = classifier.predict_proba(X_trans)[0][1]
                    pred = classifier.predict(X_trans)[0]
                except Exception:
                    # fallback: if model is raw classifier expecting raw X
                    proba = model.predict_proba(X_trans)[0][1]
                    pred = model.predict(X_trans)[0]

            # Show results
            st.success(f"Approval probability: {proba:.2f}")
            st.info("Predicted Approval: " + ("APPROVED" if pred == 1 else "REJECTED"))

            # Score card style
            st.metric("Approval probability", f"{proba:.2f}")
            st.write("**Risk Category:**", input_df.loc[0,'Risk_Category'])
        except Exception as ex:
            st.error("Prediction failed: " + str(ex))
    else:
        st.write("Click **Predict Approval Probability** to evaluate this applicant.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---- Explainability section (collapsible) ----
with st.expander("Model Explainability (SHAP) — Click to expand"):
    st.write("SHAP explanations may take a few seconds. If SHAP fails, a fallback explanation will be provided.")

    # Try SHAP and handle failures
    try:
        import shap
        # Prepare background from dataset if available
        if DATA_PATH.exists():
            df_full = pd.read_csv(DATA_PATH)
            df_full["Loan_Status"] = df_full["Loan_Status"].map({"Y":1,"N":0})
            X_full = df_full.drop(columns=["Loan_Status"])
            # sample and transform
            background_raw = X_full.sample(n=min(50, len(X_full)), random_state=42)
            background_trans = preprocessor.transform(background_raw)
        else:
            # fallback: use the input as background (not ideal but works)
            background_trans = preprocessor.transform(input_df)

        # transform current input
        X_trans = preprocessor.transform(input_df)

        # Build explainer (prefer TreeExplainer on booster if possible)
        use_kernel = False
        try:
            # Extract classifier if wrapped
            try:
                clf = model.named_steps['classifier']
            except Exception:
                # if model is not pipeline, try model directly
                clf = model
            # attempt to get raw booster for TreeExplainer
            booster = None
            if hasattr(clf, "get_booster"):
                booster = clf.get_booster()
            if booster is not None:
                explainer = shap.TreeExplainer(booster)
                sv = explainer.shap_values(X_trans)
            else:
                # if no booster, try universal explainer
                explainer = shap.Explainer(clf.predict_proba, background_trans)
                sv = explainer(X_trans).values if hasattr(explainer, "values") else explainer.shap_values(X_trans)
        except Exception:
            use_kernel = True

        # If we decided kernel (safe fallback)
        if use_kernel:
            with st.spinner("Computing SHAP (model-agnostic KernelExplainer, may take a moment)..."):
                explainer = shap.KernelExplainer(clf.predict_proba, background_trans)
                shap_values = explainer.shap_values(X_trans)
                # take class 1
                sv = np.array(shap_values[1])

        # Normalize sv to 2D and match feature count if needed
        sv = np.array(sv)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        # Match rows
        rows = sv.shape[0]
        X_matched = X_trans[:rows]
        # pad/trim cols if necessary
        if sv.shape[1] < X_matched.shape[1]:
            sv = np.pad(sv, ((0,0),(0, X_matched.shape[1]-sv.shape[1])), mode='constant')
        elif sv.shape[1] > X_matched.shape[1]:
            sv = sv[:, :X_matched.shape[1]]

        # Plot summary (small sample)
        fig = plt.figure(figsize=(7,4))
        shap.summary_plot(sv, X_matched, show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Show top contributions in a table
        abs_vals = np.abs(sv).sum(axis=0)
        idx_top = np.argsort(abs_vals)[-10:][::-1]
        if feature_names:
            feat_map = feature_names
            # safe slice
            feat_map = feat_map[:X_matched.shape[1]]
            top_features = [feat_map[i] if i < len(feat_map) else f"f_{i}" for i in idx_top]
        else:
            top_features = [f"f_{i}" for i in idx_top]
        top_vals = abs_vals[idx_top]
        contrib_df = pd.DataFrame({"feature": top_features, "importance": top_vals})
        st.write("Top feature contributions (absolute SHAP importance):")
        st.table(contrib_df.astype(str))  # convert to str to avoid Arrow issues
    except Exception as e:
        st.warning("SHAP explanation not available: " + str(e))
        # fallback short explanation: show model feature importances if available
        try:
            clf = model.named_steps['classifier'] if hasattr(model, 'named_steps') else model
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                # try to map names
                fn = feature_names or [f"f_{i}" for i in range(len(importances))]
                imp_df = pd.DataFrame({"feature": fn[:len(importances)], "importance": importances})
                st.write("Fallback: model feature importances")
                st.table(imp_df.sort_values("importance", ascending=False).head(10).astype(str))
        except Exception:
            st.info("No fallback explanations available.")

st.markdown("---")
st.write("Demo app — for portfolio use. For production, add input validation, secure model storage, and logging.")
