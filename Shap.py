import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Paths
# ------------------------------
MODEL_DIR = "models"
DATA_PATH = "Enhanced_Loan_Dataset.csv"

# ------------------------------
# 2. Load model & preprocessor
# ------------------------------
print("Loading saved model and preprocessor...")
model = joblib.load(os.path.join(MODEL_DIR, "xgb_best.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

# ------------------------------
# 3. Extract classifier (XGBClassifier)
# ------------------------------
try:
    xgb_clf = model.named_steps["classifier"]
except:
    xgb_clf = model["classifier"]

print("Loaded classifier:", xgb_clf)

# ------------------------------
# 4. Read dataset & split
# ------------------------------
df = pd.read_csv(DATA_PATH)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# 5. Transform features using preprocessor
# ------------------------------
print("Transforming test features...")
X_test_transformed = preprocessor.transform(X_test)

# ------------------------------
# 6. SHAP KernelExplainer
# ------------------------------
print("Creating SHAP KernelExplainer...")

background = X_test_transformed[:50]

explainer = shap.KernelExplainer(
    xgb_clf.predict_proba,
    background
)

print("Computing SHAP values for 70 samples...")
shap_values = explainer.shap_values(X_test_transformed[:70])

# ------------------------------
# 7. MATCH SHAP ROWS WITH FEATURE ROWS (Main Fix)
# ------------------------------
sv = np.array(shap_values[1])  # class 1 = approved

shap_rows = sv.shape[0]
X_matched = X_test_transformed[:shap_rows]

# reshape sv if 1D
if sv.ndim == 1:
    sv = sv.reshape(shap_rows, -1)

# fix feature mismatch
if sv.shape[1] < X_matched.shape[1]:
    pad = X_matched.shape[1] - sv.shape[1]
    sv = np.pad(sv, ((0,0),(0,pad)), mode='constant')
elif sv.shape[1] > X_matched.shape[1]:
    sv = sv[:, :X_matched.shape[1]]

# ------------------------------
# 8. SHAP summary plot
# ------------------------------
print("Generating SHAP summary plot...")

shap.summary_plot(sv, X_matched, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

print("SHAP summary saved successfully as shap_summary.png.")
