# src/model_pipeline.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "C:/Users/Siddhant/Downloads/loan_prediction_model/Enhanced_Loan_Dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Target and features
TARGET = 'Loan_Status'  # 'Y'/'N'
# Convert target to binary 1 = approved (Y), 0 = not approved (N)
df[TARGET] = df[TARGET].map({'Y':1,'N':0})

# Drop ID column
if 'Loan_ID' in df.columns:
    df = df.drop(columns=['Loan_ID'])

# Select features: drop obviously derived columns if you prefer
features = [
    'Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',
    'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area',
    'TotalIncome','Credit_Score','DTI','LTI','Risk_Category','Default_Probability'
]

df = df[features + [TARGET]].copy()

# 3. Train-test split
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Preprocessing pipelines
numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','TotalIncome','Credit_Score','DTI','LTI','Default_Probability']
categorical_features = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Risk_Category']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 5. Models to train: Logistic (baseline), RandomForest, XGBoost
pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])

pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])

pipe_lr = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=1000))])

# 6. Quick parameter grid for RandomizedSearchCV (XGBoost chosen)
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(pipe_xgb, param_distributions=param_dist, n_iter=12, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1, random_state=42)

print("Starting hyperparameter search for XGBoost (this may take several minutes)...")
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
best_xgb = search.best_estimator_

# 7. Evaluate function
def evaluate_model(model, X_test, y_test, tag="model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print(f"=== Evaluation: {tag} ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Average Precision (PR-AUC):", average_precision_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    return y_pred, y_proba

# Evaluate best_xgb, rf, lr
evaluate_model(best_xgb, X_test, y_test, tag="XGBoost (best)")
print("Training RandomForest baseline...")
pipe_rf.fit(X_train, y_train)
evaluate_model(pipe_rf, X_test, y_test, tag="RandomForest")
print("Training Logistic Regression baseline...")
pipe_lr.fit(X_train, y_train)
evaluate_model(pipe_lr, X_test, y_test, tag="LogisticRegression")

# 8. Save the best model
joblib.dump(best_xgb, os.path.join(MODEL_DIR, "xgb_best.pkl"))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))  # optional separate export
print("Model and preprocessor saved to", MODEL_DIR)

# 9. Feature names for model interpretation (get onehot feature names)
# Fit preprocessor on full X to get feature names
preprocessor.fit(X)
# numeric names remain the same
num_cols = numeric_features
# get ohe feature names
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([num_cols, cat_cols])
pd.Series(feature_names).to_csv(os.path.join(MODEL_DIR, "feature_names.csv"), index=False)
print("Feature names exported.")
