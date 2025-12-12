
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

DATA_PATH = "C:/Users/Siddhant/Downloads/loan_prediction_model/Enhanced_Loan_Dataset.csv"  
df = pd.read_csv(DATA_PATH)

# Basic info
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include='all').T)

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Status Distribution')
plt.show()

# Credit score distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Credit_Score'].dropna(), bins=30, kde=True)
plt.title('Credit Score Distribution')
plt.show()

# Applicant income vs Loan amount
plt.figure(figsize=(8,5))
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df, alpha=0.7)
plt.title('Applicant Income vs Loan Amount by Loan Status')
plt.show()

# TotalIncome vs LoanAmount
plt.figure(figsize=(8,5))
sns.scatterplot(x='TotalIncome', y='LoanAmount', hue='Loan_Status', data=df, alpha=0.7)
plt.title('Total Income vs Loan Amount by Loan Status')
plt.show()

# Boxplot: Credit_Score by Loan_Status
plt.figure(figsize=(7,5))
sns.boxplot(x='Loan_Status', y='Credit_Score', data=df)
plt.title('Credit Score by Loan Status')
plt.show()

# DTI vs Loan_Status
plt.figure(figsize=(7,5))
sns.boxplot(x='Loan_Status', y='DTI', data=df)
plt.title('DTI Distribution by Loan Status')
plt.show()

# Correlation heatmap for numeric features
numeric_cols = ['ApplicantIncome','CoapplicantIncome','TotalIncome','LoanAmount','Loan_Amount_Term','Credit_Score','DTI','LTI','Default_Probability']
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix (numeric features)')
plt.show()

# Categorical analysis: Property_Area vs Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
plt.title('Property Area vs Loan Status')
plt.show()

# Dependents vs Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Dependents', hue='Loan_Status', data=df)
plt.title('Dependents vs Loan Status')
plt.show()

# Save a few EDA summary plots to folder
os.makedirs('../reports/figures', exist_ok=True)
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Status Distribution')
plt.savefig('../reports/figures/loan_status_dist.png', bbox_inches='tight')
plt.close()

print("EDA complete. Check ./reports/figures for saved visuals.")
