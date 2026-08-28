"""
disease_prediction.py
CodeAlpha Machine Learning Internship - Task 4: Disease Prediction from Medical Data

Objective: Predict whether a tumor is malignant or benign using patient
diagnostic measurements (the Breast Cancer Wisconsin dataset).

Approach: Classification using Logistic Regression and Random Forest,
evaluated with accuracy, precision, recall, F1-score, and ROC-AUC.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ---------------------------------------------------------
# 1. Load and inspect the data
# ---------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass balance (0=malignant, 1=benign):")
print(df["target"].value_counts())
print("\nAny missing values?", df.isnull().sum().sum())

# ---------------------------------------------------------
# 2. Feature engineering / preprocessing
# ---------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features — important for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Train models
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaling

# ---------------------------------------------------------
# 4. Evaluate both models
# ---------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n{'-' * 60}")
    print(f"{name} RESULTS")
    print("-" * 60)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=data.target_names))

log_pred = log_reg.predict(X_test_scaled)
log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
evaluate("LOGISTIC REGRESSION", y_test, log_pred, log_proba)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
evaluate("RANDOM FOREST", y_test, rf_pred, rf_proba)

# ---------------------------------------------------------
# 5. Feature importance (Random Forest)
# ---------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_10 = importances.sort_values(ascending=False).head(10)

print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
print("=" * 60)
print(top_10)

# ---------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted_logreg"] = log_pred
results_df["predicted_rf"] = rf_pred
results_df.to_csv("prediction_results.csv", index=False)

summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, log_pred), accuracy_score(y_test, rf_pred)],
    "Precision": [precision_score(y_test, log_pred), precision_score(y_test, rf_pred)],
    "Recall": [recall_score(y_test, log_pred), recall_score(y_test, rf_pred)],
    "F1-Score": [f1_score(y_test, log_pred), f1_score(y_test, rf_pred)],
    "ROC-AUC": [roc_auc_score(y_test, log_proba), roc_auc_score(y_test, rf_proba)],
})
summary.to_csv("model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("Saved:")
print(" - prediction_results.csv  (test set predictions)")
print(" - model_comparison.csv    (metrics side-by-side)")
print("=" * 60)
