import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap


# Page settings
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.write("Predicting if a customer is likely to churn based on their details.")


# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

df = load_data()


# Preprocessing
# -----------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove("Churn")  # target variable
numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

# Encode categorical variables

encoders = {}
for col in categorical_cols + ["Churn"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & Target

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numerical features

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save feature order for prediction

feature_names = X.columns.tolist()

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Sidebar: user input
# -----------------------------
st.sidebar.header("Enter Customer Details")
user_input = {}

# Numerical features → number input

for col in numerical_cols:
    user_input[col] = st.sidebar.number_input(
        f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())
    )

# Categorical features → dropdown

for col in categorical_cols:
    option = st.sidebar.selectbox(f"{col}", encoders[col].classes_)
    user_input[col] = encoders[col].transform([option])[0]

# Convert to DataFrame

user_df = pd.DataFrame([user_input])

# Ensure correct column order

user_df = user_df.reindex(columns=feature_names, fill_value=0)

# Scale numerical columns

user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])


# Prediction
# -----------------------------
prediction = model.predict(user_df)[0]
probability = model.predict_proba(user_df)[0][1]

st.subheader("📌 Prediction Result")
if prediction == 1:
    st.error(f"The customer is **likely to churn**. (Probability: {probability:.2%})")
else:
    st.success(f"The customer is **likely to stay**. (Probability: {probability:.2%})")



# Model Evaluation
# -----------------------------
st.subheader("📊 Model Evaluation")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: **{acc:.2%}**")

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Classification Report

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))


# Logistic Regression Explanation
# -----------------------------
st.subheader("📊 Feature Importance (Logistic Regression)")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

st.write("Positive = pushes towards churn | Negative = pushes towards staying")
st.dataframe(coef_df)

# Plot coefficients

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
st.pyplot(fig)


# SHAP Explanations
# -----------------------------
st.subheader("🤖 Explainable AI with SHAP")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(user_df)

# Local explanation

st.write("**Why this customer prediction?**")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# Global importance

st.write("**Overall Feature Importance (Global SHAP values):**")
shap_values_global = explainer(X_test[:200])  # sample for speed
fig2, ax2 = plt.subplots()
shap.plots.bar(shap_values_global, show=False)
st.pyplot(fig2)
