import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Alzheimer’s Prediction Dashboard")

# Load data
df = pd.read_csv("alzheimers_prediction_dataset.csv")

st.header("Dataset Preview")
st.dataframe(df.head())
st.header("Age Distribution")
fig = px.histogram(df, x="Age", nbins=20, title="Age Distribution of Patients")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# DATA PREPARATION
# -----------------------
le = LabelEncoder()
df["Alzheimer’s Diagnosis"] = le.fit_transform(df["Alzheimer’s Diagnosis"])

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Alzheimer’s Diagnosis", axis=1)
y = df_encoded["Alzheimer’s Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------
# MODELING
# -----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------
# EVALUATION
# -----------------------
y_pred = model.predict(X_test)

st.header("Model Performance")

st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.write("Precision:", round(precision_score(y_test, y_pred), 3))
st.write("Recall:", round(recall_score(y_test, y_pred), 3))
st.write("F1 Score:", round(f1_score(y_test, y_pred), 3))
