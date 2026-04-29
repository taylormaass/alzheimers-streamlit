import streamlit as st
import pandas as pd

st.title("Alzheimer’s Prediction Dashboard")

df = pd.read_csv("alzheimers_prediction_dataset.csv")

st.write("Dataset Preview")
st.dataframe(df)