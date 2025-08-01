import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os

# Define paths
ENCODERS_PATH = "label_encoders.pkl"
TRANSFORMER_PATH = "preprocessor_label_encoded.pkl"

def display_eda(df):
    st.subheader("Basic Info")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())



def preprocess_input_data(df):
    # Check for existence
    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(f"Label encoders not found at: {ENCODERS_PATH}")
    if not os.path.exists(TRANSFORMER_PATH):
        raise FileNotFoundError(f"Preprocessing transformer not found at: {TRANSFORMER_PATH}")
    
    # Load encoders and transformer
    label_encoders = joblib.load(ENCODERS_PATH)
    transformer = joblib.load(TRANSFORMER_PATH)

    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    # Apply preprocessing
    transformed = transformer.transform(df)
    return pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
