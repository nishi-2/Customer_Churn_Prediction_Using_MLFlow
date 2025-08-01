import streamlit as st
import pandas as pd
import requests, json
from utils import preprocess_input_data, display_eda

# Set page config
st.set_page_config(page_title="MLFlow Model App", layout="wide")

# Sidebar with tabs
tab = st.sidebar.radio("Select Tab", ["Analytics", "Prediction"])

# Load dataset for EDA
DATA_PATH = "data/churn.csv"

if tab == "Analytics":
    st.title("Data Analytics")
    df = pd.read_csv(DATA_PATH)
    display_eda(df)  # Simple EDA from utils.py

elif tab == "Prediction":
    st.title("Predict using MLflow Model")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)

            st.subheader("Uploaded Data")
            st.dataframe(input_df)

            # Preprocess input data
            processed = preprocess_input_data(input_df)


            X_new_df = pd.DataFrame(processed.toarray() if hasattr(processed, "toarray") else processed)

            payload = {
                 "columns": X_new_df.columns.tolist(),
                 "inputs": X_new_df.values.tolist()
                 }
            


            response = requests.post(
                url="http://127.0.0.1:1234/invocations",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload))

            if response.status_code == 200:
                prediction = response.json()
                st.success("Prediction successful")
                st.write("### Predictions:")
                if isinstance(prediction, list):
                    mapped = ["Yes" if p == 1 else "No" for p in prediction]
                    st.dataframe(pd.DataFrame(mapped, columns=["Prediction"]))
                elif isinstance(prediction, dict) and "predictions" in prediction:
                    mapped = ["Yes" if p == 1 else "No" for p in prediction["predictions"]]
                    st.dataframe(pd.DataFrame(mapped, columns=["Prediction"]))
                else:
                    st.write(prediction)
                    
            else:
                st.error(f"Prediction failed with status code {response.status_code}")
                st.write(response.text)


        except Exception as e:
            st.error(f"Error: {e}")
