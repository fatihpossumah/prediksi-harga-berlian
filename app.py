import streamlit as st
import numpy as np
import pickle
import json
import pandas as pd

model = pickle.load(open("final_model_xgb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("category_info.json", "r") as f:
    category_info = json.load(f)

st.title("Diamond Price Prediction")

with st.form("prediction_form"):
    st.subheader("Input Diamond Features")
    
    carat = st.number_input("Carat", min_value=0.0, step=0.01, format="%.2f")
    depth = st.number_input("Depth", min_value=0.0, step=0.01, format="%.2f")
    table = st.number_input("Table", min_value=0.0, step=0.01, format="%.2f")
    x = st.number_input("X (length in mm)", min_value=0.0, step=0.01, format="%.2f")
    y = st.number_input("Y (width in mm)", min_value=0.0, step=0.01, format="%.2f")
    z = st.number_input("Z (depth in mm)", min_value=0.0, step=0.01, format="%.2f")
    
    cut = st.selectbox("Cut", category_info["cut"])
    color = st.selectbox("Color", category_info["color"])
    clarity = st.selectbox("Clarity", category_info["clarity"])
    
    submitted = st.form_submit_button("Predict Price")

if submitted:

    input_dict = {
        'carat': carat,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z,
        'cut': cut,
        'color': color,
        'clarity': clarity
    }
    input_df = pd.DataFrame([input_dict])

    for col, categories in category_info.items():
        for cat in categories:
            col_name = f"{col}_{cat}"
            input_df[col_name] = (input_df[col] == cat).astype(int)
        input_df.drop(columns=[col], inplace=True)

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    scale_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    prediction = model.predict(input_df)

    output = round(float(prediction[0]), 2)
    
    st.success(f"Predicted Diamond Price: ${output:,.2f}")