import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np

linear_model = pk.load(open("linear.pkl","rb"))
poly_features = pk.load(open("poly_features.pkl","rb"))
poly_model    = pk.load(open("poly_model.pkl","rb"))
poly_scaler = pk.load(open("poly_scaler.pkl","rb"))
ridge_model  = pk.load(open("ridge.pkl","rb"))
lasso_model  = pk.load(open("lasso.pkl","rb"))
d_t_model  = pk.load(open("decision tree.pkl","rb"))
scaler_linear = pk.load(open("scaler_linear.pkl","rb"))
encoders = pk.load(open("encoders.pkl","rb"))

heading = st.markdown("<h2 style='color:#4CAF50;'>💻 Laptop Price Prediction App</h2>", unsafe_allow_html=True)

company = st.selectbox("Select Company", encoders["Company"].classes_)

Typename = st.selectbox("Select Laptop Type", encoders["TypeName"].classes_)

Inches = st.number_input("Enter Inches", 10.0, 20.0, step=0.1)

IPS = 1 if st.radio("IPS Display", ['Yes', 'No']) == 'Yes' else 0

Touchscreen = 1 if st.radio("Touchscreen", ['Yes', 'No']) == 'Yes' else 0

Width  = st.number_input("Screen Width", 800, 5000, 1920, step=10)

Height = st.number_input("Screen Height", 600, 3000, 1080, step=10)

Cpu = st.selectbox("Select CPU", encoders["Cpu"].classes_)

Gen = st.number_input("Enter Generation", 0.9, 5.0, 1.0, step=0.1)

Ram = st.number_input("RAM (GB)", 2, 64, 8, step=2)

storage_type = st.radio("Select Storage Type", ['SSD', 'HDD', 'Hybrid', 'Flash'])
storage_value = 0

if storage_type == 'SSD':
    storage_value = st.selectbox("Select SSD (GB)", [128, 256, 512, 1024])
    SSD, HDD, Hybrid, Flash = storage_value, 0, 0, 0
elif storage_type == 'HDD':
    storage_value = st.selectbox("Select HDD (GB)", [256, 512, 1024, 2048])
    SSD, HDD, Hybrid, Flash = 0, storage_value, 0, 0
elif storage_type == 'Hybrid':
    storage_value = st.selectbox("Select Hybrid Storage (GB)", [512, 1024])
    SSD, HDD, Hybrid, Flash = 0, 0, storage_value, 0
elif storage_type == 'Flash':
    storage_value = st.selectbox("Select Flash Storage (GB)", [16, 32, 64, 128, 256])
    SSD, HDD, Hybrid, Flash = 0, 0, 0, storage_value

Gpu_brand = st.selectbox("Select GPU Brand", encoders["Gpu Brand"].classes_)

gpu = st.selectbox("Select GPU", encoders["Gpu Name"].classes_)

Opsys = st.selectbox("Select Operating System", encoders["OpSys"].classes_)

Weight = st.number_input("Enter Weight (kg)", 0.7, 5.0, 1.5, step=0.1)

company   = encoders["Company"].transform([company])[0]
Typename  = encoders["TypeName"].transform([Typename])[0]
Cpu       = encoders["Cpu"].transform([Cpu])[0]
Gpu_brand = encoders["Gpu Brand"].transform([Gpu_brand])[0]
gpu       = encoders["Gpu Name"].transform([gpu])[0]
Opsys     = encoders["OpSys"].transform([Opsys])[0]

regression_type = st.radio("Select Regression Model", 
                               ["Linear Regression", "Polynomial Regression", "Decision Tree Regression",
                                "Ridge Regression", "Lasso Regression"])

if st.button("Predict"):
    laptop = pd.DataFrame([[company,Typename,Inches,IPS,Touchscreen,Width,Height,Cpu,Gen,
                            Ram,Gpu_brand,SSD,HDD,Hybrid,Flash,gpu,Opsys,Weight]],
                          columns = ['Company', 'TypeName', 'Inches', 'IPS', 'Touchscreen',
                                'width', 'height', 'Cpu', 'Gen', 'Ram', 'Gpu Brand', 'SSD', 'HDD',
                                'Hybrid', 'Flash Storage', 'Gpu Name', 'OpSys', 'Weight'])

    if regression_type == "Linear Regression":
        scaled_linear = scaler_linear.transform(laptop)
        predict = linear_model.predict(scaled_linear)[0]

    elif regression_type == "Polynomial Regression":
        scaled_poly = poly_scaler.transform(laptop)
        laptop_poly   = poly_features.transform(scaled_poly)
        predict = poly_model.predict(scaled_poly)[0]

    elif regression_type == "Decision Tree Regression":
        scaled_dt = scaler_linear.transform(laptop)
        predict = d_t_model.predict(scaled_dt)[0]

    elif regression_type == "Ridge Regression":
        scaled_r = scaler_linear.transform(laptop)
        predict = ridge_model.predict(scaled_r)[0]

    elif regression_type == "Lasso Regression":
        scaled_l = scaler_linear.transform(laptop)
        predict = lasso_model.predict(scaled_l)[0]

    predict = max(5000,int(predict))
    st.markdown(f"💰 Laptop Price = ₹{predict:,.0f}/-")