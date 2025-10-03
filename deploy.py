import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score

# -----------------
# Load Dataset
# -----------------
st.title("Laptop Price Prediction Models 💻")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # -----------------
    # Select Features and Target
    # -----------------
    st.sidebar.header("Model Settings")
    target_col = st.sidebar.selectbox("Select target column (y)", df.columns)
    feature_cols = st.sidebar.multiselect("Select feature columns (X)", [c for c in df.columns if c != target_col])

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # Detect categorical vs numeric
        categorical_cols = [col for col in feature_cols if df[col].dtype == "object"]
        numeric_cols = [col for col in feature_cols if df[col].dtype != "object"]

        # Preprocessor (OneHot for categorical + passthrough for numeric)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols)
            ]
        )

        # -----------------
        # Model Selection
        # -----------------
        model_choice = st.sidebar.selectbox(
            "Choose a model:",
            ["Linear Regression", "Polynomial Regression", "Ridge", "Lasso", "Logistic Regression"]
        )

        # -----------------
        # Build pipeline
        # -----------------
        if model_choice == "Linear Regression":
            model = Pipeline(steps=[("preprocessor", preprocessor),
                                    ("regressor", LinearRegression())])
        elif model_choice == "Polynomial Regression":
            degree = st.sidebar.slider("Polynomial degree", 2, 5, 2)
            model = Pipeline(steps=[("preprocessor", preprocessor),
                                    ("poly", PolynomialFeatures(degree)),
                                    ("regressor", LinearRegression())])
        elif model_choice == "Ridge":
            alpha = st.sidebar.slider("Alpha for Ridge", 0.01, 10.0, 1.0)
            model = Pipeline(steps=[("preprocessor", preprocessor),
                                    ("regressor", Ridge(alpha=alpha))])
        elif model_choice == "Lasso":
            alpha = st.sidebar.slider("Alpha for Lasso", 0.01, 10.0, 1.0)
            model = Pipeline(steps=[("preprocessor", preprocessor),
                                    ("regressor", Lasso(alpha=alpha))])
        elif model_choice == "Logistic Regression":
            y = (y > y.median()).astype(int)  # convert to binary
            model = Pipeline(steps=[("preprocessor", preprocessor),
                                    ("scaler", StandardScaler(with_mean=False)),
                                    ("classifier", LogisticRegression(max_iter=1000))])

        # -----------------
        # Train
        # -----------------
        model.fit(X, y)

        # -----------------
        # User input dynamically
        # -----------------
        st.sidebar.subheader("Enter values for prediction")

        user_data = {}
        for col in feature_cols:
            if col in categorical_cols:
                user_data[col] = st.sidebar.selectbox(f"{col}:", df[col].unique())
            else:
                user_data[col] = st.sidebar.number_input(f"{col}:", value=float(df[col].median()))

        user_df = pd.DataFrame([user_data])

        prediction = model.predict(user_df)[0]

        # -----------------
        # Show results
        # -----------------
        st.subheader(f"Prediction using {model_choice}:")
        st.write(prediction)

        if model_choice == "Logistic Regression":
            st.write("Accuracy:", accuracy_score(y, model.predict(X)))
        else:
            st.write("R² Score:", r2_score(y, model.predict(X)))

