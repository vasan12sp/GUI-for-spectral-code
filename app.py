import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ----------------------- Preprocessing Functions -----------------------

def apply_snv(data):
    return (data - data.mean(axis=1).values[:, None]) / data.std(axis=1).values[:, None]

def apply_msc(data):
    ref = np.mean(data, axis=0)
    corrected = []
    for row in data.values:
        fit = np.polyfit(ref, row, 1, full=True)
        corrected.append((row - fit[0][1]) / fit[0][0])
    return pd.DataFrame(corrected, columns=data.columns)

def apply_sgd(data):
    from scipy.signal import savgol_filter
    return pd.DataFrame(savgol_filter(data, window_length=11, polyorder=2, axis=1), columns=data.columns)

def apply_preprocessing(data, method):
    if method == 'SNV':
        return apply_snv(data)
    elif method == 'MSC':
        return apply_msc(data)
    elif method == 'SGD':
        return apply_sgd(data)
    elif method == 'SGD+SNV+MSC':
        return apply_msc(apply_snv(apply_sgd(data)))
    elif method == 'None':
        return data
    else:
        return data

# ----------------------- Model Training & Evaluation -----------------------

def train_model(X_train, y_train, model_name):
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100)
    elif model_name == "SVM":
        model = SVR(kernel='rbf')
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor()
    elif model_name == "Elastic Net":
        model = ElasticNet()
    elif model_name == "GBM":
        model = GradientBoostingRegressor()
    else:
        st.error("Unknown model selected.")
        return None
    model.fit(X_train, y_train)
    return model

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rmpe = np.mean(np.abs((y_true - y_pred) / (y_true + np.finfo(float).eps))) * 100
    return rmse, r2, rmpe

# ----------------------- Streamlit GUI -----------------------

st.title("🌱 Soil Spectral Analysis for Nutrient Prediction")

input_file = st.file_uploader("📥 Upload Spectral Input CSV", type=["csv"])
preproc_method = st.selectbox("🔧 Select Preprocessing", ['None', 'SNV', 'MSC', 'SGD', 'SGD+SNV+MSC'])
model_choice = st.selectbox("🤖 Select Model", ['Random Forest', 'SVM', 'XGBoost', 'Elastic Net', 'GBM'])

if input_file is not None:
    df = pd.read_csv(input_file)
    
    if df.shape[1] <= 1:
        st.error("Uploaded CSV must contain spectral features and target nutrient.")
    else:
        # Assume last column is target (e.g., N)
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Apply preprocessing
        X_processed = apply_preprocessing(X, preproc_method)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train, model_choice)
        if model:
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            rmse, r2, rmpe = evaluate(y_test, y_pred)
            
            st.subheader("📊 Evaluation Metrics")
            st.write(f"*RMSE:* {rmse:.4f}")
            st.write(f"*R² Score:* {r2:.4f}")
            st.write(f"*RMPE:* {rmpe:.2f}%")
            
            # Show predictions
            pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.subheader("🔍 Predictions")
            st.dataframe(pred_df.head(20))
            
            # Download button
            csv = pred_df.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")