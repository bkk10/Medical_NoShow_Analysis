import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from PIL import Image

# ---------- LOAD MODEL AND PREPROCESSING OBJECTS ----------
script_dir = os.path.dirname(os.path.abspath(__file__))

# Then in load_artifacts():
model_path = os.path.join(script_dir, 'appointment_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')
features_path = os.path.join(script_dir, 'feature_columns.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
except ModuleNotFoundError as e:
    missing = str(e)
    st.error(f"ModuleNotFoundError while loading model artifacts: {missing}")
    st.warning("Attempting cloudpickle fallback (may still fail if package code is missing)...")
    try:
        import cloudpickle
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)
        st.success("Loaded model via cloudpickle fallback.")
    except Exception as e2:
        st.error(f"Fallback failed: {e2}")
        st.error("The saved model requires a package that's not installed in this environment.\n\n" \
                 "Add the missing package to `requirements.txt` (e.g., `xgboost`) or include the module's code in the app bundle, then redeploy.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# ---------- PREPROCESSING FUNCTION (same as training) ----------
def preprocess_new_data(df):
    # One‑hot encode categoricals
    cat_cols = ['gender', 'appointment_weekday', 'neighbourhood']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Scale age and waiting_days
    df[['age', 'waiting_days']] = scaler.transform(df[['age', 'waiting_days']])
    # Align columns (missing ones get 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return df

def predict_no_show(df):
    processed = preprocess_new_data(df)
    # Model outputs prob for class 0 (no-show) and class 1 (show)
    prob_no_show = model.predict_proba(processed)[:, 0]   # column 0 = no‑show
    pred_class = (prob_no_show > 0.5).astype(int)
    return prob_no_show, pred_class

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Appointment No‑Show Predictor", layout="wide")
st.title("🏥 Medical Appointment No‑Show Predictor")
st.markdown("Predict whether a patient will **not show up** for their scheduled appointment.")

# Sidebar for input method
st.sidebar.header("Choose input method")
# provide a non-empty label but keep it hidden for layout/accessibility
method = st.sidebar.radio("Input method", ["Single patient (form)", "Upload CSV file"], label_visibility="collapsed")

# ------- Single patient form -------
if method == "Single patient (form)":
    st.subheader("Enter patient details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        waiting_days = st.number_input("Waiting days (days between scheduling and appointment)", 
                                       min_value=0, max_value=200, value=10)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        sms_received = st.selectbox("SMS received", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2:
        gender = st.selectbox("Gender", ["F", "M"])
        appointment_weekday = st.selectbox("Appointment weekday (0=Monday, 6=Sunday)", 
                                           options=list(range(7)), 
                                           format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        neighbourhood = st.text_input("Neighbourhood", "Downtown")
    
    if st.button("Predict"):
        # Create a DataFrame from the input
        input_df = pd.DataFrame({
            'age': [age],
            'waiting_days': [waiting_days],
            'hypertension': [hypertension],
            'sms_received': [sms_received],
            'gender': [gender],
            'appointment_weekday': [appointment_weekday],
            'neighbourhood': [neighbourhood]
        })
        prob, pred = predict_no_show(input_df)
        st.subheader("Prediction result")
        if pred[0] == 1:
            st.error(f"⚠️ Patient is **likely to NO‑SHOW** (probability: {prob[0]:.2f})")
            st.markdown("**Recommendation:** Send a reminder SMS.")
        else:
            st.success(f"✅ Patient is **likely to SHOW** (probability of no‑show: {prob[0]:.2f})")
            st.markdown("**Recommendation:** Normal reminder (optional).")

# ------- Batch prediction via CSV -------
else:
    st.subheader("Upload a CSV file with patient data")
    st.markdown("The CSV must contain columns: `age`, `waiting_days`, `hypertension`, `sms_received`, `gender`, `appointment_weekday`, `neighbourhood`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("**Preview of uploaded data:**")
        st.dataframe(input_df.head())
        
        if st.button("Run predictions"):
            # Check required columns
            required = ['age', 'waiting_days', 'hypertension', 'sms_received', 'gender', 'appointment_weekday', 'neighbourhood']
            missing = [c for c in required if c not in input_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                prob, pred = predict_no_show(input_df)
                result_df = input_df.copy()
                result_df['no_show_probability'] = prob
                result_df['prediction'] = pred
                result_df['prediction_label'] = result_df['prediction'].map({1: 'No-Show', 0: 'Show'})
                st.subheader("Predictions")
                st.dataframe(result_df)
                # Download link
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")
                
                # Simple summary
                st.subheader("Summary")
                n_no_show = pred.sum()
                st.write(f"**{n_no_show} patients** predicted to no‑show (out of {len(pred)}).")

st.sidebar.markdown("---")
st.sidebar.info("Model trained with class balancing. For batch predictions, ensure your CSV has exactly the required columns.")