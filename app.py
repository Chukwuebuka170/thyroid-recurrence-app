
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os.path

# ---------- 1. Load artefacts ----------
model_file = "thyroid_recurrence_lr.pkl"
if not os.path.exists(model_file):
    st.error(f"Error: The file '{model_file}' does not exist in the directory. Please ensure it is uploaded or check the file path.")
    st.stop()

artefacts = joblib.load(model_file)
model = artefacts["model"]
encoder = artefacts["encoder"]
scaler = artefacts["scaler"]
exp_cols = artefacts["columns"]

# ---------- 2. Streamlit UI ----------
st.set_page_config(page_title="Thyroid Recurrence Predictor")
st.title("🧠 Thyroid Cancer Recurrence Risk")

with st.form("patient_form"):
    st.subheader("Patient information")
    age = st.number_input("Age", 1, 120, value=45)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    hx_smoking = st.selectbox("Hx Smoking", ["No", "Yes"])
    hx_radiothreapy = st.selectbox("Hx Radiotherapy", ["No", "Yes"])
    thyroid_function = st.selectbox("Thyroid Function", [
        "Euthyroid", "Clinical Hyperthyroidism", "Clinical Hypothyroidism",
        "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"])
    physical_exam = st.selectbox("Physical Examination", [
        "Normal", "Single nodular goiter-left", "Single nodular goiter-right",
        "Multinodular goiter", "Diffuse goiter"])
    adenopathy = st.selectbox("Adenopathy", [
        "No", "Left", "Right", "Bilateral", "Posterior", "Extensive"])
    pathology = st.selectbox("Pathology", [
        "Papillary", "Follicular", "Hürthle", "Other"])
    focality = st.selectbox("Focality", ["Unifocal", "Multifocal"])
    risk = st.selectbox("Risk", ["Low", "Intermediate", "High"])
    tumor = st.selectbox("Tumor (T)", ["T1", "T2", "T3", "T4"])
    lymph_nodes = st.selectbox("Nodes (N)", ["N0", "N1a", "N1b"])
    cancer_metastasis = st.selectbox("Metastasis (M)", ["M0", "M1"])
    stage = st.selectbox("Stage", ["I", "II", "III", "IVA", "IVB"])
    treatment_response = st.selectbox("Treatment Response", ["Positive", "Stable", "Negative"])

    submitted = st.form_submit_button("Predict")

# ---------- 3. Pre-process & predict ----------
if submitted:
    raw = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "hx_smoking": hx_smoking,
        "hx_radiothreapy": hx_radiothreapy,
        "енту: "thyroid_function": thyroid_function,
        "physical_examination": physical_exam,
        "adenopathy": adenopathy,
        "pathology": pathology,
        "focality": focality,
        "risk": risk,
        "tumor": tumor,
        "lymph_nodes": lymph_nodes,
        "cancer_metastasis": cancer_metastasis,
        "stage": stage,
        "treatment_response": treatment_response
    }])

    cat_cols = encoder.feature_names_in_.tolist()
    missing_cols = set(cat_cols) - set(raw.columns)
    if missing_cols:
        st.error(f"Error: Missing expected columns in input: {missing_cols}")
        st.stop()

    encoded = pd.DataFrame(
        encoder.transform(raw[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )

    if scaler.n_features_in_ != 1:
        st.error(f"Error: Scaler expects {scaler.n_features_in_} features, but only 'age' provided.")
        st.stop()
    age_scaled = pd.DataFrame(
        scaler.transform(raw[['age']]),
        columns=['age']
    )

    processed = pd.concat([age_scaled, encoded], axis=1)
    missing = set(exp_cols) - set(processed.columns)
    for col in missing:
        processed[col] = 0
    processed = processed[exp_cols]

    if hasattr(model, "classes_"):
        positive_class = 1  # Assuming 1 is the positive class ("Yes")
        prob_idx = list(model.classes_).index(positive_class)
        prob = model.predict_proba(processed)[0, prob_idx]
        pred = model.predict(processed)[0]

        st.markdown("---")
        st.subheader("📊 Prediction")
        st.write(f"**Probability of recurrence:** `{prob:.1%}`")
        st.write(f"**Predicted class:** {'Yes' if pred == positive_class else 'No'}")
        if 0.4 < prob < 0.6:
            st.warning("The prediction is close to the decision boundary, indicating uncertainty.")
    else:
        st.error("Error: Model does not support class prediction.")
        st.stop()
