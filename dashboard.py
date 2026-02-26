import streamlit as st
import pickle
import gdown
import os
import numpy as np
import pandas as pd

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="🤖",
    layout="centered"
)

# ── Load Model from Google Drive ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... please wait ⏳"):
            url = "https://drive.google.com/uc?id=1dYBJUze-GVt_2-lDbAE2fCKXXW04CLI_"
            gdown.download(url, model_path, quiet=False)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 ML Model Prediction Dashboard")
st.markdown("Fill in the input values below and click **Predict** to get a result.")
st.divider()

# ── Input Features ─────────────────────────────────────────────────────────────
st.subheader("📥 Input Features")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0, step=0.1)
    feature2 = st.number_input("Feature 2", value=0.0, step=0.1)
    feature3 = st.number_input("Feature 3", value=0.0, step=0.1)

with col2:
    feature4 = st.number_input("Feature 4", value=0.0, step=0.1)
    feature5 = st.number_input("Feature 5", value=0.0, step=0.1)
    feature6 = st.number_input("Feature 6", value=0.0, step=0.1)

st.divider()

# ── Predict Button ─────────────────────────────────────────────────────────────
if st.button("🔮 Predict", use_container_width=True):
    try:
        input_data = np.array([[feature1, feature2, feature3,
                                 feature4, feature5, feature6]])

        prediction = model.predict(input_data)

        st.success(f"### ✅ Prediction Result: `{prediction[0]}`")

        # Show probabilities if classifier
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            st.subheader("📊 Prediction Probabilities")
            proba_df = pd.DataFrame({
                "Class": [f"Class {i}" for i in range(len(proba))],
                "Probability": proba
            })
            st.bar_chart(proba_df.set_index("Class"))

        # Show confidence score if available
        if hasattr(model, "decision_function"):
            score = model.decision_function(input_data)
            st.info(f"Decision Score: `{score[0]:.4f}`")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.warning("Make sure your input features match what the model was trained on.")

# ── Model Info ─────────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Model Info"):
    st.write(f"**Model Type:** `{type(model).__name__}`")
    if hasattr(model, "n_features_in_"):
        st.write(f"**Expected Features:** `{model.n_features_in_}`")
    if hasattr(model, "classes_"):
        st.write(f"**Classes:** `{list(model.classes_)}`")
    if hasattr(model, "feature_names_in_"):
        st.write(f"**Feature Names:** `{list(model.feature_names_in_)}`")
