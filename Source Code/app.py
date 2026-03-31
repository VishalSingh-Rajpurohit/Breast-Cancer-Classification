import streamlit as st
import pdfplumber
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

# Load model & scaler
model = load_model("base_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🩺 Breast Cancer Prediction (Upload PDF Report)")

# 🔥 Session state
if "results" not in st.session_state:
    st.session_state.results = {}

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF Reports",
    type=["pdf"],
    accept_multiple_files=True
)

# 🔥 Feature extraction
def extract_features(text):
    def find_value(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0

    features = [
        find_value(r"radius_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"texture_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"perimeter_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"area_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"smoothness_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"compactness_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"concavity_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"concave points_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"symmetry_mean\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"fractal_dimension_mean\s*[:=]\s*(\d+\.?\d*)"),

        find_value(r"radius_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"texture_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"perimeter_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"area_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"smoothness_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"compactness_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"concavity_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"concave points_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"symmetry_se\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"fractal_dimension_se\s*[:=]\s*(\d+\.?\d*)"),

        find_value(r"radius_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"texture_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"perimeter_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"area_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"smoothness_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"compactness_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"concavity_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"concave points_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"symmetry_worst\s*[:=]\s*(\d+\.?\d*)"),
        find_value(r"fractal_dimension_worst\s*[:=]\s*(\d+\.?\d*)"),
    ]

    return np.array(features).reshape(1, -1)

# 🔥 Main Loop
if uploaded_files:
    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

        st.subheader(f"📄 {file_name}")

        # Extract text
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        st.text("Extracted Text Preview:")
        st.text(text[:800])

        # 🔥 Predict Button
        if st.button(f"🔍 Predict {file_name}", key=file_name):

            data = extract_features(text)
            data = scaler.transform(data)

            pred = model.predict(data)[0][0]
            confidence = pred if pred > 0.5 else 1 - pred

            # Save result
            st.session_state.results[file_name] = {
                "pred": pred,
                "confidence": confidence,
                "text": text
            }

        # 🔥 SHOW RESULT SAME FILE KE NICHE
        if file_name in st.session_state.results:

            result = st.session_state.results[file_name]

            pred = result["pred"]
            confidence = result["confidence"]
            text = result["text"]

            # Prediction
            if pred > 0.5:
                st.error("⚠️ Malignant (Cancer Detected)")
                prediction_text = "Malignant"
            else:
                st.success("✅ Benign (No Cancer)")
                prediction_text = "Benign"

            # Risk
            if pred < 0.4:
                risk = "Low"
                st.success("🟢 Low Risk")
            elif pred < 0.7:
                risk = "Medium"
                st.warning("🟡 Medium Risk")
            else:
                risk = "High"
                st.error("🔴 High Risk")

            # Confidence
            st.write(f"Prediction Confidence: {confidence:.4f}")
            st.progress(int(confidence * 100))

            # Report text
            report = f"""
Breast Cancer Report

File Name: {file_name}

--- Extracted Features ---
{text}

--- Prediction ---
Result: {prediction_text}
Risk Level: {risk}
Confidence: {confidence:.4f}
"""

            # 📄 TXT Download
            st.download_button(
                f"📄 Download TXT ({file_name})",
                report,
                file_name=f"{file_name}_report.txt"
            )

            # 📄 PDF Download
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []
            for line in report.split("\n"):
                content.append(Paragraph(line, styles["Normal"]))

            doc.build(content)
            buffer.seek(0)

            st.download_button(
                f"📄 Download PDF ({file_name})",
                buffer,
                file_name=f"{file_name}_report.pdf"
            )