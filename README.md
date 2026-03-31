# 🩺 Breast Cancer Classification using Neural Network

## 📌 Project Overview
This project focuses on predicting whether a tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using a **Neural Network (Deep Learning)** model.

The system is designed as an **end-to-end solution**, including:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model building (Base & Advanced)
- Model evaluation
- Deployment using Streamlit (PDF-based prediction)

---

## 📊 Dataset Information
- Dataset: Breast Cancer Wisconsin Dataset
- Total Features: 30 numerical features
- Target Variable: Diagnosis (M = Malignant, B = Benign)

### 🔑 Important Features:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity

---

## ⚙️ Data Preprocessing
- Removed unnecessary columns
- Label Encoding (M → 1, B → 0)
- Feature Scaling using **StandardScaler**
- Outlier Handling using **IQR Method**

---

## 📈 Exploratory Data Analysis (EDA)
- Distribution plots for all features
- Correlation heatmap
- Pairplot analysis
- Boxplots for outlier detection

### 🔍 Key Insight:
Tumor size-related features (radius, area, perimeter) are highly correlated and important for prediction.

---

## 🤖 Model Building

### 🔹 Base Model
- Dense Layers: 16 → 8
- Activation: ReLU
- Output: Sigmoid
- Accuracy: **98.24%**

### 🔹 Advanced Model
- Added Dropout (0.3)
- Increased complexity
- Accuracy: **97.36%**

---

## 📊 Model Evaluation
- Confusion Matrix
- Classification Report
- ROC Curve (AUC = **1.0**)

### ✅ Final Selection:
👉 **Base Model Selected**
- Higher accuracy
- Fewer errors
- Simpler & more efficient

---

## 🌐 Deployment
Built a **Streamlit Web Application** with features:

- Upload PDF medical reports
- Automatic feature extraction
- Real-time prediction
- Risk level detection (Low / Medium / High)
- Confidence score display
- Download result (TXT / PDF)

---

## 📤 Output Example
- Prediction: Benign / Malignant
- Risk Level: Low / Medium / High
- Confidence Score

---

## 🚀 Advantages
- High accuracy (~98%)
- Fully automated system
- User-friendly interface
- Real-time prediction

---

## ⚠️ Limitations
- Depends on structured PDF format
- Cannot replace medical professionals
- Does not predict cancer stage

---

## 🔮 Future Scope
- OCR integration for scanned reports
- Mobile application development
- Integration with hospital systems
- Improve model using larger datasets

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 👨‍💻 Author
**Vishal Singh Rajpurohit**
(Data Science & Machine Learning Enthusiast)

---

## ⭐ If you like this project, give it a star!