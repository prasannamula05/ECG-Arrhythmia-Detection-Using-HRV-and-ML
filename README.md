
# 🫀 HRV-Based ECG Arrhythmia Detection

## 📌 Overview

This project presents a signal-processing-driven approach for detecting cardiac arrhythmias using ECG recordings from the MIT-BIH Arrhythmia Database (PhysioNet).

The objective is to build an interpretable machine learning pipeline based on Heart Rate Variability (HRV) features derived from RR intervals rather than relying purely on deep learning models.

---

## 🎯 Objectives

- Preprocess raw ECG signals (noise removal & filtering)
- Detect R-peaks from ECG waveform
- Extract RR intervals
- Compute HRV-based statistical features
- Train and compare machine learning classifiers
- Evaluate performance using cross-validation and ROC analysis

---

## 📂 Dataset

- MIT-BIH Arrhythmia Database
- Source: PhysioNet
- Sampling Frequency: 360 Hz

Binary classification setup:
- 0 → Normal rhythm
- 1 → Arrhythmia

---

## ⚙️ Methodology

### 1️⃣ Signal Preprocessing
- Bandpass Filter (0.5 – 40 Hz)
- 50 Hz Notch Filter (Powerline interference removal)

### 2️⃣ R-Peak Detection
- Peak detection using amplitude thresholding
- Minimum peak distance constraint

### 3️⃣ RR Interval Extraction
\[
RR = \frac{R_{i+1} - R_i}{F_s}
\]

### 4️⃣ Feature Engineering (HRV-Based)

Time-Domain Features:
- Mean RR
- SDNN
- RMSSD
- Mean Heart Rate

Statistical Features:
- Variance
- Kurtosis

### 5️⃣ Machine Learning Models
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### 6️⃣ Evaluation
- 5-Fold Cross-Validation
- Confusion Matrix
- ROC Curve
- Accuracy, Precision, Recall, F1-Score

---

## 🧠 Project Pipeline

Raw ECG  
→ Filtering  
→ R-Peak Detection  
→ RR Interval Extraction  
→ HRV Feature Engineering  
→ ML Classification  
→ Performance Evaluation  

---

## 📊 Results (Under working 


## 🛠️ Tech Stack

- MATLAB
- Signal Processing Toolbox
- Statistics & Machine Learning Toolbox

---

## 📌 Key Insights

- HRV features provide interpretable indicators of autonomic cardiac regulation.
- Classical ML models can achieve competitive performance with carefully engineered physiological features.
- Proper signal preprocessing significantly improves classification reliability.

---

## 🚀 Future Work

- Multi-class arrhythmia classification
- Deep learning (1D CNN) comparison

---

## 👩‍🔬 Author

[Prasanna Mula]
