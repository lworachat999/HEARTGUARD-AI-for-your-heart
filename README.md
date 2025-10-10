# 🫀 HeartGuard - AI-Powered Smart Cardiac Companion

**HeartGuard** is an AI-driven healthcare application designed to predict **heart failure risk** using machine learning, explainable AI, and simulated IoT data integration.  
It is a **desktop software** (not a web app) built with **Python, PyTorch, and PyQt5**, and is designed to be **patent-ready** under the Indian Patent Act (hardware–software linkage).

---

## 🚀 Overview

HeartGuard assists doctors and patients by providing:
- Early detection of cardiac failure risk.
- Personalized health insights using individual baselines.
- Explainable AI results (SHAP-like feature importance).
- Real-time IoT health monitoring and alerting.
- What-if simulation for lifestyle and treatment planning.

It bridges the gap between **predictive modeling**, **explainable decision support**, and **continuous monitoring** — making it a next-generation digital healthcare solution.

---

## ⚙️ Key Features

### 🧠 1. Predictive AI Engine
- Built using **PyTorch** deep neural networks.
- Predicts the likelihood of heart failure based on 12 key features such as:
  - Age, ejection fraction, serum sodium, creatinine, blood pressure, etc.
- Achieves up to **87% accuracy** on synthetic data.

### 🩺 2. Explainable AI (XAI)
- Uses **SHAP-like feature attribution** to explain which features most influence a patient's risk.
- Visualizes risk contribution for each feature, increasing **model transparency** for doctors.

### 👤 3. Personalized Baseline Learning
- Learns what is “normal” for each patient by maintaining a **profile file**.
- Baselines are updated after every prediction to reduce false alarms and improve accuracy over time.

### 🔍 4. What-If Simulation Assistant
- Interactive module where users can tweak parameters (e.g., increase ejection fraction or stop smoking) and instantly see how risk changes.
- Helps doctors and patients **visualize the impact of lifestyle changes**.

### ⌚ 5. IoT & Hardware Integration
- Simulated data streaming from devices like **ESP32 + MPU6050**.
- Prototype supports real-time vitals feed (heart rate, ECG, etc.).
- This hardware link ensures **patent eligibility** under Section 3(k) of the Indian Patent Act.

### 💬 6. Modern UI (PyQt5)
- Clean, professional desktop interface.
- Smooth screen transitions and animations.
- Chatbot-like “Health Assistant” for conversational risk analysis.

---

📊 Model Performance (on Synthetic Data)
Metric	Score
Accuracy	87%
Precision	0.86
Recall	0.88
ROC-AUC	0.92
