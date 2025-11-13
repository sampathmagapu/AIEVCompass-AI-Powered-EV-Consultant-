# 🚗⚡ AIEVCompass – AI-Powered Electric Vehicle Consultant
**Shell – Edunet Foundation – AICTE Green Skill Internship Project**

AIEVCompass is an end-to-end **AI-powered EV consulting system** built with a **Hybrid AI ("Two-Brain") architecture**.  
It uses a conversational chatbot to interview users and then feeds that data into a **92.4% accurate expert ML system** that predicts:

- **EV Price** (based on specifications)  
- **Driving Range** (based on performance & efficiency)  
- **Market Segment** (Budget / Mid-Range / Premium / Luxury)

---

## 🎯 What This System Does

| Module | Task | Output |
|--------|------|--------|
| **Price Prediction System (92% accuracy)** | Predict EV market price | Estimated Price |
| **Range Prediction Model** | Predict EV driving range | Predicted Range (km) |
| **Range Category Classifier** | Classify EVs as Short/Medium/Long range | Category Label |

This enables *What-If EV Analysis*, such as:

> “If we build a ₹25L EV with 60 kWh battery, what range should it have?”  
> “If a car has 500 km range, what should be its fair price?”

---

## 🧠 Two-Brain Hybrid AI Architecture

### **🧠 Brain 1 — The “Math Brain” (Expert ML Models)**  
The 92.4% accurate price modeling system uses multiple ML models + scalers to produce mathematically reliable predictions.

| Segment | ±15% Accuracy | ±10% Accuracy | Avg Error ($) |
|--------|:--------------:|:-------------:|:-------------:|
| Budget | 75% | 66.7% | ~$3,383 |
| Mid-Range | **92.9%** | **78.6%** | ~$3,448 |
| Premium | **100%** | **100%** | ~$3,107 |
| Luxury | **100%** | 85.7% | ~$3,787 |

### **💬 Brain 2 — The “Language Brain” (Chatbot Interviewer)**  
A lightweight local LLM (**DialoGPT-small**) guides users through questions to collect EV features conversationally.

### **🔁 How the Flow Works**

1. User starts a chat.  
2. Chatbot asks guided questions (Battery size? Range? Power? etc.)  
3. Answers are stored and validated.  
4. Collected features → Scalers → ML Models  
5. Price / Range / Category predictions are generated.  
6. Chatbot presents the results conversationally.

---

## 🧰 Technology Stack

| Layer | Tools | Purpose |
|------|-------|---------|
| **Machine Learning** | scikit-learn, pandas, numpy | Expert "Math Brain" |
| **Generative AI** | transformers, torch | Chat-based "Language Brain" |
| **Frontend / App** | Streamlit | Chat Interface (app/app.py) |
| **Development** | JupyterLab, matplotlib | Analysis & validation |
| **Version Control** | Git & GitHub | Project hosting |

---

## 🗂 Project Structure

```

AIEVCompass/
│
├── AIEVCompass_Dataset/
│   ├── cars_data_RAW.csv
│   └── cars_data_cleaned.csv
│
├── models/
│   │
│   ├── price/
│   │   ├── price_category_classifier.pkl
│   │   ├── price_model_budget.pkl
│   │   ├── price_model_mid-range.pkl
│   │   ├── price_model_premium.pkl
│   │   ├── price_model_luxury.pkl
│   │   └── price_segment_scalers.pkl
│   │
│   └── range/
│       ├── range_category_classifier.pkl
│       ├── range_prediction_model.pkl
│       └── range_feature_scaler.pkl
│
├── notebooks/
│   ├── AIEVCompass.ipynb
│   └── Chatbot_Playground.ipynb
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md

````

---

## 🏃 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sampathmagapu/AIEVCompass-AI-Powered-EV-Consultant-.git
cd AIEVCompass-AI-Powered-EV-Consultant-
````

### 2️⃣ Create & activate a virtual environment

```bash
python -m venv venv
```

**Windows:**

```bash
.\venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## 👨‍💻 Author

**Sampath Magapu**
📧 Email: *[sampathmagapu11@gmail.com](mailto:sampathmagapu11@gmail.com)*
🔗 LinkedIn: [https://www.linkedin.com/in/sampath-magapu-9b5102253/](https://www.linkedin.com/in/sampath-magapu-9b5102253/)

```
