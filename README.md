# 🚗⚡ AIEVCompass – AI-Powered Electric Vehicle Consultant
Shell – Edunet Foundation – AICTE Green Skill Internship Project

AIEVCompass is an end-to-end *AI-powered EV consulting system* built with a "Hybrid AI" (Two-Brain) architecture. It uses a conversational chatbot to interview the user, then feeds that data to a 92.4% accurate "expert" machine learning system to predict:

- *Price* of an EV based on its specifications
- *Range* based on performance + efficiency features
- *Market Segment* interpretation for consumer & OEM decision-making

---

## 🎯 What This System Does

| Module | Task | Output |
|-------|------|--------|
| *Price Prediction System (92% accuracy)* | Predict EV price based on specs | $ Predicted Market Price + Confidence |
| *Range Prediction Model* | Predict driving range from specs + price | Estimated km Range |
| *Range Category Classifier* | Categorizes vehicles as Short/Medium/Long range | EV Range Category label |

This enables *What-If EV Analysis* such as:
> "If we build a ₹25L EV with 60 kWh battery, what range should it have?"
> "If a car has 500 km range, what should be its fair price?"

---

## 🧠 Core Idea: The "Two-Brain" Hybrid AI

This project's main innovation is a hybrid architecture that combines the best of Machine Learning and Generative AI.

### 1. Brain #1: The "Math Brain" (Your Expert Models)
This is the *"Segment-Aware Price Modeling"* system, which achieves 92.4% accuracy. It uses a "team" of pre-trained models (.pkl files) to perform hyper-accurate mathematical predictions based on your self-validating logic.

| Segment | Accuracy (±15%) | Accuracy (±10%) | Avg Error ($) |
|--------|:---------------:|:---------------:|:-------------:|
| Budget | 75% | 66.7% | ~$3,383 |
| Mid-Range | *92.9%* | *78.6%* | ~$3,448 |
| Premium | *100%* | *100%* | ~$3,107 |
| Luxury | *100%* | 85.7% | ~$3,787 |

### 2. Brain #2: The "Language Brain" (The AI Interviewer)
This is a lightweight, *local LLM* (microsoft/DialoGPT-small) that runs on the user's machine. It acts as a conversational "interviewer" to guide the user.

*How They Work Together (The "AI Interviewer" Flow):*
1.  *User:* "Hi, I want to predict a price."
2.  *Language Brain (Chatbot):* "Great! To run my 92% accurate models, I need to ask you a few questions. First, what is the *Range (in km)?*"
3.  *User:* "500"
4.  *Python (App):* The chatbot saves Range: 500 to its memory.
5.  *Language Brain (Chatbot):* "Got it. Next, what is the *Battery size (in kWh)?*"
6.  (...this continues until all 6+ features are collected...)
7.  *Python (App):* The app takes all the collected features, *scales them* using price_features_scaler.pkl, and feeds them to the **"Math Brain" (.pkl models).
8.  *Math Brain:* Runs your "self-validating" logic > returns "Premium" and $87,450.
9.  *Language Brain (Chatbot):* "OK, I've got the result! For those specs, the car falls into the *Premium* category, with a predicted market price of *$87,450*."

---

## 🧰 Technology Stack

| Layer | Tools Used | Purpose |
|---|---|---|
| *Machine Learning* | scikit-learn, pandas, numpy | The "Math Brain" (92.4% accurate Gradient Boosting models + Scalers). |
| *Generative AI* | transformers, torch | The "Language Brain" (a local LLM like DialoGPT-small) for conversational UI. |
| *Frontend / App* | streamlit | The "AI Interviewer" chat interface (in app/app.py). |
| *Experimentation* | jupyterlab, matplotlib, seaborn | Used in the notebooks/ folder to build and validate the models. |
| *Version Control* | git, github | Project hosting and versioning. |

---

## 🗂 Project Structure



AIEVCompass/
│
├── AIEVCompass_Dataset/                     # Datasets used for training & testing
│   ├── cars_data_RAW.csv                    # Original collected dataset
│   └── cars_data_cleaned.csv                # Preprocessed dataset used in models
│
├── models/                                  # Trained ML models for price & range prediction
│   │
│   ├── price/                               # Price prediction and classification models
│   │   ├── price_category_classifier.pkl
│   │   ├── price_model_budget.pkl
│   │   ├── price_model_mid-range.pkl
│   │   ├── price_model_premium.pkl
│   │   ├── price_model_luxury.pkl
│   │   └── price_segment_scalers.pkl        # Scaler used for all price prediction models
│   │
│   └── range/                               # Range estimation models
│       ├── range_category_classifier.pkl
│       ├── range_prediction_model.pkl
│       └── range_feature_scaler.pkl         # Scaler used for range model normalization
│
├── notebooks/                               # Development & experimentation notebooks
│   ├── AIEVCompass.ipynb                    # Core ML model analysis & logic
│   └── Chatbot_Playground.ipynb             # Chatbot development and testing
│
├── app/                                     # Streamlit web application
│   └── app.py                               # Main Streamlit chatbot UI
│
├── requirements.txt                         # Dependencies list for environment setup
└── README.md                                # Project documentation


`

---

## 🏃 How to Run

1.  *Clone the repository:*
    bash
    git clone [https://github.com/sampathmagapu/AIEVCompass-AI-Powered-EV-Consultant-.git](https://github.com/sampathmagapu/AIEVCompass-AI-Powered-EV-Consultant-.git)
    cd AIEVCompass-AI-Powered-EV-Consultant-
    

2.  *Create and activate a virtual environment:*
    bash
    python -m venv venv
    
    # On Windows
    .\venv\Scripts\activate
    
    # On Mac/Linux
    source venv/bin/activate
    

3.  *Install all required libraries:*
    bash
    pip install -r requirements.txt
    

4.  *Run the Streamlit Chatbot App:*
    bash
    streamlit run app/app.py
    

---

## 👨‍💻 Author

*Sampath Magapu*
📧 sampathmagapu11@gmail.com
🔗 LinkedIn — [https://www.linkedin.com/in/sampath-magapu-9b5102253/](https://www.linkedin.com/in/sampath-magapu-9b5102253/)
````
