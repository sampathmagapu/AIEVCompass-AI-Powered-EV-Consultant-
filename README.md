# AIEVCompass: The AI-Powered EV Consultant 🚗⚡

This project is done under the **Shell-Edunet Foundation- AICTE- AI Green skills Internship**. It's an end-to-end data science application that transforms a dataset of electric vehicle specifications into an interactive "AI Consultant" using machine learning and Generative AI.

---

## 📋 Project Overview

The electric vehicle (EV) market is growing rapidly, making it difficult for both consumers and manufacturers to analyze performance, pricing, and market trends.

This project solves this problem by creating an "AI EV Consultant." This tool serves two main purposes:

1.  **For Consumers:** It acts as an expert assistant to help them find the right car. A user can ask, "What price should I expect for an EV with a 500km range?" and get a data-backed answer.
2.  **For Managers:** It acts as a "strategy consultant." A manager can perform "what-if" analysis by asking, "If we want to launch a $30,000 EV, what kind of range should we be targeting to be competitive?"

## 🧠 The Core Idea: The "What-If" Consultant

This application is more than a simple data dashboard. It's built on a "what-if" analysis engine.

1.  **Machine Learning Backend:** We first build and train **two** separate regression models from the `cars_data_cleaned.csv` dataset:
    * **Model 1 (Price Predictor):** Predicts an EV's `Estimated_US_Value` based on its technical specs (Range, Battery, Top Speed, etc.).
    * **Model 2 (Range Predictor):** Predicts an EV's `Range_km` based on its `Estimated_US_Value` and other features.
    These models are saved as `.pkl` files.

2.  **Generative AI Frontend:** The user interacts with a **Streamlit** chatbot. This chatbot is powered by a Generative AI API (like Google's Gemini).

3.  **The "Magic":** When a user asks a "what-if" question:
    * The GenAI understands the user's *intent* (e.g., "they are asking for a price").
    * The Python backend calls the appropriate ML model (e.g., `price_model.pkl`).
    * The model returns a number (e.g., `42500`).
    * The GenAI takes this number and formulates a smart, human-friendly answer: "Based on current market data, a car with 500km of range has an estimated price of around **$42,500**."

---

## 🧰 Technology Stack

This project integrates a full stack of modern data science tools:

* **Backend:** **Python**
    * Serves the Streamlit application.
    * Manages data processing and ML model inference.
* **Frontend:** **Streamlit**
    * Used to build the interactive web application and chatbot interface.
* **Machine Learning:** **Scikit-learn**
    * Used for the entire ML workflow:
    * Data Preprocessing (StandardScaler)
    * Model Training (Linear Regression, Random Forest Regressor)
    * Model Evaluation (R-squared, MAE)
* **Generative AI:** **Google Gemini API** (or **OpenAI API**)
    * Powers the natural language understanding and response generation for the "AI Consultant" chatbot.
* **Data Analysis & Vis:** **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
    * Used in the Jupyter Notebook for cleaning, Exploratory Data Analysis (EDA), and creating visualizations like the correlation heatmap.
* **Environment & Tools:**
    * **Jupyter Notebook:** For all data analysis and model experimentation.
    * **VS Code:** As the primary code editor.
    * **Git & GitHub:** For version control and project hosting.

---

## 📂 Repository Structure

```

AIEVCompass/
│
├── 1\_EV\_Data\_Analysis.ipynb   \# Jupyter Notebook for EDA, model training, and evaluation.
├── 2\_app.py                   \# The main Python script to run the Streamlit application.
│
├── price\_model.pkl            \# Saved ML model for price prediction.
├── range\_model.pkl            \# Saved ML model for range prediction.
│
├── cars\_data\_cleaned.csv      \# The clean dataset used for training and analysis.
├── requirements.txt           \# List of all Python libraries needed to run the project.
├── .env                       \# File to store private API keys (add to .gitignore).
└── README.md                  \# This file.

````

---

## 🚀 How to Run Locally

Follow these steps to set up and run the project on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/AIEVCompass.git](https://github.com/YOUR_GITHUB_USERNAME/AIEVCompass.git)
cd AIEVCompass
````

### 2\. Create a Virtual Environment & Install Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# Activate it (macOS/Linux)
source vCenv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```

### 3\. Set Up Your API Key

The Generative AI chatbot requires an API key (e.g., from Google AI Studio for Gemini).

1.  Create a file in the main folder named `.env`
2.  Open the file and add your API key:
    ```
    GEMINI_API_KEY="YOUR_UNIQUE_API_KEY_HERE"
    ```

### 4\. Run the Streamlit App

With your environment active, run the following command in your terminal:

```bash
streamlit run 2_app.py
```

Your default web browser will open, and you can now interact with the "AI EV Consultant."

-----

## 🧑‍💻 Author

  * **Sampath Magapu**
  * **Email:** sampathmagapu11@gmail.com
  * **LinkedIn:** [https://www.linkedin.com/in/sampath-magapu-9b5102253/](https://www.linkedin.com/in/sampath-magapu-9b5102253/)

## 📜 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```
```
