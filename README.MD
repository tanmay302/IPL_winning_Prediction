# 🏏 IPL Winning Prediction

A Machine Learning project that predicts the winning probability of IPL teams based on match conditions like teams, score, overs, wickets, and venue.

---
<p align="center">
  <img src="Images/Screenshot 2026-04-03 175849.png" alt="Banner" width="100%">
</p>

## 🚀 Features

- Predicts winning probability in real-time
- Trained ML model on IPL dataset
- Interactive web app (Streamlit/Flask)
- Data preprocessing and feature engineering included

---

## 📂 Project Structure
├── Datasets/ # Raw and processed datasets
├── Models/ # Trained ML models
├── Dataset_Preparation.ipynb # Data cleaning & preprocessing
├── Model.ipynb # Model training and evaluation
├── app.py # Web app for prediction
├── requirements.txt # Dependencies
└── README.md # Project documentation
---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Deployment using Web App

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit / Flask (for deployment)

---

## 📊 Input Features

- Batting Team
- Bowling Team
- City
- Target Score
- Current Score
- Overs Completed
- Wickets Fallen

---

## 🎯 Output

- Winning Probability (%) of both teams

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/IPL_winning_Prediction.git
cd IPL_winning_Prediction
pip install -r requirements.txt
python app.py
streamlit run app.py
