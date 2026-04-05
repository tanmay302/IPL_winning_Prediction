# 🏏 IPL Winning Prediction

A Machine Learning project that predicts the winning probability of IPL teams based on match conditions like teams, score, overs, wickets, and venue.

---
<p align="center">
  <img src="Images/Screenshot 2026-04-03 175849.png" alt="Banner" height ="30%" width="100%">
</p>

## 🚀 Features

- Predicts winning probability in real-time
- Trained ML model on IPL dataset
- Interactive web app (Streamlit/Flask)
- Data preprocessing and feature engineering included

---

## 📂 Project Structure

- **Datasets/** → Raw and processed data  
- **Models/** → Trained ML models  
- **Dataset_Preparation.ipynb** → Data preprocessing  
- **Model.ipynb** → Model training  
- **app.py** → Web app  
- **requirements.txt** → Dependencies  
- **README.md** → Documentation  
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
## 🌐 Live Demo

Try the app live here 👇  
🔗 https://tanmay-ipl-winning-07.streamlit.app/

### 1. Clone the repository

```bash
git clone https://github.com/your-username/IPL_winning_Prediction.git
cd IPL_winning_Prediction
pip install -r requirements.txt
python app.py
streamlit run app.py
