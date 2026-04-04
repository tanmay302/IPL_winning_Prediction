import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ PATH FIX (VERY IMPORTANT) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "Models", "best_model.pkl")
encoder_path = os.path.join(BASE_DIR, "Models", "encoder.pkl")

# ------------------ LOAD MODEL ------------------
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        le_dict = pickle.load(f)

except Exception as e:
    st.error("❌ Error loading model or encoder")
    st.stop()

# ------------------ UI ------------------
st.set_page_config(page_title="IPL Predictor", layout="wide")

st.title("🏏 IPL Win Predictor")
st.info("Predict match winner based on live match situation 🔥")

teams = le_dict['batting_team'].classes_
cities = le_dict['city'].classes_

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)
    city = st.selectbox("City", cities)

with col2:
    target = st.number_input("Target", min_value=1)
    score = st.number_input("Current Score", min_value=0)
    overs = st.number_input("Overs Completed", min_value=0.1)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

# ------------------ VALIDATION ------------------
if batting_team == bowling_team:
    st.error("❌ Batting and Bowling teams cannot be same")
    st.stop()

# ------------------ CALCULATIONS ------------------
runs_left = target - score
balls_left = 120 - int(overs * 6)
wickets_left = 10 - wickets

crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

total_runs_x = target

# ------------------ PREDICTION ------------------
if st.button("Predict Probability"):

    try:
        input_data = np.array([[ 
            le_dict['batting_team'].transform([batting_team])[0],
            le_dict['bowling_team'].transform([bowling_team])[0],
            le_dict['city'].transform([city])[0],
            runs_left,
            balls_left,
            wickets_left,
            total_runs_x,
            crr,
            rrr
        ]])

        probs = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)

    except Exception as e:
        st.error("❌ Prediction error. Check model inputs.")
        st.stop()

    # ------------------ RESULT ------------------
    predicted_team = batting_team if prediction[0] == 1 else bowling_team

    st.subheader("🏆 Prediction Result")
    st.success(f"Predicted Winner: {predicted_team}")

    # ------------------ PROBABILITY ------------------
    batting_prob = probs[1] * 100
    bowling_prob = probs[0] * 100

    # Normalize safely
    total = batting_prob + bowling_prob
    batting_prob = (batting_prob / total) * 100
    bowling_prob = (bowling_prob / total) * 100

    st.subheader("📊 Win Probability")
    st.write(f"{batting_team}: {batting_prob:.2f}%")
    st.write(f"{bowling_team}: {bowling_prob:.2f}%")

    # ------------------ GRAPH ------------------
    fig, ax = plt.subplots(figsize=(8,5))

    teams_plot = [bowling_team, batting_team]
    probs_plot = [bowling_prob, batting_prob]

    colors = ["#4E79A7", "#E15759"]  # professional colors

    ax.barh(teams_plot, probs_plot, color=colors)

    for i, v in enumerate(probs_plot):
        ax.text(v + 1, i, f"{v:.2f}%", va='center', fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_title("Win Probability")
    ax.set_xlabel("Probability (%)")

    # Clean look
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    st.pyplot(fig)

    # ------------------ PROGRESS BARS ------------------
    st.subheader("📊 Probability Breakdown")

    st.write(bowling_team)
    st.progress(int(bowling_prob))

    st.write(batting_team)
    st.progress(int(batting_prob))