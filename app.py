import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("Models/best_model.pkl", "rb"))

# ------------------ LOAD ENCODERS ------------------
le_dict = pickle.load(open("Models/encoder.pkl", "rb"))

# ------------------ UI ------------------
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
    overs = st.number_input("Overs Completed", min_value=0.0)
    wickets = st.number_input("Wickets Fallen", min_value=0)

# ------------------ VALIDATION ------------------
if batting_team == bowling_team:
    st.error("Batting and Bowling teams cannot be same ❌")

# ------------------ CALCULATIONS ------------------
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_left = 10 - wickets

crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

total_runs_x = target  # important feature

# ------------------ PREDICTION ------------------
if st.button("Predict Probability"):

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

    # Predict
    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)[0]

    # ------------------ MAP RESULT ------------------
    if prediction[0] == 1:
        predicted_team = batting_team
    else:
        predicted_team = bowling_team

    st.subheader("🏆 Prediction Result")
    st.success(f"Predicted Winner: {predicted_team}")

    # ------------------ PROBABILITY FIX ------------------
    batting_prob = probs[1] * 100
    bowling_prob = probs[0] * 100

    # avoid 0%
    batting_prob = max(batting_prob, 1)
    bowling_prob = max(bowling_prob, 1)

    # normalize
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

    colors = ['#FF4B4B', '#00C851']  # red vs green

    ax.barh(teams_plot, probs_plot, color=colors)

    for i, v in enumerate(probs_plot):
        ax.text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_title("Win Probability")
    ax.set_xlabel("Probability (%)")

    st.pyplot(fig)

    # ------------------ PROGRESS BARS ------------------
    st.subheader("📊 Probability Breakdown")

    st.write(bowling_team)
    st.progress(bowling_prob / 100)

    st.write(batting_team)
    st.progress(batting_prob / 100)