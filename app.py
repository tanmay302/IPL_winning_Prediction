import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------ PATH SETUP ------------------
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
    st.error(f"❌ Error loading model: {e}")
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
balls_left = int(120 - (overs * 6))

crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# ------------------ PREDICTION ------------------
if st.button("Predict Probability"):

    try:
        # ✅ MATCH TRAINING FEATURES EXACTLY
        input_df = pd.DataFrame([{
            'batting_team': int(le_dict['batting_team'].transform([batting_team])[0]),
            'bowling_team': int(le_dict['bowling_team'].transform([bowling_team])[0]),
            'city': int(le_dict['city'].transform([city])[0]),
            'runs_left': float(runs_left),
            'balls_left': int(balls_left),
            'wickets': int(wickets),   # ✅ FIXED (not wickets_left)
            'total_runs_x': float(target),
            'crr': float(crr),
            'rrr': float(rrr)
        }])

        # ✅ EXACT ORDER (VERY IMPORTANT)
        input_df = input_df[[
            'batting_team',
            'bowling_team',
            'city',
            'runs_left',
            'balls_left',
            'wickets',
            'total_runs_x',
            'crr',
            'rrr'
        ]]

        probs = model.predict_proba(input_df)[0]
        classes = model.classes_

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.stop()

    # ------------------ PROBABILITY MAPPING ------------------
    try:
        batting_index = list(classes).index(1)
        bowling_index = list(classes).index(0)

        batting_prob = probs[batting_index] * 100
        bowling_prob = probs[bowling_index] * 100

    except:
        batting_prob = probs[1] * 100
        bowling_prob = probs[0] * 100

    # Normalize
    total = batting_prob + bowling_prob
    batting_prob = (batting_prob / total) * 100
    bowling_prob = (bowling_prob / total) * 100

    predicted_team = batting_team if batting_prob > bowling_prob else bowling_team

    # ------------------ OUTPUT ------------------
    st.subheader("🏆 Prediction Result")
    st.success(f"Predicted Winner: {predicted_team}")

    st.subheader("📊 Win Probability")
    st.write(f"{batting_team}: {batting_prob:.2f}%")
    st.write(f"{bowling_team}: {bowling_prob:.2f}%")

    # ------------------ GRAPH ------------------
    fig, ax = plt.subplots(figsize=(8,5))

    teams_plot = [bowling_team, batting_team]
    probs_plot = [bowling_prob, batting_prob]

    colors = ["#4E79A7", "#E15759"]

    ax.barh(teams_plot, probs_plot, color=colors)

    for i, v in enumerate(probs_plot):
        ax.text(v + 1, i, f"{v:.2f}%", va='center', fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_title("Win Probability")
    ax.set_xlabel("Probability (%)")

    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    st.pyplot(fig)

    # ------------------ PROGRESS BARS ------------------
    st.subheader("📊 Probability Breakdown")

    st.write(bowling_team)
    st.progress(int(bowling_prob))

    st.write(batting_team)
    st.progress(int(batting_prob))