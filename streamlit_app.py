import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
import csv

# Load model and encoders
model = joblib.load("model/model.pkl")
style_encoder, topic_encoder = joblib.load("model/encoders.pkl")

# Streamlit UI
st.set_page_config(page_title="DS Study Planner", layout="centered")
st.title("📘 Data Science Study Planner")
st.markdown("Plan your path to becoming a Data Scientist based on your study habits.")

# Input fields
goal = st.text_input("Your Learning Goal (e.g., Become a Data Analyst)")
hours_per_day = st.slider("Hours you can study per day", 1, 12, 2)
days_per_week = st.slider("Days you can study per week", 1, 7, 5)
preferred_style = st.selectbox("Preferred Study Style", ["reading", "video", "interactive", "article"])

# Adjust style for encoding
style_input = "reading" if preferred_style == "article" else preferred_style
style_encoded = style_encoder.transform([style_input])[0]

# Prediction trigger
if st.button("🧠 Recommend Study Topic"):
    # Prepare data
    X_new = pd.DataFrame([{
        "hours_per_day": hours_per_day,
        "days_per_week": days_per_week,
        "style_encoded": style_encoded
    }])

    predicted_index = model.predict(X_new)[0]
    predicted_topic = topic_encoder.inverse_transform([predicted_index])[0]

    # Save to CSV
    with open("user_logs.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), hours_per_day, days_per_week, preferred_style, predicted_topic])

    # Output
    st.success(f"🎯 Recommended Topic: **{predicted_topic}**")

    st.subheader("📚 Recommended Resources")
    st.markdown("- [Python Basics](https://www.geeksforgeeks.org/data-science-tutorial/)")
    st.markdown("- [Data Analysis](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)")
    st.markdown("- [Data Visualization](https://www.w3schools.com/python/matplotlib_intro.asp)")
