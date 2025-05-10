import os 
import joblib
import csv
from datetime import datetime
from flask import Flask, render_template, request
from planner.planner_logic import generate_plan

# Paths to the model and encoders
MODEL_PATH = "model/model.pkl"
ENCODERS_PATH = "model/encoders.pkl"

# Load the model and encoders
model = joblib.load(MODEL_PATH)
style_encoder, topic_encoder = joblib.load(ENCODERS_PATH)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

@app.route("/plan", methods=["POST"])
def plan():
    user_input = request.form.to_dict()
    with open("user_logs.csv","a",newline='') as file:
        writer=csv.writer(file)
        writer.writerow([datetime.now()]+[str(value) for value in user_input.values()])
    try:
        hours_per_day = int(user_input.get("hours_per_day", 1))
        days_per_week = int(user_input.get("days_per_week", 1))
        preferred_style = user_input.get("preferred_style", "").lower()

        # Convert 'article' to 'reading'
        if preferred_style == "article":
            preferred_style = "reading"

        # Encode the style input (convert text to number)
        style_encoded = style_encoder.transform([preferred_style])[0]

        # Combine inputs into a list for prediction
        X_new = [[hours_per_day, days_per_week, style_encoded]]

        # Make prediction
        predicted_index = model.predict(X_new)[0]

        # Decode the prediction (convert number back to label)
        predicted_topic = topic_encoder.inverse_transform([predicted_index])[0]

        # Plan and resources
        plan = [f"Study {predicted_topic}"]
        resources = [
            ("Python Basics", "https://www.geeksforgeeks.org/data-science-tutorial/"),
            ("Data Analysis", "https://www.dataquest.io/blog/jupyter-notebook-tutorial/"),
            ("Data Visualization", "https://www.w3schools.com/python/matplotlib_intro.asp")
        ]

        return render_template("plan.html", plan=plan, resources=resources, name=user_input.get("goal", "Your Goal"))

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during prediction. Please try again."

if __name__ == "__main__":
    app.run(debug=True)


# web_app.py
print("Received Input Data:")
print("Hours per Day:", user_input.get("hours_per_day"))
print("Days per Week:", user_input.get("days_per_week"))
print("Preferred Style:", user_input.get("preferred_style"))
