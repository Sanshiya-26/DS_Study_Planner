from flask import Flask, render_template, request
import planner_logic  # Only needed if you're using logic for study planner

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Projects Page
@app.route("/projects")
def projects():
    return render_template("projects.html")

# Study Planner Page (Flask-based)
@app.route("/study-planner", methods=["GET", "POST"])
def study_planner():
    if request.method == "POST":
        # Example result; replace with planner_logic call if needed
        return render_template("plan.html", result="Study plan goes here")
    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
