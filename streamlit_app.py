import streamlit_app as st

st.title("Data Science Study Planner")
st.write("""
Welcome to the Data Science Study Planner!

This application helps you organize and track your learning resources effectively.
""")

# Example Data
resources = [
    {"topic": "Data Analysis", "status": "Completed"},
    {"topic": "Machine Learning", "status": "In Progress"},
    {"topic": "Deep Learning", "status": "Not Started"},
]

st.subheader("Study Plan")
for resource in resources:
    st.write(f"**{resource['topic']}** - {resource['status']}")

st.success("Keep up the good work!")
