import streamlit as st
from PIL import Image

# Load Avatar
avatar = Image.open("static/avatar.png")

# Portfolio Sections
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Projects", "Contact"])

# Home Section
if selection == "Home":
    st.image(avatar, width=200)
    st.title("Sanshiya Ramesh")
    st.write("""
        Hi, I'm a passionate Data Scientist. Welcome to my portfolio!  
        Explore my projects and connect with me.
    """)

# Projects Section
elif selection == "Projects":
    st.subheader("Data Science Study Planner")
    st.write("An interactive study planner for data science topics.")
    st.write("[View Demo (Streamlit)](https://sanshiya26-ds-study-planner.streamlit.app/)")

# Contact Section
else:
    st.subheader("Contact")
    st.write("Email: sanshiya@example.com")
    st.write("[LinkedIn](https://linkedin.com/in/sanshiya)")

