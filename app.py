import streamlit as st
import pickle
import numpy as np

df   = pickle.load(open("data/df.pkl",   "rb"))
pipe = pickle.load(open("data/pipe.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor", page_icon="📚")
st.title("Student Performance Predictor")
st.write("Enter student details to predict the expected performance score.")

col1, col2 = st.columns(2)

with col1:
    hours_studied  = st.number_input("Hours Studied per Day",
                                      min_value=0, max_value=24, value=6)
    previous_score = st.number_input("Previous Exam Score (out of 100)",
                                      min_value=0, max_value=100, value=70)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

with col2:
    sleep_hours = st.number_input("Sleep Hours per Day",
                                   min_value=0, max_value=24, value=7)
    practice    = st.number_input("Sample Question Papers Practiced",
                                   min_value=0, max_value=10, value=3)

if st.button("Predict Performance Score", use_container_width=True):

    # encode extracurricular activities
    # this must be OUTSIDE the if/else so it always runs
    extra_val = "Yes" if extracurricular == "Yes" else "No"

    query = np.array([[hours_studied, previous_score, extra_val,
                       sleep_hours, practice]])

    predicted_score = pipe.predict(query)[0]
    predicted_score = round(float(predicted_score), 2)

    st.success(f"Predicted Performance Score: {predicted_score} / 100")

    if predicted_score >= 80:
        st.info("Excellent performance expected.")
    elif predicted_score >= 60:
        st.info("Good performance expected.")
    else:
        st.warning("Performance can be improved with more study hours.")
