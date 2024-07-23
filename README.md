# Student Performance Predictor

A machine learning project that predicts student performance scores based on study habits and lifestyle factors — built with Linear Regression achieving 98% R2 score.

---

## About

This project analyzes student data to understand which factors most influence academic performance and builds a prediction model. A Streamlit web app allows users to enter student details and get an instant performance score prediction.

---

## Project Structure

```
student-performance-predictor/
│
├── src/
│   ├── 01_data_analysis.py    # EDA, visualizations, correlation analysis
│   └── 02_model_training.py   # train and compare models, save pipeline
│
├── data/
│   └── student_performance.csv   # raw dataset (10000 students)
│
├── app.py                     # Streamlit web app
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

- 10000 student records
- 6 columns: Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Sample Question Papers Practiced, Performance Index

| Feature | Type | Description |
|---------|------|-------------|
| Hours Studied | Numeric | Daily study hours |
| Previous Scores | Numeric | Scores from previous exams |
| Extracurricular Activities | Categorical | Yes / No |
| Sleep Hours | Numeric | Daily sleep hours |
| Sample Question Papers Practiced | Numeric | Number of practice papers |
| Performance Index | Numeric | Target variable (0-100) |

---

## EDA Findings

- Previous Scores and Hours Studied have the highest correlation with Performance Index
- Students with extracurricular activities perform slightly better on average
- Sleep hours show a moderate positive correlation with performance
- No missing values found in the dataset

---

## Model Results

Compared 3 regression algorithms:

| Model | R2 Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.9888 | 1.47 |
| Random Forest | 0.9851 | 1.73 |
| Gradient Boosting | 0.9789 | 1.89 |

Best model: **Linear Regression**
- R2 Score: 98.88% — model explains 98.88% of performance variance
- MAE: 1.47 — average prediction error of 1.47 points out of 100

Linear regression outperforms ensemble methods here because the relationship between features and performance is largely linear.

---

## Bug Fixed

The original app had a critical indentation bug — the prediction code was inside the `else` block, so the predict button only worked when "No" was selected for extracurricular activities. When "Yes" was selected nothing happened.

```python
# wrong — query inside else block
if extra == "Yes":
    extra = 1
else:
    extra = 0
    query = ...       # only runs for "No"
    pipe.predict(...)

# fixed — query outside if/else, always runs
extra_val = "Yes" if extra == "Yes" else "No"
query = np.array([[hours, score, extra_val, sleep, practice]])
pipe.predict(query)
```

---

## How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Arifkhan171/student-performance-predictor.git
cd student-performance-predictor
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline:**
```bash
python src/01_data_analysis.py
python src/02_model_training.py
```

**4. Launch web app:**
```bash
streamlit run app.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## Author

**Arif Khan**
Final Year CS Student | University of Loralai | ML / Deep Learning / Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)
