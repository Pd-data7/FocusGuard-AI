# 🛡️ FocusGuard AI — Student Performance Intelligence System

FocusGuard AI is a Machine Learning-based student performance analysis and prediction system.

The application analyzes a student's daily lifestyle, study habits, stress level, sleep, social activities, and physical activity to estimate their **GPA** and provide personalized insights and improvement suggestions.

The project is built using **Python, Scikit-learn, Pandas, Plotly, and Streamlit**.

---

## 🚀 Live Project

🔗 Live Demo: **Coming Soon**

---

## 🎯 Problem Statement

Students often struggle to understand how their daily routine and lifestyle may be associated with their academic performance.

FocusGuard AI attempts to answer:

- How does study time relate to GPA?
- How does sleep relate to academic performance?
- How does stress relate to GPA?
- What does a student's daily routine look like?
- What GPA might be estimated from a given routine?
- How can a student improve their routine?

> **Important:** The model provides an estimate based on patterns in the dataset. It does not prove that a particular habit directly causes a change in GPA.

---

## ✨ Features

### 📊 Student Performance Dashboard

- Dataset overview
- Student statistics
- GPA distribution
- Lifestyle analysis
- Correlation analysis

### 🤖 GPA Prediction

Users can enter:

- Study hours per day
- Sleep hours per day
- Extracurricular hours
- Social hours
- Physical activity hours
- Stress level

The trained Machine Learning model then estimates GPA.

### 💡 Smart Insights

The application generates personalized observations based on the student's routine.

### 📈 What-If Lab

Users can modify their daily routine and compare the estimated GPA with an alternative routine.

### 🗓️ 7-Day Action Plan

The application generates a simple improvement plan based on the student's routine.

### 🧠 AI Explanation

The application provides information about which features are important to the trained model.

### 📊 Student Analytics

Interactive visualizations include:

- GPA distribution
- Study hours vs GPA
- Stress level vs GPA
- Feature correlation heatmap

---

## 🧠 Machine Learning

### Target Variable

```text
GPA
```

### Input Features

```text
Study_Hours_Per_Day
Extracurricular_Hours_Per_Day
Sleep_Hours_Per_Day
Social_Hours_Per_Day
Physical_Activity_Hours_Per_Day
Stress_Level
```

### Feature Engineering

The project creates additional features from the original lifestyle variables:

```text
Total_Tracked_Hours
Productive_Hours_Per_Day
Free_Time_Per_Day
```

### Models Tested

The project compares multiple regression algorithms:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best-performing model is selected based on evaluation metrics.

---

## 📏 Model Evaluation

The model is evaluated using three main metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between the predicted GPA and the actual GPA.

### RMSE — Root Mean Squared Error

Measures prediction error while giving more importance to larger errors.

### R² Score

Measures how much variation in GPA is explained by the model.

The current model performance is displayed inside the **Model Intelligence** section of the application.

---

## 🏗️ Project Architecture

```text
Student Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train / Test Split
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Saved ML Pipeline
      ↓
Streamlit Application
      ↓
Prediction + Insights + Analytics
```

---

## 📂 Project Structure

```text
DayProductivity_ML_Project/
│
├── FocusGuard_ML.ipynb
├── student_productivity.csv
├── focusguard_gpa_model.pkl
├── model_info.json
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |
| Plotly | Interactive charts |
| Streamlit | Web application |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd DayProductivity_ML_Project
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

## 📊 Dataset

The dataset contains student lifestyle and academic performance information.

Important variables include:

- Study hours
- Sleep hours
- Extracurricular activities
- Social activities
- Physical activity
- Stress level
- GPA

The dataset is used for educational and Machine Learning experimentation.

---

## 🔍 Example Workflow

A student can enter:

```text
Study Hours       → 5
Sleep Hours       → 7
Extracurricular   → 2
Social Hours      → 2
Physical Activity → 1
Stress Level      → Medium
```

FocusGuard processes these inputs through the trained Machine Learning pipeline and produces an estimated GPA along with routine insights.

---

## ⚠️ Limitations

This project has several limitations:

1. GPA prediction depends on the quality and distribution of the dataset.
2. The model does not capture every factor affecting academic performance.
3. Correlation does not imply causation.
4. The prediction should not be treated as an actual academic result.
5. The dataset may not represent every student population.

Therefore, FocusGuard should be considered an **educational Machine Learning project and decision-support tool**, not an official academic assessment system.

---

## 🔮 Future Improvements

Possible future improvements include:

- Larger and more diverse datasets
- Real student productivity data
- Time-series analysis
- Personalized recommendation models
- Explainable AI using SHAP
- Student progress tracking
- Login and student profiles
- Database integration
- Cloud deployment
- More advanced Machine Learning models
- Automatic weekly performance reports

---

## 👨‍💻 Author

**Pushpendu Das**

B.Tech Information Technology  
Haldia Institute of Technology

### Areas of Interest

- Machine Learning
- Data Science
- Data Analytics
- Artificial Intelligence
- C++ / DSA

---

## ⭐ Project Goal

FocusGuard AI was developed to demonstrate an end-to-end Machine Learning workflow:

```text
Data → EDA → Feature Engineering → ML → Evaluation → Deployment
```

The primary goal is to build a practical Machine Learning application that converts a trained model into a useful interactive product.