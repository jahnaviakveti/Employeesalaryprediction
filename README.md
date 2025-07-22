
# 💻 Employee Salary Prediction App 💰

A machine learning-powered web application that predicts whether a person's salary exceeds \$50K using demographic and work-related features. The app is built with **Streamlit** for its simplicity, speed, and ease of deployment, and uses **ngrok** for secure sharing during development.

---

## 📊 Overview

This project leverages a classification model trained on the UCI Adult Census Income dataset to predict income categories (`<=50K` or `>50K`). The main aim is to provide an intuitive, real-time prediction interface for users to explore how various factors (like education, occupation, age, hours worked, etc.) influence salary predictions.

---

## 💡 Why Streamlit?

- ✅ **Lightweight**: No need for complex front-end frameworks  
- ✅ **Interactive Widgets**: Sliders, dropdowns, and text inputs are built-in  
- ✅ **Quick Setup**: Allows fast prototyping and testing of ML models  
- ✅ **Developer Friendly**: Pure Python, minimal boilerplate

Streamlit was chosen to rapidly convert the trained ML model into a web application with minimal overhead and maximum customization for inputs.

---

## 🌐 Ngrok for Local Deployment & Sharing

During development, **ngrok** was used to expose the local Streamlit app to the internet for testing and sharing with others.


## 📁 Project Files

| File/Folder             | Description |
|------------------------|-------------|
| `adult 3.csv`          | Cleaned version of the UCI Adult dataset used for model training and evaluation. |
| `app.py`               | Streamlit-based frontend application script for user interaction and prediction. |
| `salary_model.pkl.zip` | Zipped trained model file used to make predictions (unzip before running the app). |
| `label_encoders.pkl`   | Pre-fitted label encoders used to transform categorical inputs. |
| `log.txt`              | Log file that records input data and model predictions for debugging and tracking. |

## 🔮Features
- Real-time salary prediction using trained ML model
- Clean UI built with **Streamlit**
- Encodes categorical inputs automatically
- Logging mechanism for traceability
- Easy to deploy and run locally
  
## 🎯Prediction Target
Classifies whether an individual's salary is:
- `<=50K`
- `>50K`
Based on features such as:
- Age
- Education
- Occupation
- Workclass
- Hours per week
- etc.]
  
## 📌 Requirements
- Python 3.7+
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib
- LabelEncoder
- ngrok for sharing

## 📓 Notes
The dataset used is a cleaned version of the UCI Adult Income Dataset

The app was developed for educational purposes and demonstration of ML deployment techniques

Streamlit + ngrok combination makes it easy to test and share without full-scale cloud deployment 

## 🙋‍♀️ Author
Jahnavi Akveti


