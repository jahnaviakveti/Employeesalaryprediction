# Employee salary prediction
A machine learning based web application built using **Streamlit** to predict whether a person's salary exceeds \$50K based on demographic and employment-related features.

## Overview

This project uses the **Adult Census Income Dataset** to train a classification model that predicts income levels. The app allows users to input details such as age, education, occupation, hours worked, etc., and returns a prediction in real-time.

## 📁 Project Files

| File/Folder             | Description |
|------------------------|-------------|
| `adult 3.csv`          | Cleaned version of the UCI Adult dataset used for model training and evaluation. |
| `app.py`               | Streamlit-based frontend application script for user interaction and prediction. |
| `salary_model.pkl.zip` | Zipped trained model file used to make predictions (unzip before running the app). |
| `label_encoders.pkl`   | Pre-fitted label encoders used to transform categorical inputs. |
| `log.txt`              | Log file that records input data and model predictions for debugging and tracking. |

## Features
- Real-time salary prediction using trained ML model
- Clean UI built with **Streamlit**
- Encodes categorical inputs automatically
- Logging mechanism for traceability
- Easy to deploy and run locally
- 
## Prediction Target
Classifies whether an individual's salary is:
- `<=50K`
- `>50K`
Based on features such as:
- Age
- Education
- Occupation
- Workclass
- Hours per week
- etc.


