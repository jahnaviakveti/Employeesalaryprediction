import streamlit as st
import pandas as pd
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/stack-dollar-blue-background-with-earning-profit_903752-4857.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = joblib.load("salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.markdown(
    "<h1 style='color: black; text-align: center;'>💻Employee Salary Predictor💰</h1>",
    unsafe_allow_html=True
)

def user_input():
    st.sidebar.header("Input")
    age = st.sidebar.slider("Age", 18, 90, 30)
    workclass = st.sidebar.selectbox("Workclass", label_encoders['workclass'].classes_)
    education = st.sidebar.selectbox("Education", label_encoders['education'].classes_)
    marital_status = st.sidebar.selectbox("Marital Status", label_encoders['marital-status'].classes_)
    occupation = st.sidebar.selectbox("Occupation", label_encoders['occupation'].classes_)
    relationship = st.sidebar.selectbox("Relationship", label_encoders['relationship'].classes_)
    race = st.sidebar.selectbox("Race", label_encoders['race'].classes_)
    sex = st.sidebar.selectbox("Sex", label_encoders['sex'].classes_)
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
    hours = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", label_encoders['native-country'].classes_)
    edu_num = st.sidebar.slider("Education Number", 1, 16, 10)
    fnlwgt = st.sidebar.number_input("Final Weight", 10000, 1000000, 300000)

    data = {
        "age": age,
        "workclass": label_encoders["workclass"].transform([workclass])[0],
        "fnlwgt": fnlwgt,
        "education": label_encoders["education"].transform([education])[0],
        "educational-num": edu_num,
        "marital-status": label_encoders["marital-status"].transform([marital_status])[0],
        "occupation": label_encoders["occupation"].transform([occupation])[0],
        "relationship": label_encoders["relationship"].transform([relationship])[0],
        "race": label_encoders["race"].transform([race])[0],
        "sex": label_encoders["sex"].transform([sex])[0],
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "native-country": label_encoders["native-country"].transform([native_country])[0]
    }

    return pd.DataFrame([data])

input_df = user_input()
pred = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][pred]
label = label_encoders["income"].inverse_transform([pred])[0]

st.markdown(
    f"""
    <style>
    .result-box {{
        background: linear-gradient(145deg, #ffffff, #e6e6e6);
        border: 2px solid #ccc;
        padding: 30px;
        border-radius: 20px;
        width: 60%;
        margin: 40px auto;
        text-align: center;
        box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    .result-box:hover {{
        transform: scale(1.02);
        box-shadow: 10px 10px 25px rgba(0, 0, 0, 0.2);
    }}

    .result-title {{
        font-size: 28px;
        font-weight: 800;
        color: #222;
        margin-bottom: 25px;
    }}

    .result-text {{
        font-size: 20px;
        color: #222;
        margin: 10px 0;
    }}

    .result-tag {{
        display: inline-block;
        background-color: #111;
        color: #00FF88;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        margin-left: 10px;
    }}
    </style>

    <div class="result-box">
        <div class="result-title">📜 Result 📜</div>
        <div class="result-text">
            Predicted Income: <span class="result-tag">{label}</span>
        </div>
        <div class="result-text">
            Confidence: <span class="result-tag">{proba*100:.2f}%</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
