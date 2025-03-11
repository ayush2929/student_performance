import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# MongoDB Connection
uri = "mongodb+srv://ayush_bishnoi:ayush1234@cluster0.yiczw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Student']
collection = db['student_pred']

def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocess_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocess_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.set_page_config(page_title="Student Performance Prediction", layout="centered")
    st.title("🎓 Student Performance Predictor")
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 12px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.write("Fill in your details to predict your performance:")

    col1, col2 = st.columns(2)

    with col1:
        hour_studied = st.number_input("📚 Hours Studied", min_value=1, max_value=10, value=5)
        previous_score = st.number_input("📊 Previous Score", min_value=40, max_value=100, value=70)
        sleeping_hour = st.number_input("😴 Sleeping Hours", min_value=4, max_value=10, value=7)
    
    with col2:
        extra = st.selectbox("🎯 Extracurricular Activities", ['Yes', "No"])
        number_of_paper_solved = st.number_input("📝 Number of Question Papers Solved", min_value=0, max_value=10, value=5)

    if st.button("🚀 Predict Your Score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }

        prediction = predict_data(user_data)
        user_data['prediction'] = float(prediction)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}

        collection.insert_one(user_data)
        st.success(f"🎉 Your predicted score is: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()