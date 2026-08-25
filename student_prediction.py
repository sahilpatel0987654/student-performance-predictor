import subprocess
import sys

# ऑटो-इंस्टॉलेशन कोड: यह कोड वेबसाइट पर बिना किसी एरर के लाइब्रेरीज़ खुद इंस्टॉल कर देगा
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception:
        pass

# ज़रूरी लाइब्रेरीज़ को ज़बरदस्ती लोड करना
for lib in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']:
    try:
        __import__(lib)
    except ImportError:
        install(lib)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- पेज का टाइटल और लेआउट सेट करना ---
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Intelligent Student Performance Prediction & Academic Risk Analysis")
st.write("This web application uses **Machine Learning (Random Forest)** to predict student performance and analyze academic risk early.")

# --- बैकएंड: डेटासेट तैयार करना और MODEL ट्रेन करना ---
@st.cache_data
def load_and_train():
    np.random.seed(42)
    n_students = 200
    attendance = np.random.randint(50, 100, n_students)     
    study_hours = np.random.randint(1, 10, n_students)       
    previous_score = np.random.randint(40, 100, n_students)  
    assignment_score = np.random.randint(30, 100, n_students)

    academic_risk = []
    for i in range(n_students):
        if attendance[i] < 75 or previous_score[i] < 50:
            academic_risk.append(1) 
        else:
            academic_risk.append(0)

    df = pd.DataFrame({
        'Attendance': attendance,
        'Study_Hours': study_hours,
        'Previous_Score': previous_score,
        'Assignment_Score': assignment_score,
        'Academic_Risk': academic_risk
    })
    
    X = df[['Attendance', 'Study_Hours', 'Previous_Score', 'Assignment_Score']]
    y = df['Academic_Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return df, model, acc

df, model, accuracy = load_and_train()

# --- साइडबार में मॉडल की एक्युरेसी दिखाना ---
st.sidebar.header("📊 Model Metrics")
st.sidebar.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- वेब पेज को दो हिस्सों (Tabs) में बांटना ---
tab1, tab2 = st.tabs(["🖥️ Project Dashboard & Visuals", "🔮 Live Risk Predictor Form"])

with tab1:
    st.header("📊 Student Dataset & Data Analytics")
    st.subheader("📋 Sample Student Dataset (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📈 Graphical Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Student Distribution Chart**")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x='Academic_Risk', data=df, palette='Set2', ax=ax1)
        st.pyplot(fig1)
        
    with col2:
        st.write("**Attendance vs Previous Score Scatter Plot**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        sns.scatterplot(x='Attendance', y='Previous_Score', hue='Academic_Risk', data=df, palette='coolwarm', ax=ax2)
        st.pyplot(fig2)

with tab2:
    st.header("🔮 Test Academic Risk for a New Student")
    with st.form("prediction_form"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            input_attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=85)
            input_study = st.slider("Daily Study Hours", min_value=0, max_value=24, value=4)
        with col_in2:
            input_prev = st.number_input("Previous Exam Score (Out of 100)", min_value=0, max_value=100, value=70)
            input_assign = st.number_input("Assignment Score (Out of 100)", min_value=0, max_value=100, value=75)
            
        submit_btn = st.form_submit_button("Predict Academic Risk")
        
    if submit_btn:
        new_data = pd.DataFrame([[input_attendance, input_study, input_prev, input_assign]], 
                                columns=['Attendance', 'Study_Hours', 'Previous_Score', 'Assignment_Score'])
        prediction = model.predict(new_data)
        
        st.subheader("📢 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ **HIGH ACADEMIC RISK!** This student needs immediate mentorship.")
        else:
            st.success("✅ **SAFE / NO RISK.** The student is performing well.")
