# =========================================================================
# SUBJECT: DATA SCIENCE, AI & MACHINE LEARNING
# WEB APP: Student Performance & Academic Risk Predictor (Fully Corrected)
# =========================================================================

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Student Risk Analyzer", page_icon="🎓", layout="centered")

st.title("🎓 Intelligent Student Performance & Academic Risk Predictor")
st.write("This AI-powered system analyzes student data to predict academic categories and find students at risk.")
st.markdown("---")

# 2. Data Preparation Function (ट्रेनिंग पहले होगी ताकि सारे टूल्स तैयार रहें)
@st.cache_data
def load_and_train_model():
    dataset = pd.read_csv("student_performance_dataset.csv")
    
    np.random.seed(42)
    n_extra = 190
    attendance = np.random.randint(52, 99, n_extra)
    study_hours = np.random.randint(5, 26, n_extra)
    prev_grades = np.random.randint(42, 96, n_extra)
    assignments = np.random.randint(56, 99, n_extra)
    participation = np.random.randint(3, 11, n_extra)
    
    score = (attendance * 0.3) + (prev_grades * 0.3) + (study_hours * 1.4) + (assignments * 0.1)
    categories = []
    for s in score:
        if s > 72: categories.append("Good")
        elif s > 55: categories.append("Average")
        else: categories.append("At Risk")
        
    df_extra = pd.DataFrame({
        'StudentID': [f"STU_{i}" for i in range(1011, 1011 + n_extra)],
        'Attendance_Rate': attendance,
        'Study_Hours_Weekly': study_hours,
        'Previous_Grade': prev_grades,
        'Assignments_Completed': assignments,
        'Participation_Score': participation,
        'Performance_Category': categories
    })
    
    final_df = pd.concat([dataset, df_extra], ignore_index=True)
    
    encoder = LabelEncoder()
    final_df['Performance_Label'] = encoder.fit_transform(final_df['Performance_Category'])
    
    X = final_df[['Attendance_Rate', 'Study_Hours_Weekly', 'Previous_Grade', 'Assignments_Completed', 'Participation_Score']]
    y = final_df['Performance_Label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, encoder

# मॉडल लोड करना (अब 'scaler' कंप्यूटर की मेमोरी में आ चुका है)
model, scaler, encoder = load_and_train_model()

# 3. Bulk Data Upload Feature (फ़ाइल अपलोडर अब यहाँ बिल्कुल सही काम करेगा)
st.subheader(" Upload Class Dataset (Bulk Analysis)")
uploaded_file = st.file_uploader("Choose a CSV file containing student records", type=["csv"])

if uploaded_file is not None:
    bulk_data = pd.read_csv(uploaded_file)
    st.write("###  Uploaded Student Data:")
    st.dataframe(bulk_data.head(10))
    
    if st.button(" Run AI Analysis on Whole Class"):
        X_bulk = bulk_data[['Attendance_Rate', 'Study_Hours_Weekly', 'Previous_Grade', 'Assignments_Completed', 'Participation_Score']]
        X_bulk_scaled = scaler.transform(X_bulk)
        
        bulk_preds = model.predict(X_bulk_scaled)
        bulk_data['AI_Predicted_Status'] = encoder.inverse_transform(bulk_preds)
        
        st.success(" Analysis Completed for the whole class!")
        st.dataframe(bulk_data[['StudentID', 'Attendance_Rate', 'Previous_Grade', 'AI_Predicted_Status']])
st.markdown("---")

# 4. User Inputs via Web UI (एक छात्र का डेटा चेक करने के लिए)
st.subheader(" Enter Student Academic Indicators (Single Student Check):")

col1, col2 = st.columns(2)

with col1:
    attendance_input = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=75, step=1)
    study_hours_input = st.slider("Weekly Study Hours", min_value=0, max_value=40, value=12, step=1)
    participation_input = st.slider("Class Participation Score (1-10)", min_value=1, max_value=10, value=6, step=1)

with col2:
    prev_grade_input = st.number_input("Previous Exam Score (%)", min_value=0, max_value=100, value=65)
    assignments_input = st.number_input("Assignments Completed (%)", min_value=0, max_value=100, value=80)

st.markdown("---")

# 5. Predict Button & Logic for Single Student
if st.button(" Analyze & Predict Performance (Single Student)", type="primary"):
    features = np.array([[attendance_input, study_hours_input, prev_grade_input, assignments_input, participation_input]])
    scaled_features = scaler.transform(features)
    
    pred_code = model.predict(scaled_features)
    result_category = encoder.inverse_transform(pred_code)
    
    st.subheader(" Analysis Result:")
    if result_category == "Good":
        st.success(f"Predicted Status: **{result_category[0]}** \n\nThe student is performing well.")
    elif result_category == "Average":
        st.info(f"Predicted Status: **{result_category[0]}** \n\nThe student is on track but has scope for improvement.")
    else:
        st.error(f"Predicted Status: **{result_category[0]}  (Academic Risk)**\n\nWarning: This student requires immediate support.")
        
    st.metric(label="Calculated Risk Status", value=str(result_category[0]))