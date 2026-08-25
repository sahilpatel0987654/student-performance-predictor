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
st.markdown("""
    <style>
    /* 1. रॉयल ब्लू और लग्जरी डार्क ग्रेडिएंट बैकग्राउंड */
    .stApp {
        background: linear-gradient(135deg, #060a17 0%, #0d1527 40%, #1e1136 100%) !important;
        color: #e2e8f0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* 2. मुख्य हेडिंग को Apple स्टाइल में साफ़ और चमकदार बनाना */
    .stHeading h1, .main-title {
        font-size: 40px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 10px;
    }
    
    /* 3. साइडबार को प्रीमियम डार्क और फ्रॉस्टेड ग्लास लुक देना */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 17, 34, 0.85) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* 4. बटन्स को बेहद मॉडर्न, स्मूथ और आकर्षक बनाना */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important; /* गोल और स्मूथ किनारे */
        padding: 12px 28px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* बटन पर माउस ले जाने पर स्मूथ ग्लो इफेक्ट */
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #60a5fa 0%, #4f46e5 100%) !important;
    }
    
    /* 5. कार्ड्स और इनपुट बॉक्स को 'कांच' (Glassmorphism) जैसा लुक देना */
    div[data-testid="metric-container"], .element-container, div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important; /* प्रीमियम कट्स */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        padding: 20px !important;
        transition: all 0.3s ease;
    }
    
    /* कार्ड्स पर माउस ले जाने पर चमकना */
    div[data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
    
    /* 6. टेक्स्ट और स्लाइडर्स की स्टाइलिंग */
    label p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    input, select {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
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
