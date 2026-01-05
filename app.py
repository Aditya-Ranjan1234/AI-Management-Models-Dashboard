import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(page_title="AI Management Dashboard", layout="wide")

st.title("AI-Driven Managerial Transformation Dashboard")
st.markdown("""
This dashboard demonstrates three AI models supporting managerial roles in hybrid work environments:
1.  **Productivity Predictor** (RQ1)
2.  **Attrition Risk Model** (RQ2)
3.  **Task Recommender** (RQ3)
""")

# Load Models
@st.cache_resource
def load_models():
    prod_model = joblib.load('Productivity_Predictor/productivity_model.pkl')
    att_model = joblib.load('Attrition_Risk_Model/attrition_model.pkl')
    task_model = joblib.load('Task_Recommender/task_model.pkl')
    return prod_model, att_model, task_model

try:
    prod_model, att_model, task_model = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Productivity Predictor", "Attrition Risk", "Task Recommender"])

# --- Tab 1: Productivity Predictor ---
with tab1:
    st.header("Productivity Predictor (RQ1)")
    st.write("Predicts employee productivity levels in distributed work settings.")
    
    # Load Data for Viz
    df_prod = pd.read_csv('Productivity_Predictor/data/remote_work_productivity.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Sample")
        st.dataframe(df_prod.head())
        
    with col2:
        st.subheader("Productivity Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df_prod, x='Productivity_Score', ax=ax, palette="viridis")
        st.pyplot(fig)

    st.divider()
    st.subheader("Make a Prediction")
    
    # Inputs
    c1, c2, c3, c4 = st.columns(4)
    hours = c1.number_input("Hours Worked (Weekly)", min_value=10, max_value=80, value=40)
    meetings = c2.number_input("Meeting Hours", min_value=0, max_value=40, value=10)
    remote_days = c3.slider("Remote Days per Week", 0, 5, 3)
    wellbeing = c4.slider("Well-Being Score (1-10)", 1, 10, 7)
    
    if st.button("Predict Productivity"):
        # Create input array
        input_data = pd.DataFrame([[hours, meetings, remote_days, wellbeing]], 
                                  columns=['Hours_Worked', 'Meeting_Hours', 'Remote_Days', 'Well_Being_Score'])
        prediction = prod_model.predict(input_data)[0]
        
        if prediction == "High":
            st.success(f"Predicted Productivity: **{prediction}** 🚀")
        elif prediction == "Medium":
            st.warning(f"Predicted Productivity: **{prediction}** ⚖️")
        else:
            st.error(f"Predicted Productivity: **{prediction}** ⚠️")

# --- Tab 2: Attrition Risk ---
with tab2:
    st.header("Attrition Risk Model (RQ2)")
    st.write("Identifies employees at risk of leaving to improve organizational resilience.")
    
    df_att = pd.read_csv('Attrition_Risk_Model/data/ibm_hr_attrition.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Sample")
        st.dataframe(df_att.head())
    with col2:
        st.subheader("Attrition Rate")
        fig, ax = plt.subplots()
        df_att['Attrition'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'])
        st.pyplot(fig)
        
    st.divider()
    st.subheader("Assess Retention Risk")
    
    # Inputs
    # Features: Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance, YearsAtCompany
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", 18, 70, 30)
    dist = c2.number_input("Distance From Home (km)", 1, 50, 5)
    daily_rate = c3.number_input("Daily Rate ($)", 100, 2000, 800)
    years = c4.number_input("Years at Company", 0, 40, 5)
    
    c5, c6, c7, c8 = st.columns(4)
    edu = c5.selectbox("Education Level", [1, 2, 3, 4, 5])
    env_sat = c6.slider("Env. Satisfaction (1-4)", 1, 4, 3)
    job_sat = c7.slider("Job Satisfaction (1-4)", 1, 4, 3)
    work_life = c8.slider("Work Life Balance (1-4)", 1, 4, 3)
    
    if st.button("Predict Attrition Risk"):
        input_df = pd.DataFrame([[age, daily_rate, dist, edu, env_sat, job_sat, work_life, years]], 
                                columns=['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany'])
        
        pred = att_model.predict(input_df)[0]
        prob = att_model.predict_proba(input_df)[0][1]
        
        st.metric("Attrition Probability", f"{prob:.1%}")
        
        if pred == "Yes":
            st.error("Risk Assessment: **High Risk of Leaving** 🚨")
        else:
            st.success("Risk Assessment: **Low Risk** ✅")

# --- Tab 3: Task Recommender ---
with tab3:
    st.header("Task Recommender (RQ3)")
    st.write("Classifies tasks for Human-AI Collaboration: Automate, Augment, or Human-Only.")
    
    df_task = pd.read_csv('Task_Recommender/data/task_data.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Task Examples")
        st.dataframe(df_task[['Task_Description', 'Category']].head())
    with col2:
        st.subheader("Task Category Distribution")
        fig, ax = plt.subplots()
        sns.countplot(y='Category', data=df_task, ax=ax, palette="magma")
        st.pyplot(fig)
        
    st.divider()
    st.subheader("Classify a New Task")
    
    task_input = st.text_area("Enter Task Description", "Draft quarterly financial report for board meeting")
    
    if st.button("Recommend Action"):
        pred_cat = task_model.predict([task_input])[0]
        
        st.subheader(f"Recommendation: **{pred_cat}**")
        
        if pred_cat == "Automate":
            st.info("🤖 This task is suitable for full AI automation.")
        elif pred_cat == "Augment":
            st.warning("🤝 This task is best for Human-AI Augmentation.")
        else:
            st.success("👤 This task requires Human judgment and empathy.")
