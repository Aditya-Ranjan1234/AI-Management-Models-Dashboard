# AI-Driven Managerial Transformation: Project Report

This document provides a comprehensive overview of the AI models developed to support managerial roles in hybrid work environments. It details the algorithms, datasets, performance metrics, and strategic impact of each component.

---

## 1. Productivity Predictor (RQ1)

### **Objective**
To analyze and predict employee productivity levels in distributed/hybrid work settings, enabling managers to identify high performers and those needing support.

### **Dataset**
*   **Source:** Synthetic data mimicking the [Remote Work Productivity Dataset (Kaggle)](https://www.kaggle.com/datasets/melodyyiphoiching/remote-working-survey).
*   **Features:**
    *   `Hours_Worked`: Weekly hours logged.
    *   `Meeting_Hours`: Hours spent in virtual meetings.
    *   `Remote_Days`: Number of days working from home per week.
    *   `Well_Being_Score`: Self-reported wellness (1-10).
*   **Target:** `Productivity_Score` (Categorical: High, Medium, Low).

### **Algorithm & Training**
*   **Model:** **Random Forest Classifier**
*   **Why Random Forest?** It handles non-linear relationships well (e.g., too many meetings might drastically reduce productivity despite high work hours) and provides feature importance.
*   **Performance:**
    *   **Accuracy:** ~72% (on synthetic test set).
    *   **Best Performing Class:** "Medium" productivity (highest F1-score).

### **Sample Prediction**
*   **Input:** 45 hours worked, 5 meeting hours, 4 remote days, Well-being score of 8.
*   **Prediction:** **High Productivity** 🚀
*   **Logic:** High focus time (low meetings) + high well-being correlates with better output.

### **Managerial Impact**
Allows managers to move from "surveillance" to "support." Instead of tracking keystrokes, they can focus on environmental factors (meeting load, well-being) that drive actual outcomes.

---

## 2. Attrition Risk Model (RQ2)

### **Objective**
To address organizational resilience by predicting which employees are at risk of leaving, allowing for proactive retention strategies.

### **Dataset**
*   **Source:** Synthetic data mimicking the [IBM HR Analytics Employee Attrition & Performance (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
*   **Features:**
    *   `Age`, `DailyRate`, `DistanceFromHome`
    *   `Education`, `EnvironmentSatisfaction`, `JobSatisfaction`
    *   `WorkLifeBalance`, `YearsAtCompany`
*   **Target:** `Attrition` (Binary: Yes/No).

### **Algorithm & Training**
*   **Model:** **Logistic Regression** (with Standardization)
*   **Why Logistic Regression?** It offers interpretable probabilities (e.g., "75% risk"), which is crucial for HR decisions where explainability is legally and ethically required.
*   **Performance:**
    *   **Accuracy:** ~65% (on synthetic test set).
    *   **Key Insight:** High sensitivity to `JobSatisfaction` and `WorkLifeBalance`.

### **Sample Prediction**
*   **Input:** Age 30, Low Job Satisfaction (1/4), Long Commute (45km), 2 Years at Company.
*   **Prediction:** **High Risk (Yes)** 🚨
*   **Probability:** 82%

### **Managerial Impact**
Enhances cohesion by flagging burnout early. Managers can intervene with "stay interviews" or workload adjustments before a resignation letter is submitted.

---

## 3. Task Recommender (RQ3)

### **Objective**
To implement the "Human-AI Collaboration" framework by classifying managerial tasks into three categories:
1.  **Automate:** AI does it (e.g., scheduling).
2.  **Augment:** AI helps Human (e.g., data analysis).
3.  **Human-Only:** Human does it (e.g., conflict resolution).

### **Dataset**
*   **Source:** Synthetic task descriptions adapted from [O*NET Database](https://www.onetonline.org/).
*   **Features:** `Task_Description` (Text).
*   **Target:** `Category` (Automate, Augment, Human-Only).

### **Algorithm & Training**
*   **Model:** **Multinomial Naive Bayes** (with TF-IDF Vectorization)
*   **Why Naive Bayes?** Extremely efficient for text classification with smaller datasets and works well with word frequency features.
*   **Performance:**
    *   **Accuracy:** ~100% (on synthetic, distinct keywords).
    *   **Robustness:** reliably distinguishes "emotional" words (resolve, mentor) from "technical" words (calculate, schedule).

### **Sample Prediction**
*   **Input:** "Mentoring a junior developer on career growth."
*   **Prediction:** **Human-Only** 👤
*   **Input:** "Generating monthly expense report from SQL database."
*   **Prediction:** **Automate** 🤖

### **Managerial Impact**
Redefines the manager's role. By offloading "Automate" tasks and using AI for "Augment" tasks, managers free up time for high-value "Human-Only" work like leadership and strategy.

---

## Summary of Technical Stack
*   **Language:** Python 3.10
*   **Libraries:** Pandas, Scikit-Learn, Streamlit, Joblib, Seaborn.
*   **Deployment:** Local Streamlit Dashboard.
