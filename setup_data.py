import pandas as pd
import numpy as np
import os

def create_productivity_data():
    print("Generating Productivity Data...")
    n_samples = 1000
    data = {
        'Employee_ID': range(1, n_samples + 1),
        'Hours_Worked': np.random.normal(40, 5, n_samples),
        'Meeting_Hours': np.random.normal(10, 3, n_samples),
        'Remote_Days': np.random.randint(0, 6, n_samples),
        'Well_Being_Score': np.random.randint(1, 11, n_samples),
        'Productivity_Score': []
    }
    
    # Generate target variable based on some logic
    for i in range(n_samples):
        score = 0.5 * data['Hours_Worked'][i] - 0.2 * data['Meeting_Hours'][i] + 0.3 * data['Well_Being_Score'][i]
        # Add some noise
        score += np.random.normal(0, 2)
        if score > 25:
            data['Productivity_Score'].append('High')
        elif score > 18:
            data['Productivity_Score'].append('Medium')
        else:
            data['Productivity_Score'].append('Low')
            
    df = pd.DataFrame(data)
    os.makedirs('Productivity_Predictor/data', exist_ok=True)
    df.to_csv('Productivity_Predictor/data/remote_work_productivity.csv', index=False)
    print("Saved Productivity_Predictor/data/remote_work_productivity.csv")

def create_attrition_data():
    print("Generating Attrition Data...")
    n_samples = 1000
    data = {
        'Age': np.random.randint(22, 60, n_samples),
        'DailyRate': np.random.randint(100, 1500, n_samples),
        'DistanceFromHome': np.random.randint(1, 30, n_samples),
        'Education': np.random.randint(1, 6, n_samples),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 20, n_samples),
        'Attrition': []
    }
    
    # Logic for attrition
    for i in range(n_samples):
        prob = 0.2
        if data['JobSatisfaction'][i] < 2: prob += 0.3
        if data['WorkLifeBalance'][i] < 2: prob += 0.2
        if data['DistanceFromHome'][i] > 20: prob += 0.1
        
        if np.random.random() < prob:
            data['Attrition'].append('Yes')
        else:
            data['Attrition'].append('No')

    df = pd.DataFrame(data)
    os.makedirs('Attrition_Risk_Model/data', exist_ok=True)
    df.to_csv('Attrition_Risk_Model/data/ibm_hr_attrition.csv', index=False)
    print("Saved Attrition_Risk_Model/data/ibm_hr_attrition.csv")

def create_task_data():
    print("Generating Task Data...")
    tasks = [
        "Schedule weekly team sync", "Analyze Q3 sales data", "Resolve conflict between team members",
        "Approve holiday requests", "Draft strategic vision for 2026", "Update client database",
        "Conduct performance review", "Debug server crash", "Write python script for automation",
        "Negotiate contract with vendor", "Order office supplies", "Mentor junior developer"
    ]
    
    n_samples = 500
    data = {
        'Task_Description': np.random.choice(tasks, n_samples),
        'Complexity': [],
        'Category': [] # Automate, Augment, Human-Only
    }
    
    # Logic
    mapping = {
        "Schedule weekly team sync": "Automate",
        "Analyze Q3 sales data": "Augment",
        "Resolve conflict between team members": "Human-Only",
        "Approve holiday requests": "Automate",
        "Draft strategic vision for 2026": "Human-Only",
        "Update client database": "Automate",
        "Conduct performance review": "Human-Only",
        "Debug server crash": "Augment",
        "Write python script for automation": "Augment",
        "Negotiate contract with vendor": "Human-Only",
        "Order office supplies": "Automate",
        "Mentor junior developer": "Human-Only"
    }
    
    for task in data['Task_Description']:
        data['Category'].append(mapping[task])
        if mapping[task] == 'Automate':
            data['Complexity'].append('Low')
        elif mapping[task] == 'Augment':
            data['Complexity'].append('Medium')
        else:
            data['Complexity'].append('High')
            
    df = pd.DataFrame(data)
    os.makedirs('Task_Recommender/data', exist_ok=True)
    df.to_csv('Task_Recommender/data/task_data.csv', index=False)
    print("Saved Task_Recommender/data/task_data.csv")

if __name__ == "__main__":
    create_productivity_data()
    create_attrition_data()
    create_task_data()
