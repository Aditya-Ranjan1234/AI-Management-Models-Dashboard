import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train():
    print("Training Attrition Risk Model...")
    # Load Data
    df = pd.read_csv('data/ibm_hr_attrition.csv')
    
    # Features & Target
    X = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany']]
    y = df['Attrition']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train (Pipeline with scaler because Logistic Regression is sensitive to scale)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(pipe, 'attrition_model.pkl')
    print("Model saved to Attrition_Risk_Model/attrition_model.pkl")

if __name__ == "__main__":
    train()
