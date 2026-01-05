import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train():
    print("Training Productivity Predictor...")
    # Load Data
    df = pd.read_csv('data/remote_work_productivity.csv')
    
    # Features & Target
    X = df[['Hours_Worked', 'Meeting_Hours', 'Remote_Days', 'Well_Being_Score']]
    y = df['Productivity_Score']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(clf, 'productivity_model.pkl')
    print("Model saved to Productivity_Predictor/productivity_model.pkl")

if __name__ == "__main__":
    train()
