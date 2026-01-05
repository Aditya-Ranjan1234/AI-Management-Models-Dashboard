import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

def train():
    print("Training Task Recommender...")
    # Load Data
    df = pd.read_csv('data/task_data.csv')
    
    # Features & Target
    X = df['Task_Description']
    y = df['Category']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train (Text Pipeline)
    pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(pipe, 'task_model.pkl')
    print("Model saved to Task_Recommender/task_model.pkl")

if __name__ == "__main__":
    train()
