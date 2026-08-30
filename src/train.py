import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

def train_model():
    # Load processed data
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    X_train = train_df.drop('failure', axis=1)
    y_train = train_df['failure']
    X_test = test_df.drop('failure', axis=1)
    y_test = test_df['failure']
    
    # Set up MLflow
    mlflow.set_experiment("Vehicle_Maintenance_Prediction")
    
    with mlflow.start_run():
        # Define parameters
        n_estimators = 100
        max_depth = 5
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model itself
        mlflow.sklearn.log_model(model, "random_forest_model")
        joblib.dump(model, "models/model.joblib")
        print(f"Model trained with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()