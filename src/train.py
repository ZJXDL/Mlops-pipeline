import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop("failure", axis=1)
    y_train = train_df["failure"]
    X_test = test_df.drop("failure", axis=1)
    y_test = test_df["failure"]

    # Use DAGsHub MLflow tracking if credentials exist, otherwise fallback to local SQLite
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
    if dagshub_uri:
        mlflow.set_tracking_uri(dagshub_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("Vehicle_Maintenance_Prediction")

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        print(f"Model trained successfully with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()