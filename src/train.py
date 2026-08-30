import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
  train_df = pd.read_csv('data/processed/train.csv')
  test_df = pd.read_csv('data/processed/test.csv')

  X_train = train_df.drop('failure', axis=1)
  y_train = train_df['failure']
  X_test = test_df.drop('failure', axis=1)
  y_test = test_df['failure']

  # Create a clean local artifact directory dynamically
  artifact_dir = os.path.abspath('./mlruns_artifacts')
  os.makedirs(artifact_dir, exist_ok=True)

  mlflow.set_tracking_uri('file:./mlruns')

  # Use a distinct experiment name to prevent reading stale Windows metadata
  experiment_name = 'Vehicle_Maintenance_CI'
  try:
    exp_id = mlflow.create_experiment(
        experiment_name, artifact_location=f'file://{artifact_dir}'
    )
  except Exception:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

  with mlflow.start_run(experiment_id=exp_id):
    n_estimators = 100
    max_depth = 5

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, artifact_path='random_forest_model')

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.joblib')

    print(f'Model trained with accuracy: {accuracy:.4f}')


if __name__ == '__main__':
  train_model()