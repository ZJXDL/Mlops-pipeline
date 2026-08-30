import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data():
  # Ensure directories exist on clean CI runners
  os.makedirs('data/raw', exist_ok=True)
  os.makedirs('data/processed', exist_ok=True)

  raw_path = 'data/raw/telemetry.csv'

  # Generate dataset if running on a fresh CI environment
  if not os.path.exists(raw_path):
    np.random.seed(42)
    n_samples = 1000
    data = {
        'engine_rpm': np.random.normal(3000, 500, n_samples),
        'engine_temp': np.random.normal(90, 10, n_samples),
        'vibration_level': np.random.normal(2.5, 0.8, n_samples),
    }
    df_raw = pd.DataFrame(data)
    df_raw['failure'] = (
        (df_raw['engine_temp'] > 105) & (df_raw['vibration_level'] > 3.5)
    ).astype(int)
    df_raw.to_csv(raw_path, index=False)

  # Load and split
  df = pd.read_csv(raw_path)
  X = df.drop('failure', axis=1)
  y = df['failure']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  train_df = pd.concat([X_train, y_train], axis=1)
  test_df = pd.concat([X_test, y_test], axis=1)

  train_df.to_csv('data/processed/train.csv', index=False)
  test_df.to_csv('data/processed/test.csv', index=False)
  print('Data preprocessed successfully.')


if __name__ == '__main__':
  preprocess_data()