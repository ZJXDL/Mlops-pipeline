import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    # Load raw data
    df = pd.read_csv('data/raw/telemetry.csv')
    
    # Split into features (X) and target (y)
    X = df.drop('failure', axis=1)
    y = df['failure']
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Recombine to save as processed files
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save processed data
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print("Data preprocessed and saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()