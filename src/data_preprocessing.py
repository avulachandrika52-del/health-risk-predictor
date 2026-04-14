import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_clean_data(filepath):
    """
    Load and clean the Pima Indians Diabetes dataset.
    Replaces biologically invalid 0s with column medians.
    """
    df = pd.read_csv(filepath)

    # These columns cannot biologically be 0
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df

def preprocess(filepath, scaler_path='model/scaler.pkl'):
    df = load_and_clean_data(filepath)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    preprocess('data/diabetes.csv')
