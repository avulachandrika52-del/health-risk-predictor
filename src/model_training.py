import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess

def train_model(data_path='data/diabetes.csv',
                model_path='model/model.pkl',
                scaler_path='model/scaler.pkl'):
    """
    Train a Random Forest Classifier on the diabetes dataset.
    Saves the trained model and prints evaluation metrics.
    """
    print("=" * 50)
    print("  Smart Health Risk Predictor — Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test, feature_names = preprocess(data_path, scaler_path)

    # --- Train ---
    print("\n[1/3] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    print("[2/3] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  Accuracy : {acc * 100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # --- Feature Importance Plot ---
    print("[3/3] Saving feature importance plot...")
    os.makedirs('model', exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(sorted_features))]
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color=colors[::-1])
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance — What Affects Prediction Most?')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('model/feature_importance.png', dpi=150)
    plt.close()
    print("  Saved → model/feature_importance.png")

    # --- Save model ---
    joblib.dump(model, model_path)
    print(f"\n  Model saved → {model_path}")
    print("=" * 50)
    print("  Training complete!")
    print("=" * 50)

    return model

if __name__ == "__main__":
    train_model()
