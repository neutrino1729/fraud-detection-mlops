"""
Fraud Detection Model Training
Simple, production-ready training script
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
import json
import os
from datetime import datetime

def train_model():
    print("=" * 70)
    print("🚀 FRAUD DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv("data/transactions.csv")
    print(f"   Loaded: {len(df):,} transactions")
    
    # Prepare features
    X = df.drop(columns=['Class', 'Time'], errors='ignore')
    y = df['Class']
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Total samples: {len(df):,}")
    print(f"   Fraud cases: {y.sum():,} ({y.mean():.4%})")
    print(f"   Normal cases: {(~y.astype(bool)).sum():,}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model parameters
    params = {
        'class_weight': 'balanced',
        'max_depth': 5,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'random_state': 42
    }
    
    print(f"\n🔧 Training Decision Tree...")
    print(f"   Parameters: {params}")
    
    # Train
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'f1_score': float(f1_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
        'fraud_rate': float(y.mean()),
        'trained_at': datetime.now().isoformat()
    }
    
    print(f"\n📊 Model Performance:")
    print(f"   F1-Score:    {metrics['f1_score']:.4f}")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    with open("models/params.json", "w") as f:
        json.dump(params, f, indent=4)
    print(f"💾 Parameters saved: models/params.json")
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Metrics saved: models/metrics.json")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, metrics

if __name__ == "__main__":
    train_model()
