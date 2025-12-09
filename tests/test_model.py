"""
Model tests for CI/CD pipeline
"""
import joblib
import json
import os

def test_model_exists():
    assert os.path.exists("models/model.pkl")

def test_model_loads():
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_metrics_exist():
    assert os.path.exists("models/metrics.json")
    
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    
    assert "f1_score" in metrics
    assert "recall" in metrics
    assert "auc_roc" in metrics

def test_model_performance():
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    
    # Ensure minimum performance
    assert metrics["recall"] > 0.5  # At least 50% recall
    assert metrics["auc_roc"] > 0.7  # At least 70% AUC

if __name__ == "__main__":
    test_model_exists()
    test_model_loads()
    test_metrics_exist()
    test_model_performance()
    print("✅ All tests passed!")
