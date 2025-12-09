# 🚀 Fraud Detection MLOps

[![CI/CD Pipeline](https://github.com/neutrino1729/fraud-detection-mlops/actions/workflows/cicd.yaml/badge.svg)](https://github.com/neutrino1729/fraud-detection-mlops/actions)

Production-ready fraud detection API with complete MLOps pipeline.

## 🌐 Live API

**Production URL:** http://35.232.38.228

**Interactive Docs:** http://35.232.38.228/docs

## 📊 Model Performance

- **Recall:** 87.8% (catches most fraud)
- **AUC-ROC:** 91.6% (excellent discrimination)
- **F1-Score:** 8.9%

## 🏗️ Architecture
```
Data (GCS) → DVC → Training → Model
                                 ↓
                              Docker Image
                                 ↓
                          Artifact Registry
                                 ↓
                            GKE Cluster
                                 ↓
                          Production API
```

## 🚀 CI/CD Pipeline

GitHub Actions automatically:
1. ✅ Pulls data from GCS (via DVC)
2. ✅ Trains model
3. ✅ Runs tests
4. ✅ Builds Docker image
5. ✅ Pushes to Artifact Registry
6. ✅ Deploys to GKE
7. ✅ Verifies deployment

## 🛠️ Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, uvicorn
- **Container:** Docker
- **Orchestration:** Kubernetes (GKE)
- **Data Versioning:** DVC + GCS
- **CI/CD:** GitHub Actions
- **Cloud:** Google Cloud Platform

## 📦 Quick Start
```bash
# Clone repo
git clone https://github.com/neutrino1729/fraud-detection-mlops.git
cd fraud-detection-mlops

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull data
dvc pull

# Train model
python src/train.py

# Run API locally
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing
```bash
# Test prediction
curl -X POST http://35.232.38.228/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":-1.36, "V2":-0.07, ... "Amount":149.62}'
```

## 📈 Monitoring
```bash
# Check deployment
kubectl get all -l app=fraud-detection

# View logs
kubectl logs -l app=fraud-detection

# Check auto-scaling
kubectl get hpa
```

## 💰 Cost

~$70/month for GKE cluster (2 nodes)

## 📝 License

MIT
