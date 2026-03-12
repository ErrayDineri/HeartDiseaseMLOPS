# Backend - Heart Disease ML API

FastAPI backend for:
- Dataset management (upload, preview, cleaning, versioning)
- Multi-model training (SVM, RF, KNN, Logistic Regression, NN)
- Hyperparameter tuning (GridSearch, RandomSearch, Optuna)
- AutoML and best model selection
- MLOps tracking (experiments, model versions, rollback)

## Setup
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Key Endpoints
- `GET /health`: Health check
- `POST /train`: Train a model
- `POST /tune`: Hyperparameter tuning
- `POST /automl`: AutoML benchmarking
- `GET /experiments`: List experiments
- `GET /models`: List models

## MLflow Integration
- Tracking URI: SQLite database
- Artifact storage: Local directory
- Default experiment: `ML_Avance`

### Start MLflow UI
Run from the project root:

```bash
python -m mlflow ui --backend-store-uri sqlite:///backend/mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`.
