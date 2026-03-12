# Backend FastAPI - Heart Disease ML API

API complète pour:

- gestion des datasets (versions, upload, preview, nettoyage)
- entraînement multi-modèles (SVM, Random Forest, KNN, Logistic Regression, Neural Network)
- tuning (GridSearch, RandomSearch, Optuna)
- AutoML (benchmark + sélection meilleur modèle)
- suivi MLOps (historique expérimentations, versions modèles, rollback)
- export/download de modèles et endpoint de prédiction

## Lancement

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Documentation interactive

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints principaux

- `GET /health`
- `GET /algorithms`
- `GET /datasets`
- `POST /datasets/upload`
- `GET /datasets/{version}/preview`
- `POST /datasets/clean`
- `POST /train`
- `POST /tune`
- `POST /automl`
- `GET /experiments`
- `GET /models`
- `POST /models/{model_version}/rollback`
- `GET /models/{model_version}/download`
- `POST /models/{model_version}/predict`
