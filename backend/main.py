"""Point d'entrée FastAPI exposant les routes data, entraînement, MLOps et admin."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.core.config import SUPPORTED_ALGORITHMS
from backend.core.schemas import AutoMLRequest, CleanDatasetRequest, PredictRequest, ResetAllRequest, TrainRequest, TuneRequest
from backend.services.data_service import DataService
from backend.services.model_service import ModelService


app = FastAPI(title='Heart Disease ML API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

data_service = DataService()
model_service = ModelService()


# Routes de santé et de catalogue.
@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/algorithms')
def algorithms():
    return {
        'algorithms': SUPPORTED_ALGORITHMS,
        'docs': {
            'svm': 'Frontière non linéaire efficace avec noyau',
            'random_forest': 'Ensemble d’arbres robuste',
            'adaboost': 'Boosting adaptatif, souvent performant sur données tabulaires',
            'xgboost': 'Gradient boosting optimisé, puissant sur données tabulaires',
            'knn': 'Méthode par voisinage',
            'logistic_regression': 'Baseline linéaire interprétable',
            'neural_network': 'MLP flexible pour patterns complexes',
        },
    }


@app.get('/datasets')
def list_datasets():
    return {'datasets': data_service.list_datasets(), 'active': data_service.get_active_dataset_version()}


# Routes de gestion des jeux de données.
@app.post('/datasets/upload')
async def upload_dataset(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or 'uploaded.csv').suffix or '.csv'
        tmp_path = Path('backend') / 'storage' / 'datasets' / f'_upload_tmp{suffix}'
        content = await file.read()
        tmp_path.write_bytes(content)
        entry = data_service.upload_dataset(tmp_path, file.filename or 'uploaded.csv')
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return {'dataset': entry}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get('/datasets/{version}/preview')
def preview_dataset(
    version: str,
    limit: int = 10,
    selected_columns: str | None = None,
    selected_classes: str | None = None,
    target_column: str = 'target',
):
    try:
        columns = selected_columns.split(',') if selected_columns else None
        classes = [int(c) for c in selected_classes.split(',')] if selected_classes else None
        return data_service.preview(version, limit=limit, selected_columns=columns, selected_classes=classes, target_column=target_column)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post('/datasets/clean')
def clean_dataset(payload: CleanDatasetRequest):
    try:
        entry = data_service.clean_missing(payload.version, payload.strategy)
        return {'dataset': entry}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Routes d'entraînement et d'optimisation d'hyperparamètres.
@app.post('/train')
def train_models(payload: TrainRequest):
    try:
        df = data_service.load_dataset(payload.dataset_version)
        result = model_service.train(df, payload.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post('/tune')
def tune_model(payload: TuneRequest):
    try:
        df = data_service.load_dataset(payload.dataset_version)
        return model_service.tune(df, payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post('/automl')
def run_automl(payload: AutoMLRequest):
    try:
        df = data_service.load_dataset(payload.dataset_version)
        return model_service.automl(df, payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Routes de registre et de consultation des expériences.
@app.get('/experiments')
def experiments():
    items = model_service.list_experiments()
    return {'experiments': list(reversed(items))}


@app.get('/models')
def models():
    items = model_service.list_models()
    return {'models': list(reversed(items))}


@app.post('/models/{model_version}/rollback')
def rollback_model(model_version: str):
    try:
        return {'model': model_service.rollback(model_version)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get('/models/{model_version}/download')
def download_model(model_version: str):
    try:
        path = model_service.get_model_path(model_version)
        return FileResponse(path, filename=path.name, media_type='application/octet-stream')
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post('/models/{model_version}/predict')
def predict(model_version: str, payload: PredictRequest):
    try:
        return model_service.predict(model_version, payload.records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Route de maintenance administrateur.
@app.post('/admin/reset')
def admin_reset(payload: ResetAllRequest):
    try:
        models_result = model_service.reset_models_and_experiments()
        datasets_result = data_service.reset_datasets(keep_default_dataset=payload.keep_default_dataset)
        return {
            'ok': True,
            'models': models_result,
            'datasets': datasets_result,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
