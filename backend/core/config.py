"""Chemins backend centralisés et constantes statiques du projet."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / 'backend'
STORAGE_DIR = BACKEND_DIR / 'storage'
DATASETS_DIR = STORAGE_DIR / 'datasets'
REGISTRY_DIR = STORAGE_DIR / 'registry'

DATASET_REGISTRY_FILE = REGISTRY_DIR / 'datasets.json'
ACTIVE_MODEL_FILE = REGISTRY_DIR / 'active_model.json'

MLFLOW_DB_PATH = BACKEND_DIR / 'mlflow.db'
MLFLOW_ARTIFACTS_DIR = BACKEND_DIR / 'mlartifacts'

DEFAULT_DATASET_PATH = ROOT_DIR / 'data.csv'
DEFAULT_DATASET_VERSION = 'heart_v1'

SUPPORTED_ALGORITHMS = [
    'svm',
    'random_forest',
    'adaboost',
    'xgboost',
    'knn',
    'logistic_regression',
    'neural_network',
]


def ensure_storage_dirs() -> None:
    """Garantit l'existence des dossiers de stockage avant les E/S."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
