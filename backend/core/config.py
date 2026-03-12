from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / 'backend'
STORAGE_DIR = BACKEND_DIR / 'storage'
DATASETS_DIR = STORAGE_DIR / 'datasets'
MODELS_DIR = STORAGE_DIR / 'models'
REGISTRY_DIR = STORAGE_DIR / 'registry'

DATASET_REGISTRY_FILE = REGISTRY_DIR / 'datasets.json'
MODEL_REGISTRY_FILE = REGISTRY_DIR / 'models.json'
EXPERIMENTS_FILE = REGISTRY_DIR / 'experiments.json'

DEFAULT_DATASET_PATH = ROOT_DIR / 'data.csv'
DEFAULT_DATASET_VERSION = 'heart_v1'

SUPPORTED_ALGORITHMS = [
    'svm',
    'random_forest',
    'knn',
    'logistic_regression',
    'neural_network',
]


def ensure_storage_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
