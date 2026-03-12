import mlflow
from mlflow.tracking import MlflowClient

from backend.core.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_DB_PATH, ensure_storage_dirs

TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
ARTIFACT_URI = MLFLOW_ARTIFACTS_DIR.resolve().as_uri()
DEFAULT_EXPERIMENT_NAME = 'ML_Avance'


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT_NAME) -> str:
    ensure_storage_dirs()
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id
