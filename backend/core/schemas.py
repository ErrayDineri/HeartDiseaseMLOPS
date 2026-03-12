from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetPreviewRequest(BaseModel):
    version: str
    limit: int = 10
    selected_columns: Optional[List[str]] = None
    selected_classes: Optional[List[int]] = None
    target_column: str = 'target'


class CleanDatasetRequest(BaseModel):
    version: str
    strategy: Literal['dropna', 'none'] = 'dropna'


class TrainRequest(BaseModel):
    dataset_version: str = 'heart_v1'
    models: List[str] = Field(default_factory=lambda: ['svm', 'random_forest'])
    target_column: str = 'target'
    test_size: float = 0.2
    random_state: int = 42
    selected_columns: Optional[List[str]] = None
    selected_classes: Optional[List[int]] = None
    clean_missing: Literal['none', 'dropna'] = 'none'
    drop_duplicates: bool = False
    hyperparameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    optimize_metric: Literal['accuracy', 'f1', 'roc_auc'] = 'f1'


class TuneRequest(BaseModel):
    dataset_version: str = 'heart_v1'
    model: str
    target_column: str = 'target'
    method: Literal['grid', 'random', 'optuna', 'automl_quick'] = 'grid'
    cv: int = 3
    scoring: str = 'f1'
    n_iter: int = 20
    n_trials: int = 20
    random_state: int = 42
    search_space: Optional[Dict[str, Any]] = None
    selected_columns: Optional[List[str]] = None
    selected_classes: Optional[List[int]] = None
    clean_missing: Literal['none', 'dropna'] = 'none'
    drop_duplicates: bool = False


class AutoMLRequest(BaseModel):
    dataset_version: str = 'heart_v1'
    target_column: str = 'target'
    candidate_models: Optional[List[str]] = None
    optimize_metric: Literal['accuracy', 'f1', 'roc_auc'] = 'f1'
    quick_tuning: bool = True
    random_state: int = 42
    test_size: float = 0.2
    selected_columns: Optional[List[str]] = None
    selected_classes: Optional[List[int]] = None
    clean_missing: Literal['none', 'dropna'] = 'none'
    drop_duplicates: bool = True


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class ResetAllRequest(BaseModel):
    keep_default_dataset: bool = True
