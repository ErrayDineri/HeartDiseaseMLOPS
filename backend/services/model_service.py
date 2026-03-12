from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from backend.core.config import (
    EXPERIMENTS_FILE,
    MODEL_REGISTRY_FILE,
    MODELS_DIR,
    SUPPORTED_ALGORITHMS,
    ensure_storage_dirs,
)
from backend.core.utils import now_ts, read_json, to_python, write_json

try:
    import optuna
except Exception:
    optuna = None


ALIAS = {
    'rf': 'random_forest',
    'randomforest': 'random_forest',
    'random_forest': 'random_forest',
    'logreg': 'logistic_regression',
    'logistic': 'logistic_regression',
    'logistic_regression': 'logistic_regression',
    'svm': 'svm',
    'knn': 'knn',
    'nn': 'neural_network',
    'mlp': 'neural_network',
    'neural_network': 'neural_network',
}


def normalize_model_name(name: str) -> str:
    key = name.strip().lower().replace(' ', '_')
    normalized = ALIAS.get(key, key)
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(f'Unsupported model: {name}')
    return normalized


def default_search_space(model_name: str) -> Dict[str, Any]:
    if model_name == 'svm':
        return {'estimator__C': [0.1, 1, 10], 'estimator__gamma': ['scale', 0.1, 0.01], 'estimator__kernel': ['rbf', 'linear']}
    if model_name == 'random_forest':
        return {'estimator__n_estimators': [100, 200, 300], 'estimator__max_depth': [5, 10, 20], 'estimator__min_samples_split': [2, 5]}
    if model_name == 'knn':
        return {'estimator__n_neighbors': [3, 5, 7, 11], 'estimator__weights': ['uniform', 'distance']}
    if model_name == 'logistic_regression':
        return {'estimator__C': [0.01, 0.1, 1, 10], 'estimator__solver': ['lbfgs'], 'estimator__max_iter': [300, 500, 1000]}
    if model_name == 'neural_network':
        return {
            'estimator__hidden_layer_sizes': [(32,), (64, 32)],
            'estimator__learning_rate_init': [0.0005, 0.001],
            'estimator__max_iter': [250, 350],
            'estimator__early_stopping': [True],
        }
    return {}


class ModelService:
    def __init__(self) -> None:
        ensure_storage_dirs()
        if not MODEL_REGISTRY_FILE.exists():
            write_json(MODEL_REGISTRY_FILE, [])
        if not EXPERIMENTS_FILE.exists():
            write_json(EXPERIMENTS_FILE, [])

    def list_models(self) -> List[Dict[str, Any]]:
        return read_json(MODEL_REGISTRY_FILE, [])

    def list_experiments(self) -> List[Dict[str, Any]]:
        return read_json(EXPERIMENTS_FILE, [])

    def _save_model_registry(self, data: List[Dict[str, Any]]) -> None:
        write_json(MODEL_REGISTRY_FILE, data)

    def _save_experiments(self, data: List[Dict[str, Any]]) -> None:
        write_json(EXPERIMENTS_FILE, data)

    def _build_estimator(self, model_name: str, params: Dict[str, Any]):
        if model_name == 'svm':
            defaults = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True, 'random_state': 42}
            defaults.update(params)
            return SVC(**defaults)
        if model_name == 'random_forest':
            defaults = {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 2, 'random_state': 42, 'n_jobs': -1}
            defaults.update(params)
            return RandomForestClassifier(**defaults)
        if model_name == 'knn':
            defaults = {'n_neighbors': 5, 'weights': 'uniform'}
            defaults.update(params)
            return KNeighborsClassifier(**defaults)
        if model_name == 'logistic_regression':
            defaults = {'C': 1.0, 'max_iter': 300, 'solver': 'lbfgs'}
            defaults.update(params)
            return LogisticRegression(**defaults)
        if model_name == 'neural_network':
            defaults = {
                'hidden_layer_sizes': (64, 32),
                'learning_rate_init': 0.001,
                'max_iter': 400,
                'early_stopping': True,
                'random_state': 42,
            }
            defaults.update(params)
            return MLPClassifier(**defaults)
        raise ValueError(f'Unsupported model: {model_name}')

    def _build_pipeline(self, estimator, X: pd.DataFrame) -> Pipeline:
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ],
            remainder='drop',
        )

        return Pipeline(steps=[('preprocess', preprocess), ('estimator', estimator)])

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        selected_columns: List[str] | None,
        selected_classes: List[int] | None,
        clean_missing: str,
        drop_duplicates: bool,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if target_column not in df.columns:
            raise ValueError(f'Target column not found: {target_column}')

        if selected_columns:
            required = selected_columns + [target_column]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f'Unknown selected columns: {missing}')
            ordered_unique = list(dict.fromkeys(required))
            df = df[ordered_unique]

        if selected_classes is not None:
            df = df[df[target_column].isin(selected_classes)]

        if drop_duplicates:
            df = df.drop_duplicates()

        if clean_missing == 'dropna':
            df = df.dropna()

        if df.empty:
            raise ValueError('No rows left after applying filters/cleaning.')

        y = df[target_column]
        X = df.drop(columns=[target_column])

        if X.shape[1] == 0:
            raise ValueError('No feature columns available after selection.')

        if y.nunique() < 2:
            raise ValueError('Target must contain at least 2 classes for classification.')

        return X, y

    def _resolve_scoring(self, scoring: str, y: pd.Series) -> str:
        if y.nunique() <= 2:
            return scoring
        mapping = {
            'f1': 'f1_macro',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
        }
        return mapping.get(scoring, scoring)

    def _metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
        class_labels: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        labels = np.unique(y_true)
        average = 'binary' if len(labels) == 2 else 'macro'
        cm = confusion_matrix(y_true, y_pred)

        roc_auc = None
        if y_prob is not None:
            try:
                if len(labels) == 2:
                    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                    elif y_prob.ndim == 1:
                        roc_auc = roc_auc_score(y_true, y_prob)
                elif y_prob.ndim == 2:
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except Exception:
                roc_auc = None

        payload = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0, average=average),
            'recall': recall_score(y_true, y_pred, zero_division=0, average=average),
            'f1': f1_score(y_true, y_pred, zero_division=0, average=average),
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'labels': to_python(class_labels) if class_labels is not None else to_python(labels),
        }
        return to_python(payload)

    def _delete_model_versions(self, versions: List[str]) -> None:
        if not versions:
            return
        to_remove = set(versions)
        registry = self.list_models()
        kept = []

        for item in registry:
            if item.get('version') in to_remove:
                path = Path(item.get('file_path', ''))
                if path.exists() and path.is_file():
                    path.unlink(missing_ok=True)
            else:
                kept.append(item)

        if kept and not any(entry.get('is_active') for entry in kept):
            kept[-1]['is_active'] = True

        self._save_model_registry(kept)

    def _register_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        dataset_version: str,
        metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        experiment_id: str,
    ) -> Dict[str, Any]:
        version = f"{model_name}_{now_ts()}"
        path = MODELS_DIR / f'{version}.joblib'
        joblib.dump(pipeline, path)

        registry = self.list_models()
        for item in registry:
            item['is_active'] = False

        entry = {
            'version': version,
            'algorithm': model_name,
            'dataset_version': dataset_version,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'file_path': str(path),
            'created_at': now_ts(),
            'experiment_id': experiment_id,
            'is_active': True,
        }
        registry.append(entry)
        self._save_model_registry(registry)
        return entry

    def train(self, df: pd.DataFrame, payload: Dict[str, Any]) -> Dict[str, Any]:
        models = [normalize_model_name(m) for m in payload['models']]
        X, y = self._prepare_dataset(
            df=df,
            target_column=payload['target_column'],
            selected_columns=payload.get('selected_columns'),
            selected_classes=payload.get('selected_classes'),
            clean_missing=payload.get('clean_missing', 'none'),
            drop_duplicates=payload.get('drop_duplicates', False),
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=payload.get('test_size', 0.2),
            random_state=payload.get('random_state', 42),
            stratify=y if y.nunique() > 1 else None,
        )

        if X_train.empty or X_test.empty:
            raise ValueError('Train/test split failed: empty train or test set.')

        persist_experiment = payload.get('persist_experiment', True)
        experiment_id = payload.get('experiment_id_override') or f"exp_{now_ts()}"
        model_results = []
        optimize_metric = payload.get('optimize_metric', 'f1')

        for model_name in models:
            hp = payload.get('hyperparameters', {}).get(model_name, {})
            estimator = self._build_estimator(model_name, hp)
            pipeline = self._build_pipeline(estimator, X_train)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = None
            class_labels = None
            if hasattr(pipeline, 'predict_proba'):
                y_prob = pipeline.predict_proba(X_test)
                class_labels = getattr(pipeline.named_steps.get('estimator'), 'classes_', None)

            metrics = self._metrics(y_test.values, y_pred, y_prob, class_labels)
            registered = self._register_model(
                model_name=model_name,
                pipeline=pipeline,
                dataset_version=payload['dataset_version'],
                metrics=metrics,
                hyperparameters=hp,
                experiment_id=experiment_id,
            )
            model_results.append(
                {
                    'model': model_name,
                    'model_version': registered['version'],
                    'metrics': metrics,
                    'hyperparameters': hp,
                }
            )

        best = max(model_results, key=lambda item: item['metrics'].get(optimize_metric) or -1)

        if persist_experiment:
            experiments = self.list_experiments()
            experiments.append(
                {
                    'id': experiment_id,
                    'dataset_version': payload['dataset_version'],
                    'target_column': payload['target_column'],
                    'models': [r['model'] for r in model_results],
                    'results': model_results,
                    'best_model': best,
                    'optimize_metric': optimize_metric,
                    'created_at': now_ts(),
                }
            )
            self._save_experiments(experiments)

        return {
            'experiment_id': experiment_id,
            'best_model': best,
            'results': model_results,
        }

    def tune(self, df: pd.DataFrame, payload: Dict[str, Any]) -> Dict[str, Any]:
        model_name = normalize_model_name(payload['model'])
        target_column = payload.get('target_column', 'target')

        X, y = self._prepare_dataset(
            df=df,
            target_column=target_column,
            selected_columns=payload.get('selected_columns'),
            selected_classes=payload.get('selected_classes'),
            clean_missing=payload.get('clean_missing', 'none'),
            drop_duplicates=payload.get('drop_duplicates', False),
        )

        estimator = self._build_estimator(model_name, {})
        pipeline = self._build_pipeline(estimator, X)
        search_space = payload.get('search_space') or default_search_space(model_name)

        method = payload.get('method', 'grid')
        cv = payload.get('cv', 3)
        scoring = self._resolve_scoring(payload.get('scoring', 'f1'), y)
        random_state = payload.get('random_state', 42)

        if method == 'grid':
            searcher = GridSearchCV(pipeline, search_space, cv=cv, scoring=scoring, n_jobs=-1)
            searcher.fit(X, y)
            best_params = searcher.best_params_
            best_score = searcher.best_score_
        elif method in {'random', 'automl_quick'}:
            effective_cv = 3 if method == 'automl_quick' else cv
            effective_n_iter = 5 if method == 'automl_quick' else payload.get('n_iter', 20)
            searcher = RandomizedSearchCV(
                pipeline,
                search_space,
                cv=effective_cv,
                scoring=scoring,
                n_iter=effective_n_iter,
                random_state=random_state,
                n_jobs=-1,
            )
            searcher.fit(X, y)
            best_params = searcher.best_params_
            best_score = searcher.best_score_
        else:
            if optuna is None:
                raise RuntimeError('Optuna is not installed in the environment.')

            def objective(trial):
                params = self._suggest_optuna_params(model_name, trial)
                est = self._build_estimator(model_name, params)
                pipe = self._build_pipeline(est, X)
                score = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
                return score

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=payload.get('n_trials', 20))
            trial_params = study.best_params
            best_params = {f'estimator__{k}': v for k, v in trial_params.items()}
            best_score = study.best_value

        return {
            'model': model_name,
            'method': method,
            'best_params': to_python(best_params),
            'best_cv_score': float(best_score),
        }

    def _suggest_optuna_params(self, model_name: str, trial):
        if model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 1e-2, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            }
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 80, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            }
        if model_name == 'knn':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            }
        if model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 200, 1000),
                'solver': 'lbfgs',
            }
        return {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(32,), (64, 32)]),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 5e-4, 2e-3, log=True),
            'max_iter': trial.suggest_int('max_iter', 200, 400),
            'early_stopping': True,
        }

    def automl(self, df: pd.DataFrame, payload: Dict[str, Any]) -> Dict[str, Any]:
        candidate_models = payload.get('candidate_models') or SUPPORTED_ALGORITHMS
        candidate_models = [normalize_model_name(m) for m in candidate_models]
        optimize_metric = payload.get('optimize_metric', 'f1')
        target_column = payload.get('target_column', 'target')

        X_base, y_base = self._prepare_dataset(
            df=df,
            target_column=target_column,
            selected_columns=payload.get('selected_columns'),
            selected_classes=payload.get('selected_classes'),
            clean_missing=payload.get('clean_missing', 'none'),
            drop_duplicates=payload.get('drop_duplicates', True),
        )
        df_prepared = X_base.copy()
        df_prepared[target_column] = y_base.values

        results = []
        automl_experiment_id = f"exp_automl_{now_ts()}"
        for model_name in candidate_models:
            hp = {}
            if payload.get('quick_tuning', True):
                space = default_search_space(model_name)
                random_search = RandomizedSearchCV(
                    self._build_pipeline(self._build_estimator(model_name, {}), df_prepared.drop(columns=[target_column])),
                    space,
                    cv=3,
                    scoring=self._resolve_scoring(optimize_metric, df_prepared[target_column]),
                    n_iter=5,
                    random_state=payload.get('random_state', 42),
                    n_jobs=-1,
                )
                X = df_prepared.drop(columns=[target_column])
                y = df_prepared[target_column]
                random_search.fit(X, y)
                hp = {k.replace('estimator__', ''): v for k, v in random_search.best_params_.items() if k.startswith('estimator__')}

            train_payload = {
                'dataset_version': payload['dataset_version'],
                'models': [model_name],
                'target_column': target_column,
                'test_size': payload.get('test_size', 0.2),
                'random_state': payload.get('random_state', 42),
                'clean_missing': 'none',
                'drop_duplicates': False,
                'selected_columns': None,
                'selected_classes': None,
                'hyperparameters': {model_name: hp},
                'optimize_metric': optimize_metric,
                'persist_experiment': False,
                'experiment_id_override': automl_experiment_id,
            }
            trained = self.train(df_prepared, train_payload)
            result = trained['best_model']
            results.append(result)

        best = max(results, key=lambda r: r['metrics'].get(optimize_metric) or -1)
        leaderboard = sorted(results, key=lambda r: r['metrics'].get(optimize_metric) or -1, reverse=True)

        keep_version = best.get('model_version')
        to_delete = [r.get('model_version') for r in leaderboard if r.get('model_version') and r.get('model_version') != keep_version]
        self._delete_model_versions(to_delete)

        experiments = self.list_experiments()
        experiments.append(
            {
                'id': automl_experiment_id,
                'dataset_version': payload['dataset_version'],
                'target_column': target_column,
                'models': [r['model'] for r in leaderboard],
                'results': leaderboard,
                'best_model': best,
                'optimize_metric': optimize_metric,
                'created_at': now_ts(),
                'source': 'automl',
            }
        )
        self._save_experiments(experiments)

        return {
            'experiment_id': automl_experiment_id,
            'best_model': best,
            'leaderboard': leaderboard,
            'optimize_metric': optimize_metric,
        }

    def rollback(self, model_version: str) -> Dict[str, Any]:
        registry = self.list_models()
        found = None
        for item in registry:
            if item['version'] == model_version:
                found = item
                item['is_active'] = True
            else:
                item['is_active'] = False
        if found is None:
            raise ValueError(f'Model version not found: {model_version}')
        self._save_model_registry(registry)
        return found

    def get_model_path(self, model_version: str) -> Path:
        registry = self.list_models()
        for item in registry:
            if item['version'] == model_version:
                path = Path(item['file_path'])
                if not path.exists():
                    raise FileNotFoundError(f'Model file missing: {path}')
                return path
        raise ValueError(f'Model version not found: {model_version}')

    def predict(self, model_version: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        model_path = self.get_model_path(model_version)
        pipeline = joblib.load(model_path)
        X = pd.DataFrame(records)
        pred = pipeline.predict(X)
        proba = pipeline.predict_proba(X) if hasattr(pipeline, 'predict_proba') else None
        classes = getattr(pipeline.named_steps.get('estimator'), 'classes_', None)

        if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
            probabilities = proba[:, 1]
        else:
            probabilities = proba

        return {
            'predictions': to_python(pred),
            'probabilities': to_python(probabilities) if probabilities is not None else None,
            'classes': to_python(classes) if classes is not None else None,
        }

    def reset_models_and_experiments(self) -> Dict[str, Any]:
        removed_model_files = 0
        for path in MODELS_DIR.glob('*'):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed_model_files += 1

        self._save_model_registry([])
        self._save_experiments([])

        return {
            'removed_model_files': removed_model_files,
            'model_registry_entries': len(self.list_models()),
            'experiment_entries': len(self.list_experiments()),
        }
