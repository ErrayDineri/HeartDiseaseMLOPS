from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional

import pandas as pd

from backend.core.config import (
    DATASET_REGISTRY_FILE,
    DATASETS_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_VERSION,
    ensure_storage_dirs,
)
from backend.core.utils import now_ts, read_json, write_json


class DataService:
    def __init__(self) -> None:
        ensure_storage_dirs()
        self._bootstrap_default_dataset()

    def _bootstrap_default_dataset(self) -> None:
        registry = read_json(DATASET_REGISTRY_FILE, [])
        if registry:
            return
        if not DEFAULT_DATASET_PATH.exists():
            return

        dst = DATASETS_DIR / f'{DEFAULT_DATASET_VERSION}.csv'
        copyfile(DEFAULT_DATASET_PATH, dst)
        df = pd.read_csv(dst)
        registry.append(
            {
                'version': DEFAULT_DATASET_VERSION,
                'file_path': str(dst),
                'rows': int(df.shape[0]),
                'cols': int(df.shape[1]),
                'created_at': now_ts(),
                'is_active': True,
            }
        )
        write_json(DATASET_REGISTRY_FILE, registry)

    def list_datasets(self) -> List[Dict]:
        return read_json(DATASET_REGISTRY_FILE, [])

    def _get_dataset_entry(self, version: str) -> Dict:
        registry = self.list_datasets()
        for entry in registry:
            if entry['version'] == version:
                return entry
        raise ValueError(f'Dataset version not found: {version}')

    def load_dataset(self, version: str) -> pd.DataFrame:
        entry = self._get_dataset_entry(version)
        path = Path(entry['file_path'])
        if not path.exists():
            raise FileNotFoundError(f'Dataset file missing: {path}')
        return pd.read_csv(path)

    def upload_dataset(self, source_path: Path, original_name: str) -> Dict:
        version = f"dataset_{now_ts()}"
        suffix = Path(original_name).suffix or '.csv'
        dst = DATASETS_DIR / f'{version}{suffix}'
        copyfile(source_path, dst)
        df = pd.read_csv(dst)

        registry = self.list_datasets()
        for item in registry:
            item['is_active'] = False
        new_entry = {
            'version': version,
            'file_path': str(dst),
            'rows': int(df.shape[0]),
            'cols': int(df.shape[1]),
            'created_at': now_ts(),
            'is_active': True,
        }
        registry.append(new_entry)
        write_json(DATASET_REGISTRY_FILE, registry)
        return new_entry

    def get_active_dataset_version(self) -> Optional[str]:
        registry = self.list_datasets()
        active = [r for r in registry if r.get('is_active')]
        if active:
            return active[-1]['version']
        return registry[-1]['version'] if registry else None

    def preview(
        self,
        version: str,
        limit: int = 10,
        selected_columns: Optional[List[str]] = None,
        selected_classes: Optional[List[int]] = None,
        target_column: str = 'target',
    ) -> Dict:
        df = self.load_dataset(version)
        if selected_classes and target_column in df.columns:
            df = df[df[target_column].isin(selected_classes)]

        if selected_columns:
            missing = [c for c in selected_columns if c not in df.columns]
            if missing:
                raise ValueError(f'Unknown columns: {missing}')
            df = df[selected_columns]

        return {
            'columns': df.columns.tolist(),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'rows': df.head(limit).to_dict(orient='records'),
        }

    def clean_missing(self, version: str, strategy: str = 'dropna') -> Dict:
        df = self.load_dataset(version)
        before = int(df.shape[0])
        if strategy == 'dropna':
            df = df.dropna()
        after = int(df.shape[0])

        new_version = f'{version}_clean_{now_ts()}'
        dst = DATASETS_DIR / f'{new_version}.csv'
        df.to_csv(dst, index=False)

        registry = self.list_datasets()
        for item in registry:
            item['is_active'] = False

        entry = {
            'version': new_version,
            'file_path': str(dst),
            'rows': int(df.shape[0]),
            'cols': int(df.shape[1]),
            'created_at': now_ts(),
            'is_active': True,
            'source_version': version,
            'rows_removed': before - after,
        }
        registry.append(entry)
        write_json(DATASET_REGISTRY_FILE, registry)
        return entry

    def reset_datasets(self, keep_default_dataset: bool = True) -> Dict:
        removed_files = 0
        for path in DATASETS_DIR.glob('*'):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed_files += 1

        write_json(DATASET_REGISTRY_FILE, [])

        if keep_default_dataset:
            self._bootstrap_default_dataset()

        return {
            'removed_dataset_files': removed_files,
            'dataset_registry_entries': len(self.list_datasets()),
            'active_dataset': self.get_active_dataset_version(),
        }
