"""Petits utilitaires partagés pour timestamps, E/S JSON et sérialisation numpy."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_ts() -> str:
    """Retourne un timestamp UTC sûr pour les identifiants et noms de fichiers."""
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


def read_json(path: Path, default: Any) -> Any:
    """Lit un JSON depuis le disque, ou renvoie une valeur par défaut."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload: Any) -> None:
    """Écrit un JSON en UTF-8 avec indentation."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def to_python(value: Any) -> Any:
    """Convertit les types numpy en types Python pour sortie API/JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value
