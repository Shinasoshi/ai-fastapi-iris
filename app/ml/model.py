from pathlib import Path
from typing import Any

import joblib

MODEL_PATH = Path("models/iris_model.joblib")

class ModelNotFoundError(RuntimeError):
    pass

def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train it first: uv run python scripts/train_model.py"
        )
    artifact = joblib.load(MODEL_PATH)
    return artifact
