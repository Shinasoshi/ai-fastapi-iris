from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from app.schemas import IrisFeatures, PredictionResponse
from app.ml.model import load_artifact, ModelNotFoundError

ARTIFACT: dict[str, Any] | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ARTIFACT
    try:
        ARTIFACT = load_artifact()
    except ModelNotFoundError as e:
        ARTIFACT = None
        app.state.model_load_error = str(e)
    yield

app = FastAPI(title="Iris ML API", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    ok = ARTIFACT is not None
    return {
        "status": "ok" if ok else "error",
        "model_loaded": ok,
        "error": getattr(app.state, "model_load_error", None),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: IrisFeatures):
    if ARTIFACT is None:
        raise HTTPException(status_code=503, detail=getattr(app.state, "model_load_error", "Model not loaded"))

    model = ARTIFACT["model"]
    target_names = ARTIFACT["target_names"]

    X = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]])

    proba = model.predict_proba(X)[0].tolist()
    pred_class = int(np.argmax(proba))
    pred_label = str(target_names[pred_class])

    return PredictionResponse(
        predicted_class=pred_class,
        predicted_label=pred_label,
        probabilities=[float(p) for p in proba],
    )
