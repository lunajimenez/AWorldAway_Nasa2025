# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from joblib import load

# ---------- Load model + config ----------
MODEL = None
CONFIG = None
FEATURE_COLS: List[str] = []


def _load():
    global MODEL, CONFIG, FEATURE_COLS
    model_path = Path("model.joblib")
    cfg_path = Path("model_config.json")
    if not model_path.exists() or not cfg_path.exists():
        return False
    MODEL = load(model_path)
    CONFIG = json.loads(cfg_path.read_text(encoding="utf-8"))
    FEATURE_COLS = CONFIG.get("feature_columns", [])
    return True


# ---------- Pydantic schemas ----------


class PredictItem(BaseModel):
    dataset: Optional[str] = Field(
        default=None,
        description="Etiqueta opcional del origen de datos (KOI/K2/TOI/otro)",
    )
    object_id: Optional[str] = Field(
        default=None, description="Identificador opcional del objeto"
    )

    class Config:
        extra = "allow"


class PredictRequest(BaseModel):
    items: List[PredictItem] = Field(..., description="Lista de objetos a evaluar")


app = FastAPI(
    title="Exoplanets Param API",
    description="Predicción por parámetros (sin CSV). Requiere model.joblib + model_config.json en el working dir.",
)


@app.on_event("startup")
def _startup():
    ok = _load()
    if not ok:
        print(
            "WARN: model.joblib / model_config.json not found in working dir. Load will be retried on /health."
        )


@app.get("/health")
def health():
    ok = MODEL is not None and CONFIG is not None and len(FEATURE_COLS) > 0
    return {
        "ok": ok,
        "features": FEATURE_COLS,
        "metrics": CONFIG.get("metrics", {}) if CONFIG else {},
    }


def _rows_from_items(items: List[Dict[str, Any]]) -> pd.DataFrame:
    # Build a dataframe with exactly the FEATURE_COLS (order matters)
    # Missing or null fields are set to NaN; extra fields are ignored.
    rows = []
    meta = []  # to echo back dataset/object_id if provided
    for it in items:
        # Separate meta from features
        ds = it.get("dataset")
        oid = it.get("object_id")
        meta.append({"dataset": ds, "object_id": oid})
        row = []
        for f in FEATURE_COLS:
            v = it.get(f, None)
            try:
                row.append(float(v) if v is not None else np.nan)
            except Exception:
                row.append(np.nan)
        rows.append(row)
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    meta_df = pd.DataFrame(meta)
    return X, meta_df


@app.post("/predict_params")
def predict_params(req: PredictRequest):
    if MODEL is None or CONFIG is None or len(FEATURE_COLS) == 0:
        if not _load():
            return JSONResponse(
                {
                    "error": "Modelo no cargado. Asegura model.joblib y model_config.json en el cwd."
                },
                status_code=500,
            )

    items = [it.dict() for it in req.items]
    if len(items) == 0:
        return JSONResponse({"error": "Lista 'items' vacía."}, status_code=400)

    X, meta_df = _rows_from_items(items)
    probs = MODEL.predict_proba(X)[:, 1]
    thr = float(CONFIG.get("threshold", 0.5))
    preds = (probs >= thr).astype(int)

    out = meta_df.copy()
    out["score_confirmed"] = probs
    out["pred_confirmed"] = preds

    # Preview
    preview = out.copy()
    # Include only a few feature contributions? (future: SHAP per-row)
    return {
        "threshold": thr,
        "count": int(len(out)),
        "features_expected": FEATURE_COLS,
        "predictions": out.to_dict(orient="records"),
    }


# Convenience: single-item GET with query params (handy for quick tests)
@app.get("/predict_one")
def predict_one(**query):
    if MODEL is None or CONFIG is None or len(FEATURE_COLS) == 0:
        if not _load():
            return JSONResponse(
                {
                    "error": "Modelo no cargado. Asegura model.joblib y model_config.json en el cwd."
                },
                status_code=500,
            )

    # Build single item from query string
    item = {k: (float(v) if k in FEATURE_COLS else v) for k, v in query.items()}
    X, meta_df = _rows_from_items([item])
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    thr = float(CONFIG.get("threshold", 0.5))
    pred = int(prob >= thr)

    return {
        "threshold": thr,
        "features_expected": FEATURE_COLS,
        "input_received": item,
        "prediction": {
            **meta_df.iloc[0].to_dict(),
            "score_confirmed": prob,
            "pred_confirmed": pred,
        },
    }
