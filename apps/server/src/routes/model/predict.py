# type: ignore

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services import model_service

router = APIRouter()


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


def _rows_from_items(items: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[List[float]] = []
    meta: List[Dict[str, Any]] = []

    for it in items:
        ds = it.get("dataset")
        oid = it.get("object_id")

        meta.append({"dataset": ds, "object_id": oid})

        row = []

        for f in model_service.feature_cols:
            v = it.get(f, None)
            try:
                row.append(float(v) if v is not None else np.nan)
            except Exception:
                row.append(np.nan)

        rows.append(row)

    return pd.DataFrame(rows, columns=model_service.feature_cols), pd.DataFrame(meta)


@router.post("/predict-params")
def predict_params(request: PredictRequest):
    if not model_service.is_loaded or len(model_service.feature_cols) == 0:
        return JSONResponse({"error": "model is not loaded yet"}, status_code=500)

    items = [item.model_dump() for item in request.items]
    if len(items) == 0:
        return JSONResponse({"error": "list items is empty"}, status_code=400)

    X, meta_df = _rows_from_items(items)
    probs = model_service.model.predict_proba(X)[:, 1]
    threshold = float(model_service.config.get("threshold", 0.5))
    predictions = (probs >= threshold).astype(int)

    out = meta_df.copy()
    out["score_confirmed"] = probs
    out["pred_confirmed"] = predictions

    return JSONResponse(
        {
            "threshold": threshold,
            "count": int(len(out)),
            "features_expected": model_service.feature_cols,
            "predictions": out.to_dict(orient="records"),
        },
        status_code=200,
    )


@router.get("/predict-one")
def predict_one(request: Request):
    query = dict(request.query_params)

    if not model_service.is_loaded or len(model_service.feature_cols) == 0:
        return JSONResponse({"error": "model is not loaded yet"}, status_code=500)

    item = {
        k: (float(v) if k in model_service.feature_cols else v)
        for k, v in query.items()
    }

    X, meta_df = _rows_from_items([item])
    prob = model_service.model.predict_proba(X)[:, 1]
    threshold = float(model_service.config.get("threshold", 0.5))
    prediction = (prob >= threshold).astype(int)

    return JSONResponse(
        {
            "threshold": threshold,
            "features_expected": model_service.feature_cols,
            "input_received": item,
            "prediction": {
                **meta_df.iloc[0].to_dict(),
                "score_confirmed": prob,
                "prediction_confirmed": prediction,
            },
        },
        status_code=200,
    )
