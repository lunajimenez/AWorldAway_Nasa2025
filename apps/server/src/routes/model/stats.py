from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services import model_service

router = APIRouter()


@router.get("/stats")
def get_model_stats():
    """
    Retorna estadísticas completas del modelo:
    - Métricas de rendimiento (ROC_AUC, PR_AUC, F1, etc.)
    - Información de features (numéricas, categóricas)
    - Configuración del modelo
    - Threshold de decisión
    """
    if not model_service.is_loaded:
        return JSONResponse(
            {"error": "Model not loaded", "loaded": False}, status_code=503
        )

    return JSONResponse(model_service.get_model_info(), status_code=200)


@router.get("/features")
def get_features_info():
    """
    Retorna información detallada de las features esperadas por el modelo.
    Útil para construir formularios dinámicos en el frontend.
    """
    if not model_service.is_loaded:
        return JSONResponse(
            {"error": "Model not loaded", "loaded": False}, status_code=503
        )

    return JSONResponse(model_service.get_feature_info(), status_code=200)


@router.get("/metrics")
def get_metrics():
    """
    Retorna solo las métricas de rendimiento del modelo.
    """
    if not model_service.is_loaded:
        return JSONResponse(
            {"error": "Model not loaded", "loaded": False}, status_code=503
        )

    return JSONResponse(
        {
            "threshold": model_service.threshold,
            "metrics": model_service.detailed_metrics,
        },
        status_code=200,
    )


@router.get("/health")
def health_check():
    """
    Health check detallado del servicio de ML.
    """
    return JSONResponse(
        {
            "status": "healthy" if model_service.is_loaded else "unhealthy",
            "model_loaded": model_service.is_loaded,
            "features_count": len(model_service.feature_cols),
            "threshold": model_service.threshold,
            "metrics_available": bool(model_service.detailed_metrics),
        },
        status_code=200 if model_service.is_loaded else 503,
    )
