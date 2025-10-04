from fastapi import APIRouter
from .predict import router as predict_router
from .stats import router as stats_router

router = APIRouter(prefix="/model", tags=["model"])

router.include_router(predict_router)
router.include_router(stats_router)

__all__ = ["router"]
