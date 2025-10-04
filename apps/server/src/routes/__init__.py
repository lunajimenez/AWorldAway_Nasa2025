from fastapi import APIRouter
from .model import router as model_router

router = APIRouter(prefix="/api")

router.include_router(model_router)

__all__ = ["router"]
