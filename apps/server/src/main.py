import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import EnvConfig
from routes import router
from services import model_service
from utils import LOGGING_CONFIG, ROOTDIR, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    success = model_service.on_init()

    if success:
        logger.info("model loaded succesfully")

    yield


load_dotenv()
settings = EnvConfig()
app = FastAPI(debug=settings.debug, lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOTDIR / "assets"), name="assets")
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    ok = (
        model_service.is_loaded
        and model_service.model is not None
        and len(model_service.feature_cols) > 0
    )

    return JSONResponse(
        {
            "ok": ok,
            "model_loaded": model_service.is_loaded,
            "features_count": len(model_service.feature_cols),
            "features": model_service.feature_cols,
            "metrics": model_service.metrics,
            "timestamp": datetime.datetime.now().isoformat(),
        },
        status_code=200 if ok else 503,
    )


@app.get("/favicon.ico")
def favicon():
    return FileResponse("assets/favicon.ico")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_config=LOGGING_CONFIG,
    )
