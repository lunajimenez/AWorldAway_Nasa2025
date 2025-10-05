import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
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


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}


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
