from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from config import EnvConfig
from utils import ROOTDIR
import datetime

load_dotenv()
settings = EnvConfig()
app = FastAPI(debug=settings.debug)

app.mount(
    str((ROOTDIR / "assets").absolute()), StaticFiles(directory="assets"), name="assets"
)


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}


@app.get("/favicon.ico")
def favicon():
    return FileResponse("assets/favicon.ico")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
