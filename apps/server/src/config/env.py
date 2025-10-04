from pydantic_settings import BaseSettings
from utils import ROOTDIR


class EnvConfig(BaseSettings):
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ROOTDIR / ".env", "env_file_encoding": "utf-8"}


if __name__ == "__main__":
    pass
