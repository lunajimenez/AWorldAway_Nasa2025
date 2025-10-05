from pydantic_settings import BaseSettings, SettingsConfigDict

from utils import ROOTDIR


class EnvConfig(BaseSettings):
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=ROOTDIR / ".env" if (ROOTDIR / ".env").exists() else None,
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


if __name__ == "__main__":
    pass
