import json
from pathlib import Path
from typing import Any, Dict, List

from joblib import load  # type: ignore

from utils import ROOTDIR, Singleton, logger


class ModelService(object, metaclass=Singleton):
    @property
    def model(self) -> Any:
        return self.__model

    @property
    def config(self) -> Dict[str, Any]:
        return self.__config or {}

    @property
    def feature_cols(self) -> List[str]:
        return self.__feature_cols

    def __init__(self) -> None:
        self.__model = None
        self.__config = None
        self.__feature_cols: List[str] = []
        self.is_loaded: bool = False

    def on_init(
        self,
        model_path: Path = ROOTDIR / "models" / "model.joblib",
        config_path: Path = ROOTDIR / "models" / "model_config.json",
    ):
        try:
            if not model_path.exists() or not config_path.exists():
                return False

            self.__model = load(model_path)
            self.__config = json.loads(config_path.read_text(encoding="utf-8"))
            self.__feature_cols = self.__config.get("feature_columns", [])
            self.is_loaded = True

            return True
        except Exception as E:
            logger.critical(f"error loading model: {E}")
            return False


model_service = ModelService()
