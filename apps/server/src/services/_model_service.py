import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    @property
    def numeric_columns(self) -> List[str]:
        return self.__numeric_columns

    @property
    def categorical_columns(self) -> List[str]:
        return self.__categorical_columns

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.__metrics

    @property
    def detailed_metrics(self) -> Dict[str, Any]:
        return self.__detailed_metrics

    @property
    def threshold(self) -> float:
        return self.__threshold

    def __init__(self) -> None:
        self.__model = None
        self.__config: Dict[str, Any] = {}
        self.__feature_cols: List[str] = []
        self.__numeric_columns: List[str] = []
        self.__categorical_columns: List[str] = []
        self.__metrics: Dict[str, Any] = {}
        self.__detailed_metrics: Dict[str, Any] = {}
        self.__threshold: float = 0.5
        self.is_loaded: bool = False

    def on_init(
        self,
        model_path: Path = ROOTDIR / "models" / "model.joblib",
        config_path: Path = ROOTDIR / "models" / "model_config.json",
        metrics_path: Path = ROOTDIR / "models" / "metrics.json",
        detailed_metrics_path: Optional[Path] = ROOTDIR
        / "models"
        / "metrics_detailed.json",
    ):
        try:
            if not model_path.exists() or not config_path.exists():
                logger.warning(f"Model files not found at {model_path.parent}")
                return False

            self.__model = load(model_path)
            logger.info(f"✅ Model loaded from {model_path}")

            self.__config: Dict[str, Any] = json.loads(
                config_path.read_text(encoding="utf-8")
            )
            logger.info(f"✅ Config loaded from {config_path}")

            self.__numeric_columns = self.__config.get("numeric_columns", [])
            self.__categorical_columns = self.__config.get("categorical_columns", [])
            self.__feature_cols = self.__numeric_columns + self.__categorical_columns

            self.__threshold = float(self.__config.get("threshold", 0.5))

            self.__metrics = self.__config.get("metrics", {})

            if metrics_path.exists():
                self.__detailed_metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                logger.info(f"✅ Detailed metrics loaded from {metrics_path}")
            else:
                logger.warning(f"⚠️  Metrics file not found: {metrics_path}")
                self.__detailed_metrics = self.__metrics

            if detailed_metrics_path and detailed_metrics_path.exists():
                extra_metrics = json.loads(
                    detailed_metrics_path.read_text(encoding="utf-8")
                )
                self.__detailed_metrics.update(extra_metrics)
                logger.info(f"✅ Extra metrics loaded from {detailed_metrics_path}")

            self.is_loaded = True
            logger.info(
                f"🚀 Model ready | Features: {len(self.__feature_cols)} | Threshold: {self.__threshold:.4f}"
            )

            return True

        except Exception as e:
            logger.critical(f"❌ Error loading model: {e}")
            return False

    def get_feature_info(self) -> Dict[str, Any]:
        """Retorna información detallada de las features esperadas."""
        return {
            "total_features": len(self.__feature_cols),
            "numeric_features": {
                "count": len(self.__numeric_columns),
                "columns": self.__numeric_columns,
            },
            "categorical_features": {
                "count": len(self.__categorical_columns),
                "columns": self.__categorical_columns,
            },
            "all_features": self.__feature_cols,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información completa del modelo."""
        return {
            "loaded": self.is_loaded,
            "threshold": self.__threshold,
            "metrics": self.__detailed_metrics,
            "features": self.get_feature_info(),
            "config": {
                "random_state": self.__config.get("random_state"),
                "test_size": self.__config.get("test_size"),
                "source_csv": self.__config.get("source_csv"),
            },
        }


model_service = ModelService()
