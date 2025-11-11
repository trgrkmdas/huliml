"""
Model persistence helper sınıfı - Model kaydetme ve yükleme
"""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any

from ..logger import get_logger
from ..preprocessing import Preprocessor
from .base import BaseModel
from .constants import FILE_MODEL_PREFIX, FILE_MODEL_EXTENSION, FILE_SCALER_NAME

logger = get_logger("MLProject.ModelPersistence")


class ModelPersistence:
    """Model kaydetme ve yükleme için helper sınıf"""

    def __init__(self, paths_config, model_config):
        """
        Args:
            paths_config: Paths config
            model_config: Model config
        """
        self._paths_config = paths_config
        self._model_config = model_config

    def save_model(
        self,
        model: BaseModel,
        preprocessor: Optional[Preprocessor],
        feature_columns: Optional[list],
        config: Any,
        filepath: Optional[str] = None,
    ) -> str:
        """
        Model ve preprocessor'ı kaydet.

        Args:
            model: Eğitilmiş model
            preprocessor: Preprocessor (scaler)
            feature_columns: Feature column isimleri
            config: Config objesi
            filepath: Dosya yolu (None ise otomatik oluşturulur)

        Returns:
            str: Kaydedilen dosya yolu
        """
        if filepath is None:
            filepath = (
                self._paths_config.models_dir
                / f"{FILE_MODEL_PREFIX}{self._model_config.model_type}{FILE_MODEL_EXTENSION}"
            )

        filepath_path = Path(filepath)
        filepath_path.parent.mkdir(parents=True, exist_ok=True)
        filepath_str = str(filepath_path)

        with open(filepath_str, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "preprocessor": preprocessor,
                    "feature_columns": feature_columns,
                    "config": config,
                },
                f,
            )

        logger.info(f"💾 Model kaydedildi: {filepath_str}")

        # Preprocessor'ı da kaydet (ayrı dosya)
        if preprocessor is not None:
            preprocessor_path = filepath_path.parent / FILE_SCALER_NAME
            preprocessor.save_scaler(str(preprocessor_path))

        return filepath_str

    def load_model(self, filepath: str, config: Any) -> Dict[str, Any]:
        """
        Model ve preprocessor'ı yükle.

        Args:
            filepath: Dosya yolu
            config: Config objesi (Preprocessor için gerekli)

        Returns:
            Dictionary with model, preprocessor, feature_columns
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = data["model"]
        preprocessor = data.get("preprocessor")
        feature_columns = data.get("feature_columns")

        logger.info(f"📂 Model yüklendi: {filepath}")

        # Preprocessor'ı da yükle (ayrı dosya)
        if preprocessor is None:
            preprocessor_path = Path(filepath).parent / FILE_SCALER_NAME
            if preprocessor_path.exists():
                preprocessor = Preprocessor(config=config)
                preprocessor.load_scaler(str(preprocessor_path))

        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
        }
