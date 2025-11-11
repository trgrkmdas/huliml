"""
Ana preprocessing sınıfı - Scaling işlemlerini yönetir
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, List

from ..config import get_config
from ..logger import get_logger
from .scalers import RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer

logger = get_logger("MLProject.Preprocessing")


class Preprocessor:
    """Ana preprocessing sınıfı - Scaling işlemlerini yönetir"""

    def __init__(self, config=None):
        """
        Args:
            config: Config objesi (None ise get_config() kullanılır)
        """
        self.config = config or get_config()
        self.scaler = None
        self.feature_columns: Optional[List[str]] = None
        self.excluded_columns: Optional[List[str]] = None

    def _create_scaler(self):
        """Config'e göre scaler oluştur"""
        preprocess_config = self.config.preprocessing
        exclude_cols = preprocess_config.exclude_columns

        scaler_type = preprocess_config.scaler_type.lower()

        if scaler_type == "standard":
            return StandardScaler(
                exclude_columns=exclude_cols,
                with_mean=preprocess_config.standard_with_mean,
                with_std=preprocess_config.standard_with_std,
            )
        elif scaler_type == "minmax":
            return MinMaxScaler(
                exclude_columns=exclude_cols,
                feature_range=preprocess_config.minmax_feature_range,
            )
        elif scaler_type == "robust":
            return RobustScaler(
                exclude_columns=exclude_cols,
                quantile_range=preprocess_config.robust_quantile_range,
            )
        elif scaler_type == "quantile":
            return QuantileTransformer(
                exclude_columns=exclude_cols,
                n_quantiles=preprocess_config.quantile_n_quantiles,
                output_distribution=preprocess_config.quantile_output_distribution,
            )
        else:
            raise ValueError(f"Geçersiz scaler_type: {scaler_type}")

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Fit scaler on training data.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        if not self.config.preprocessing.enable_scaling:
            logger.info("⚠️  Scaling devre dışı, atlanıyor...")
            return self

        logger.info("🔧 Scaler fit ediliyor...")
        self.scaler = self._create_scaler()
        self.scaler.fit(df)
        self.feature_columns = self.scaler.feature_columns
        self.excluded_columns = self.scaler.exclude_columns

        logger.info(
            f"✅ Scaler fit edildi. {len(self.feature_columns)} feature scale edilecek."
        )
        logger.info(f"   Excluded columns: {len(self.excluded_columns)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data (train/test).

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.config.preprocessing.enable_scaling:
            return df

        if self.scaler is None:
            raise ValueError("Scaler henüz fit edilmedi. Önce fit() çağrılmalı.")

        logger.info("🔄 Veri transform ediliyor...")
        df_scaled = self.scaler.transform(df)
        logger.info("✅ Veri transform edildi.")

        return df_scaled  # type: ignore[no-any-return]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform.

        Args:
            df: DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)

    def save_scaler(self, filepath: Optional[str] = None) -> str:
        """
        Save scaler for production.

        Args:
            filepath: Dosya yolu (None ise otomatik oluşturulur)

        Returns:
            str: Kaydedilen dosya yolu
        """
        if self.scaler is None:
            raise ValueError("Scaler henüz fit edilmedi.")

        if filepath is None:
            models_dir = self.config.paths.models_dir
            filepath = models_dir / "scaler.pkl"

        filepath_path = Path(filepath)
        filepath_path.parent.mkdir(parents=True, exist_ok=True)
        filepath = str(filepath_path)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "feature_columns": self.feature_columns,
                    "excluded_columns": self.excluded_columns,
                },
                f,
            )

        logger.info(f"💾 Scaler kaydedildi: {filepath}")
        return str(filepath)

    def load_scaler(self, filepath: str) -> "Preprocessor":
        """
        Load scaler for production.

        Args:
            filepath: Dosya yolu

        Returns:
            self
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.excluded_columns = data["excluded_columns"]

        logger.info(f"📂 Scaler yüklendi: {filepath}")
        return self
