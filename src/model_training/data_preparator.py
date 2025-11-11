"""
Data preparation helper sınıfı - Veri hazırlama ve split işlemleri
"""

import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split

from ..logger import get_logger
from .constants import COLUMN_DATETIME, COLUMN_FUTURE_RETURN, TASK_TYPE_BINARY

logger = get_logger("MLProject.DataPreparator")


class DataPreparator:
    """Veri hazırlama ve split işlemleri için helper sınıf"""

    def __init__(self, preprocessing_config, fe_config, model_config):
        """
        Args:
            preprocessing_config: Preprocessing config
            fe_config: Feature engineering config
            model_config: Model config
        """
        self._preprocessing_config = preprocessing_config
        self._fe_config = fe_config
        self._model_config = model_config

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        exclude_columns: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        X ve y'yi ayır.

        Args:
            df: DataFrame
            target_column: Target column adı
            exclude_columns: Exclude edilecek kolonlar

        Returns:
            Tuple of (X, y, feature_columns)
        """
        logger.info("📊 Veri hazırlanıyor...")

        if exclude_columns is None:
            # Config'den exclude columns'ı al, yoksa varsayılan değerleri kullan
            exclude_columns = self._preprocessing_config.exclude_columns.copy()
            # datetime ve future_return her zaman exclude edilmeli
            if COLUMN_DATETIME not in exclude_columns:
                exclude_columns.append(COLUMN_DATETIME)
            if COLUMN_FUTURE_RETURN not in exclude_columns:
                exclude_columns.append(COLUMN_FUTURE_RETURN)

        # X ve y'yi ayır
        feature_columns = [
            col for col in df.columns if col not in exclude_columns + [target_column]
        ]

        X = df[feature_columns]
        y = df[target_column]

        logger.info(
            f"✅ Veri hazırlandı: {len(X)} satır, {len(feature_columns)} feature"
        )
        logger.info(f"   Target dağılımı: {y.value_counts().to_dict()}")

        return X, y, feature_columns

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Train/test split yap.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test size ratio
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self._model_config.test_size
        random_state = random_state or self._model_config.random_seed

        logger.info("✂️  Train/Test split yapılıyor...")

        # Stratify için kontrol
        stratify = None
        if self._fe_config.target_type == TASK_TYPE_BINARY:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        logger.info("✅ Split tamamlandı:")
        logger.info(f"   Train: {len(X_train)} satır")
        logger.info(f"   Test: {len(X_test)} satır")

        return X_train, X_test, y_train, y_test
