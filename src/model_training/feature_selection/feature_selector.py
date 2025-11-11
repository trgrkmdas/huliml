"""
Feature selector main class - Strategy pattern ile farklı selection metodlarını yönetir
"""

import pandas as pd
from typing import Optional, List, Dict

from ...logger import get_logger
from .base import BaseFeatureSelector
from .selectors import (
    UnivariateFeatureSelector,
    VarianceThresholdSelector,
    RFEFeatureSelector,
    CorrelationFeatureSelector,
)

logger = get_logger("MLProject.FeatureSelection")


class FeatureSelector:
    """Ana feature selection sınıfı - Strategy pattern"""

    def __init__(self, config=None):
        """
        Args:
            config: Config objesi (None ise get_config() kullanılır)
        """
        from ...config import get_config

        self.config = config or get_config()
        self.selector: Optional[BaseFeatureSelector] = None
        self.selected_features: Optional[List[str]] = None

    def _create_selector(
        self, task_type: str = "classification"
    ) -> BaseFeatureSelector:
        """Config'e göre selector oluştur"""
        fs_config = self.config.model.feature_selection
        method = fs_config.method.lower()

        exclude_columns = fs_config.exclude_columns or []

        if method == "univariate":
            return UnivariateFeatureSelector(
                method=fs_config.univariate_method,
                k=fs_config.k,
                percentile=fs_config.percentile,
                score_func=fs_config.score_func,
                exclude_columns=exclude_columns,
            )
        elif method == "rfe":
            return RFEFeatureSelector(
                n_features_to_select=fs_config.n_features_to_select,
                step=fs_config.rfe_step,
                estimator=None,  # Default RandomForest
                cv=fs_config.rfe_cv,
                exclude_columns=exclude_columns,
            )
        elif method == "variance":
            return VarianceThresholdSelector(
                threshold=fs_config.variance_threshold,
                exclude_columns=exclude_columns,
            )
        elif method == "correlation":
            return CorrelationFeatureSelector(
                threshold=fs_config.correlation_threshold,
                method=fs_config.correlation_method,
                exclude_columns=exclude_columns,
            )
        else:
            raise ValueError(f"Geçersiz feature selection method: {method}")

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        task_type: Optional[str] = None,
    ) -> tuple:
        """
        Feature selection yap.

        Args:
            X_train: Training features
            y_train: Training targets (some methods require this)
            X_test: Test features (optional, will be transformed)
            task_type: 'classification' or 'regression' (None ise config'den alınır)

        Returns:
            Tuple of (X_train_selected, X_test_selected, selected_features)
        """
        if task_type is None:
            task_type = (
                "regression"
                if self.config.feature_engineering.target_type == "regression"
                else "classification"
            )

        logger.info("🔍 Feature selection başlatılıyor...")
        fs_config = self.config.model.feature_selection
        logger.info(f"   Method: {fs_config.method}")
        logger.info(f"   Başlangıç feature sayısı: {len(X_train.columns)}")

        # Create selector
        self.selector = self._create_selector(task_type)

        # Fit selector
        if isinstance(self.selector, RFEFeatureSelector):
            if y_train is None:
                raise ValueError("RFE için y_train gerekli.")
            self.selector.fit(X_train, y_train, task_type=task_type)
        else:
            self.selector.fit(X_train, y_train)

        # Get selected features
        self.selected_features = self.selector.get_selected_features()

        logger.info("✅ Feature selection tamamlandı!")
        logger.info(f"   Seçilen feature sayısı: {len(self.selected_features)}")
        logger.info(
            f"   Azalma: {len(X_train.columns) - len(self.selected_features)} feature çıkarıldı"
        )

        # Transform training data
        X_train_selected = self.selector.transform(X_train)

        # Transform test data if provided
        X_test_selected = None
        if X_test is not None:
            X_test_selected = self.selector.transform(X_test)

        return X_train_selected, X_test_selected, self.selected_features

    def get_selected_features(self) -> List[str]:
        """Get selected feature names"""
        if self.selected_features is None:
            raise ValueError("Feature selection henüz yapılmadı.")
        return self.selected_features.copy()

    def get_feature_scores(self) -> Optional[Dict[str, float]]:
        """Get feature scores (if available)"""
        if self.selector is None:
            return None
        return self.selector.get_feature_scores()
