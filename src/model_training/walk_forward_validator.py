"""
Walk-forward validation helper sınıfı
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit

from ..logger import get_logger
from ..preprocessing import Preprocessor
from .evaluator import ModelEvaluator
from .constants import (
    WINDOW_TYPE_EXPANDING,
    WINDOW_TYPE_ROLLING,
    TASK_TYPE_REGRESSION,
    LOG_SEPARATOR,
)

logger = get_logger("MLProject.WalkForwardValidator")


class WalkForwardValidator:
    """Walk-forward validation için helper sınıf"""

    def __init__(self, model_config, fe_config, config):
        """
        Args:
            model_config: Model config
            fe_config: Feature engineering config
            config: Full config (Preprocessor için gerekli)
        """
        self._model_config = model_config
        self._fe_config = fe_config
        self.config = config

    def validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_factory,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: Optional[int] = None,
        window_type: Optional[str] = None,
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Walk-forward validation yap.

        Args:
            X_train: Training features
            y_train: Training targets
            model_factory: Model oluşturma fonksiyonu
            n_splits: Fold sayısı
            test_size: Test boyutu
            gap: Gap boyutu
            window_type: Window tipi
            window_size: Window boyutu

        Returns:
            Validation sonuçları
        """
        n_splits = n_splits or self._model_config.walk_forward_n_splits
        test_size = test_size or self._model_config.walk_forward_test_size
        gap = gap if gap is not None else self._model_config.walk_forward_gap
        window_type = window_type or self._model_config.walk_forward_type
        window_size = window_size or self._model_config.walk_forward_window_size

        logger.info("\n" + LOG_SEPARATOR)
        logger.info("🔄 WALK-FORWARD VALIDATION")
        logger.info(LOG_SEPARATOR)
        logger.info(f"   Window Type: {window_type.upper()}")
        logger.info(f"   Fold sayısı: {n_splits}")
        logger.info(f"   Test size: {test_size or 'Otomatik'}")
        if window_type == WINDOW_TYPE_ROLLING:
            logger.info(f"   Window size: {window_size or 'Otomatik'}")
        logger.info(f"   Gap: {gap}")

        # Walk-forward splits oluştur
        if window_type == WINDOW_TYPE_EXPANDING:
            splits = self._create_expanding_splits(X_train, n_splits, test_size, gap)
        elif window_type == WINDOW_TYPE_ROLLING:
            splits = self._create_rolling_splits(
                X_train, n_splits, test_size, gap, window_size
            )
        else:
            raise ValueError(
                f"Geçersiz window_type: {window_type}. '{WINDOW_TYPE_EXPANDING}' veya '{WINDOW_TYPE_ROLLING}' olmalı."
            )

        # Walk-forward validation
        fold_scores = []
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n📊 Fold {fold + 1}/{n_splits}:")
            logger.info(f"   Train: {len(train_idx)} satır")
            logger.info(f"   Validation: {len(val_idx)} satır")

            # Fold verilerini hazırla
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            # Preprocessing (her fold için ayrı fit)
            preprocessor_fold = Preprocessor(config=self.config)
            X_train_fold_scaled = preprocessor_fold.fit_transform(X_train_fold)
            X_val_fold_scaled = preprocessor_fold.transform(X_val_fold)

            # Model eğit
            model_fold = model_factory()
            model_fold.fit(X_train_fold_scaled, y_train_fold)

            # Prediction
            y_pred_fold = model_fold.predict(X_val_fold_scaled)
            y_proba_fold = model_fold.predict_proba(X_val_fold_scaled)

            # Evaluation
            evaluator_fold = ModelEvaluator(config=self.config)
            metrics_fold = evaluator_fold.evaluate(
                y_val_fold.values,
                y_pred_fold,
                y_proba_fold,
            )

            fold_metrics.append(metrics_fold)

            # Score kaydet (classification için accuracy, regression için r2)
            if self._fe_config.target_type == TASK_TYPE_REGRESSION:
                score = metrics_fold.get("r2_score", 0)
            else:
                score = metrics_fold.get("accuracy", 0)

            fold_scores.append(score)
            logger.info(f"   Score: {score:.4f}")

        # Sonuçları özetle
        results = {
            "fold_scores": fold_scores,
            "fold_metrics": fold_metrics,
            "mean_score": np.mean(fold_scores),
            "std_score": np.std(fold_scores),
            "min_score": np.min(fold_scores),
            "max_score": np.max(fold_scores),
        }

        logger.info("\n" + LOG_SEPARATOR)
        logger.info("📊 WALK-FORWARD VALIDATION SONUÇLARI")
        logger.info(LOG_SEPARATOR)
        logger.info(f"   Mean Score: {results['mean_score']:.4f}")
        logger.info(f"   Std Score: {results['std_score']:.4f}")
        logger.info(f"   Min Score: {results['min_score']:.4f}")
        logger.info(f"   Max Score: {results['max_score']:.4f}")
        logger.info(LOG_SEPARATOR)

        return results

    def _create_expanding_splits(
        self, X_train: pd.DataFrame, n_splits: int, test_size: Optional[int], gap: int
    ) -> list:
        """Expanding window splits oluştur"""
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        return list(tscv.split(X_train))

    def _create_rolling_splits(
        self,
        X_train: pd.DataFrame,
        n_splits: int,
        test_size: Optional[int],
        gap: int,
        window_size: Optional[int],
    ) -> list:
        """Rolling window splits oluştur"""
        n_samples = len(X_train)

        # Test size'ı belirle
        if test_size is None:
            test_size = n_samples // (n_splits + 1)

        # Window size'ı belirle
        if window_size is None:
            multiplier = self._model_config.walk_forward_window_multiplier
            percentage = self._model_config.walk_forward_window_percentage
            window_size = max(test_size * multiplier, int(n_samples * percentage))

        splits = []
        step_size = test_size

        # İlk fold için başlangıç pozisyonu
        train_start = 0
        train_end = window_size

        for fold in range(n_splits):
            # Test pozisyonları
            test_start = train_end + gap
            test_end = test_start + test_size

            # Yeterli veri var mı kontrol et
            if test_end > n_samples:
                logger.warning(
                    f"Fold {fold + 1}: Yeterli veri yok, atlanıyor. "
                    f"Gerekli: {test_end}, Mevcut: {n_samples}"
                )
                break

            # Train ve test index'leri
            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(test_start, test_end)

            splits.append((train_idx, val_idx))

            # Sonraki fold için pozisyonları güncelle (rolling)
            train_start += step_size
            train_end = train_start + window_size

            # Sonraki fold için yeterli veri var mı kontrol et
            if train_end + gap + test_size > n_samples:
                break

        logger.info(f"   Oluşturulan fold sayısı: {len(splits)}")
        return splits
