"""
Model evaluation sınıfı
"""

import numpy as np
from typing import Optional, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from ..config import get_config
from ..logger import get_logger

logger = get_logger("MLProject.ModelEvaluation")


class ModelEvaluator:
    """Model evaluation sınıfı"""

    def __init__(self, config=None):
        """
        Args:
            config: Config objesi (None ise get_config() kullanılır)
        """
        self.config = config or get_config()
        self.metrics: Dict[str, float] = {}

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Classification metriklerini hesapla.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        logger.info("📊 Classification metrikleri hesaplanıyor...")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # ROC AUC (binary classification için)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                # Binary classification için
                if y_proba.ndim > 1:
                    y_proba_binary = y_proba[:, 1]
                else:
                    y_proba_binary = y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba_binary)
            except ValueError as e:
                # ValueError: y_true veya y_proba geçersiz format veya değerler içeriyor
                logger.warning(f"ROC AUC hesaplanamadı (ValueError): {e}")
            except TypeError as e:
                # TypeError: y_true veya y_proba yanlış tip
                logger.warning(f"ROC AUC hesaplanamadı (TypeError): {e}")
            except Exception as e:
                # Diğer beklenmeyen hatalar için genel exception (fallback)
                logger.warning(f"ROC AUC hesaplanamadı (beklenmeyen hata): {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        self.metrics.update(metrics)

        logger.info("✅ Metrikler hesaplandı:")
        for key, value in metrics.items():
            if key != "confusion_matrix":
                logger.info(f"   {key}: {value:.4f}")

        return metrics

    def evaluate_regression(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Regression metriklerini hesapla.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        logger.info("📊 Regression metrikleri hesaplanıyor...")

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }

        self.metrics.update(metrics)

        logger.info("✅ Metrikler hesaplandı:")
        for key, value in metrics.items():
            logger.info(f"   {key}: {value:.4f}")

        return metrics

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_proba: Predicted probabilities (classification için)
            task_type: 'classification' or 'regression' (None ise config'den alınır)

        Returns:
            Dictionary of metrics
        """
        if task_type is None:
            task_type = self.config.feature_engineering.target_type

        if task_type == "regression":
            return self.evaluate_regression(y_true, y_pred)
        else:
            return self.evaluate_classification(y_true, y_pred, y_proba)

    def print_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Classification report yazdır.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        logger.info("\n📋 Classification Report:")
        logger.info("\n" + classification_report(y_true, y_pred))

    def get_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Confusion matrix döndür.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)  # type: ignore[no-any-return]
