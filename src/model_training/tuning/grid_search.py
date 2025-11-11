"""
GridSearchCV tuner - Exhaustive search
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import time

from ...config import get_config
from ...logger import get_logger
from .base import BaseTuner

logger = get_logger("MLProject.HyperparameterTuning")


class GridSearchTuner(BaseTuner):
    """GridSearchCV tuner - Exhaustive grid search"""

    def __init__(
        self,
        cv: int = 5,
        n_jobs: int = -1,
        scoring: str = "accuracy",
        config=None,
    ):
        """
        Args:
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs (-1 = all cores)
            scoring: Scoring metric
            config: Config objesi (None ise get_config() kullanılır)
        """
        super().__init__()
        self.config = config or get_config()
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.search_cv: Optional[GridSearchCV] = None

    def tune(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with GridSearchCV.

        Args:
            model: Model instance (BaseModel)
            X: Feature matrix
            y: Target vector
            param_space: Parameter grid (dict of lists)
            **kwargs: Additional arguments

        Returns:
            Dictionary of best parameters
        """
        # Calculate total combinations
        total_combinations = 1
        for values in param_space.values():
            total_combinations *= len(values)

        logger.info("🔍 GridSearchCV tuning başlatılıyor...")
        logger.info(f"   Toplam kombinasyon: {total_combinations}")
        logger.info(f"   CV folds: {self.cv}")
        logger.info(f"   Parallel jobs: {self.n_jobs}")

        threshold = self.config.model.tuning.grid_combination_threshold
        if total_combinations > threshold:
            logger.warning(
                f"⚠️  Çok fazla kombinasyon ({total_combinations}). "
                "RandomizedSearchCV kullanmanız önerilir."
            )

        # Get sklearn model
        sklearn_model = model.model if hasattr(model, "model") else model

        # Create GridSearchCV
        verbose_level = self.config.model.tuning.verbose_level
        self.search_cv = GridSearchCV(
            estimator=sklearn_model,
            param_grid=param_space,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=verbose_level,
            **kwargs,
        )

        # Tuning
        start_time = time.time()
        self.search_cv.fit(X, y)
        elapsed_time = time.time() - start_time

        # Get best results
        self.best_params = self.search_cv.best_params_.copy()
        self.best_score = self.search_cv.best_score_

        # Tuning history
        self.tuning_history = [
            {
                "params": params,
                "score": score,
            }
            for params, score in zip(
                self.search_cv.cv_results_["params"],
                self.search_cv.cv_results_[f"mean_test_{self.scoring}"],
            )
        ]

        logger.info("✅ GridSearchCV tuning tamamlandı!")
        logger.info(f"   En iyi skor: {self.best_score:.4f}")
        logger.info(f"   Süre: {elapsed_time:.2f} saniye")
        logger.info("   En iyi parametreler:")
        for key, value in self.best_params.items():
            logger.info(f"      {key}: {value}")

        return self.best_params
