"""
RandomizedSearchCV tuner - Hızlı ve etkili
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import time

from ...config import get_config
from ...logger import get_logger
from .base import BaseTuner

logger = get_logger("MLProject.HyperparameterTuning")


class RandomizedSearchTuner(BaseTuner):
    """RandomizedSearchCV tuner - Hızlı random sampling"""

    def __init__(
        self,
        n_iter: int = 50,
        cv: int = 5,
        n_jobs: int = -1,
        scoring: str = "accuracy",
        random_state: Optional[int] = None,
        config=None,
    ):
        """
        Args:
            n_iter: Number of iterations
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs (-1 = all cores)
            scoring: Scoring metric
            random_state: Random seed (None ise config'den alınır)
            config: Config objesi (None ise get_config() kullanılır)
        """
        super().__init__()
        self.config = config or get_config()
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.random_state = random_state or self.config.model.random_seed
        self.search_cv: Optional[RandomizedSearchCV] = None

    def tune(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with RandomizedSearchCV.

        Args:
            model: Model instance (BaseModel)
            X: Feature matrix
            y: Target vector
            param_space: Parameter space (dict of lists)
            **kwargs: Additional arguments

        Returns:
            Dictionary of best parameters
        """
        logger.info("🔍 RandomizedSearchCV tuning başlatılıyor...")
        logger.info(f"   Iterations: {self.n_iter}")
        logger.info(f"   CV folds: {self.cv}")
        logger.info(f"   Parallel jobs: {self.n_jobs}")

        # Get sklearn model
        sklearn_model = model.model if hasattr(model, "model") else model

        # Create RandomizedSearchCV
        verbose_level = self.config.model.tuning.verbose_level
        self.search_cv = RandomizedSearchCV(
            estimator=sklearn_model,
            param_distributions=param_space,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
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

        logger.info("✅ RandomizedSearchCV tuning tamamlandı!")
        logger.info(f"   En iyi skor: {self.best_score:.4f}")
        logger.info(f"   Süre: {elapsed_time:.2f} saniye")
        logger.info("   En iyi parametreler:")
        for key, value in self.best_params.items():
            logger.info(f"      {key}: {value}")

        return self.best_params
