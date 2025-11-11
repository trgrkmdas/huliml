"""
Optuna tuner - Bayesian optimization (En gelişmiş ve hızlı)
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.base import clone
import time

from ...config import get_config
from ...logger import get_logger
from .base import BaseTuner

logger = get_logger("MLProject.HyperparameterTuning")


class OptunaTuner(BaseTuner):
    """Optuna tuner - Bayesian optimization with TPE sampler"""

    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        direction: str = "maximize",
        use_pruning: bool = True,
        config=None,
    ):
        """
        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds (None = no timeout)
            n_jobs: Number of parallel jobs (Optuna için genelde 1)
            study_name: Study name for Optuna
            direction: 'maximize' or 'minimize'
            use_pruning: Use pruning to stop unpromising trials early
            config: Config objesi (None ise get_config() kullanılır)
        """
        super().__init__()
        self.config = config or get_config()
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name or "hyperparameter_tuning"
        self.direction = direction
        self.use_pruning = use_pruning
        self.study: Optional[optuna.Study] = None

    def _create_study(self) -> optuna.Study:
        """Create Optuna study"""
        random_seed = self.config.model.random_seed
        sampler = TPESampler(seed=random_seed)  # Tree-structured Parzen Estimator
        pruner = MedianPruner() if self.use_pruning else None

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
        )

        return study

    def _objective(
        self,
        trial: optuna.Trial,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Dict[str, Any],
        cv: int,
        scoring: str,
        use_time_series_split: bool,
    ) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial
            model: Model instance
            X: Feature matrix
            y: Target vector
            param_space: Parameter space definition
            cv: Cross-validation folds
            scoring: Scoring metric

        Returns:
            CV score
        """
        # Suggest parameters from space
        params: Dict[str, Any] = {}
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                # Optuna suggest method
                suggest_type = param_config.get("type", "categorical")

                if suggest_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                        log=param_config.get("log", False),
                    )
                elif suggest_type == "float":
                    params[param_name] = float(
                        trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            step=param_config.get("step", None),
                            log=param_config.get("log", False),
                        )
                    )
                elif suggest_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
            elif isinstance(param_config, list):
                # Simple list - categorical
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            else:
                raise ValueError(f"Geçersiz param_config tipi: {type(param_config)}")

        # Model wrapper'dan sklearn model'i al
        if hasattr(model, "model") and model.model is not None:
            # BaseModel wrapper kullanılıyor - sklearn model'i clone et
            sklearn_model = clone(model.model)
        elif hasattr(model, "_create_model"):
            # BaseModel wrapper ama henüz model oluşturulmamış - yeni model oluştur
            model_clone = type(model)(
                params=model.params.copy(),
                task_type=getattr(model, "task_type", "classification"),
            )
            sklearn_model = model_clone._create_model()
        else:
            # Direkt sklearn model
            sklearn_model = clone(model)

        # Parametreleri sklearn model'e set et
        sklearn_model.set_params(**params)

        # Time series için TimeSeriesSplit kullan
        if use_time_series_split:
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = cv

        cv_scores = cross_val_score(
            sklearn_model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=self.n_jobs
        )

        score = cv_scores.mean()

        # Report intermediate value for pruning
        trial.report(score, step=len(cv_scores))

        # Pruning check
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(score)  # Ensure float return type

    def tune(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Dict[str, Any],
        cv: Optional[int] = None,
        scoring: Optional[str] = None,
        use_time_series_split: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with Optuna.

        Args:
            model: Model instance (BaseModel)
            X: Feature matrix
            y: Target vector
            param_space: Parameter space definition
            cv: Cross-validation folds
            scoring: Scoring metric
            **kwargs: Additional arguments

        Returns:
            Dictionary of best parameters
        """
        # Config'den varsayılan değerleri al
        tuning_config = self.config.model.tuning
        cv = cv if cv is not None else tuning_config.randomized_cv
        scoring = scoring if scoring is not None else tuning_config.scoring
        use_time_series_split = (
            use_time_series_split
            if use_time_series_split is not None
            else tuning_config.use_time_series_split
        )

        logger.info("🔍 Optuna Bayesian Optimization başlatılıyor...")
        logger.info(f"   Trials: {self.n_trials}")
        logger.info(f"   Timeout: {self.timeout or 'Yok'}")
        logger.info(f"   CV folds: {cv}")
        logger.info(f"   Scoring: {scoring}")

        # Create study
        self.study = self._create_study()

        # Create objective function
        def objective(trial):
            return self._objective(
                trial, model, X, y, param_space, cv, scoring, use_time_series_split
            )

        # Optimize
        start_time = time.time()

        try:
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.warning("⚠️  Tuning kullanıcı tarafından durduruldu.")

        elapsed_time = time.time() - start_time

        # Get best results
        if len(self.study.trials) == 0:
            raise ValueError("Hiç trial çalıştırılamadı.")

        self.best_params = self.study.best_params.copy()
        self.best_score = self.study.best_value

        # Tuning history
        self.tuning_history = [
            {
                "trial": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name,
            }
            for trial in self.study.trials
        ]

        logger.info("✅ Optuna tuning tamamlandı!")
        logger.info(f"   En iyi skor: {self.best_score:.4f}")
        logger.info(f"   Denenen trial sayısı: {len(self.study.trials)}")
        logger.info(f"   Süre: {elapsed_time:.2f} saniye")
        logger.info("   En iyi parametreler:")
        for key, value in self.best_params.items():
            logger.info(f"      {key}: {value}")

        return self.best_params

    def get_study(self) -> Optional[optuna.Study]:
        """Get Optuna study object"""
        return self.study
