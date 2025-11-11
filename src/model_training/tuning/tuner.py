"""
HyperparameterTuner ana sınıfı - Strategy pattern ile farklı tuning metodlarını yönetir
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import time

from ...config import get_config
from ...logger import get_logger
from .base import BaseTuner
from .optuna_tuner import OptunaTuner
from .randomized_search import RandomizedSearchTuner
from .grid_search import GridSearchTuner

logger = get_logger("MLProject.HyperparameterTuning")


class HyperparameterTuner:
    """Ana hyperparameter tuning sınıfı - Strategy pattern"""

    def __init__(self, config=None):
        """
        Args:
            config: Config objesi (None ise get_config() kullanılır)
        """
        self.config = config or get_config()
        self.tuner: Optional[BaseTuner] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.tuning_time: Optional[float] = None

    def _create_tuner(self) -> BaseTuner:
        """Config'e göre tuner oluştur"""
        tuning_config = self.config.model.tuning
        method = tuning_config.tuning_method.lower()

        if method == "optuna":
            return OptunaTuner(
                n_trials=tuning_config.optuna_n_trials,
                timeout=tuning_config.optuna_timeout,
                n_jobs=tuning_config.n_jobs,
                study_name=tuning_config.study_name,
                direction=tuning_config.direction,
                use_pruning=tuning_config.use_pruning,
                config=self.config,
            )
        elif method == "randomized":
            return RandomizedSearchTuner(
                n_iter=tuning_config.randomized_n_iter,
                cv=tuning_config.randomized_cv,
                n_jobs=tuning_config.n_jobs,
                scoring=tuning_config.scoring,
                random_state=self.config.model.random_seed,
                config=self.config,
            )
        elif method == "grid":
            return GridSearchTuner(
                cv=tuning_config.grid_cv,
                n_jobs=tuning_config.n_jobs,
                scoring=tuning_config.scoring,
                config=self.config,
            )
        else:
            raise ValueError(f"Geçersiz tuning_method: {method}")

    def tune(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning yap.

        Args:
            model: Model instance (BaseModel)
            X: Feature matrix
            y: Target vector
            param_space: Parameter space (None ise config'den alınır)
            **kwargs: Additional arguments

        Returns:
            Dictionary of best parameters
        """
        tuning_config = self.config.model.tuning

        if param_space is None:
            param_space = tuning_config.param_grids

        if param_space is None or len(param_space) == 0:
            raise ValueError("Parametre space tanımlanmamış.")

        logger.info("=" * 60)
        logger.info("🔍 HYPERPARAMETER TUNING")
        logger.info("=" * 60)
        logger.info(f"   Method: {tuning_config.tuning_method.upper()}")
        logger.info(f"   Parametre sayısı: {len(param_space)}")

        # Tuner oluştur
        self.tuner = self._create_tuner()

        # Tuning
        start_time = time.time()
        self.best_params = self.tuner.tune(model, X, y, param_space, **kwargs)
        self.tuning_time = time.time() - start_time
        self.best_score = self.tuner.get_best_score()

        logger.info("=" * 60)
        logger.info("✅ HYPERPARAMETER TUNING TAMAMLANDI!")
        logger.info("=" * 60)
        logger.info(f"   En iyi skor: {self.best_score:.4f}")
        logger.info(f"   Tuning süresi: {self.tuning_time:.2f} saniye")
        logger.info("=" * 60)

        return self.best_params

    def incremental_tune(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        coarse_space: Dict[str, Any],
        fine_space: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        İki aşamalı incremental tuning (coarse → fine).

        Args:
            model: Model instance
            X: Feature matrix
            y: Target vector
            coarse_space: Coarse parametre space (geniş aralıklar)
            fine_space: Fine parametre space (None ise otomatik oluşturulur)
            **kwargs: Additional arguments

        Returns:
            Dictionary of best parameters
        """
        logger.info("=" * 60)
        logger.info("🔍 INCREMENTAL TUNING (Coarse → Fine)")
        logger.info("=" * 60)

        # 1. Coarse tuning
        logger.info("\n📊 AŞAMA 1: Coarse Tuning (Geniş Aralıklar)")
        tuning_config = self.config.model.tuning

        # Coarse tuning için n_iter azalt
        original_n_iter = tuning_config.randomized_n_iter
        tuning_config.randomized_n_iter = tuning_config.coarse_n_iter

        coarse_tuner = self._create_tuner()
        coarse_best = coarse_tuner.tune(model, X, y, coarse_space, **kwargs)
        coarse_score = coarse_tuner.get_best_score()

        logger.info(f"   Coarse best score: {coarse_score:.4f}")
        logger.info(f"   Coarse best params: {coarse_best}")

        # 2. Fine tuning (best çevresinde)
        if fine_space is None:
            fine_space = self._create_fine_space(coarse_best, coarse_space)

        logger.info("\n📊 AŞAMA 2: Fine Tuning (Dar Aralıklar)")
        tuning_config.randomized_n_iter = tuning_config.fine_n_iter

        fine_tuner = self._create_tuner()
        fine_best = fine_tuner.tune(model, X, y, fine_space, **kwargs)
        fine_score = fine_tuner.get_best_score()

        logger.info(f"   Fine best score: {fine_score:.4f}")
        logger.info(f"   Fine best params: {fine_best}")

        # Restore original n_iter
        tuning_config.randomized_n_iter = original_n_iter

        # En iyi sonucu seç
        if fine_score > coarse_score:
            self.best_params = fine_best
            self.best_score = fine_score
            logger.info("\n✅ Fine tuning daha iyi sonuç verdi!")
        else:
            self.best_params = coarse_best
            self.best_score = coarse_score
            logger.info("\n✅ Coarse tuning daha iyi sonuç verdi!")

        logger.info("=" * 60)
        logger.info("✅ INCREMENTAL TUNING TAMAMLANDI!")
        logger.info("=" * 60)

        return self.best_params

    def _create_fine_space(
        self, best_params: Dict[str, Any], coarse_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fine parametre space oluştur (best çevresinde).

        Args:
            best_params: Coarse tuning'den gelen best parametreler
            coarse_space: Coarse parametre space

        Returns:
            Fine parametre space
        """
        fine_space = {}

        for param_name, best_value in best_params.items():
            if param_name not in coarse_space:
                continue

            coarse_values = coarse_space[param_name]

            if isinstance(coarse_values, list):
                # List ise, best value çevresinde dar aralık oluştur
                if isinstance(best_value, (int, float)):
                    # Numeric değer
                    range_ratio = self.config.model.tuning.fine_space_range_ratio
                    if isinstance(best_value, int):
                        # Integer için config'den aralık oranı
                        step = max(1, int(best_value * range_ratio))
                        fine_space[param_name] = [
                            max(1, best_value - step),
                            best_value,
                            best_value + step,
                        ]
                    else:
                        # Float için config'den aralık oranı
                        step = best_value * range_ratio
                        # Type ignore: list can contain mixed types (int/float)
                        fine_space[param_name] = [  # type: ignore[assignment]
                            max(0.001, best_value - step),  # type: ignore[list-item]
                            best_value,  # type: ignore[list-item]
                            best_value + step,  # type: ignore[list-item]
                        ]
                else:
                    # Categorical - best value ve yakın değerler
                    idx = coarse_values.index(best_value)
                    fine_space[param_name] = [
                        coarse_values[max(0, idx - 1)],
                        best_value,
                        coarse_values[min(len(coarse_values) - 1, idx + 1)],
                    ]
            elif isinstance(coarse_values, dict):
                # Optuna format - fine space oluştur
                range_ratio = self.config.model.tuning.fine_space_range_ratio
                if coarse_values.get("type") == "int":
                    step = max(1, int(best_value * range_ratio))
                    fine_space[param_name] = {  # type: ignore[assignment]
                        "type": "int",
                        "low": max(1, best_value - step),
                        "high": best_value + step,
                        "step": 1,
                    }
                elif coarse_values.get("type") == "float":
                    step = best_value * range_ratio
                    fine_space[param_name] = {  # type: ignore[assignment]
                        "type": "float",
                        "low": max(0.001, best_value - step),
                        "high": best_value + step,
                        "step": None,
                    }

        return fine_space

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters"""
        if self.best_params is None:
            raise ValueError("Tuning henüz yapılmadı.")
        return self.best_params.copy()

    def get_best_score(self) -> float:
        """Get best score"""
        if self.best_score is None:
            raise ValueError("Tuning henüz yapılmadı.")
        return self.best_score

    def get_tuning_history(self) -> list:
        """Get tuning history"""
        if self.tuner is None:
            raise ValueError("Tuning henüz yapılmadı.")
        return self.tuner.get_tuning_history()

    def apply_best_params(self, model) -> None:
        """
        Best parametreleri modele uygula.

        Args:
            model: Model instance
        """
        if self.best_params is None:
            raise ValueError("Tuning henüz yapılmadı.")

        model.set_params(**self.best_params)
        logger.info("✅ Best parametreler modele uygulandı.")
