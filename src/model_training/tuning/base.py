"""
Base tuner interface - Abstract base class for all tuners
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class BaseTuner(ABC):
    """Base tuner interface"""

    def __init__(self):
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.tuning_history: List[Dict[str, Any]] = []

    @abstractmethod
    def tune(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        param_space: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.

        Args:
            model: Model instance (BaseModel)
            X: Feature matrix
            y: Target vector
            param_space: Parameter space to search
            **kwargs: Additional arguments

        Returns:
            Dictionary of best parameters
        """
        pass

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

    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """Get tuning history"""
        return self.tuning_history.copy()
