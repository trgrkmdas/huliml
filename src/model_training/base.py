"""
Base model interface - Abstract base class for all models
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Base model interface - sklearn-like API"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params: Model parametreleri
        """
        self.params = params or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def _create_model(self):
        """Create the underlying model"""
        pass

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional arguments

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:
        """
        Predict probabilities (for classification).

        Args:
            X: Feature matrix

        Returns:
            Probability array or None if not applicable
        """
        pass

    @abstractmethod
    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> Optional[Dict[str, float]]:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'split', etc.)

        Returns:
            Dictionary of feature importance or None
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters"""
        self.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self
