"""
Base scaler interface - Abstract base class for all scalers
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseScaler(ABC):
    """Base scaler interface - sklearn transformer pattern"""

    def __init__(self, exclude_columns: Optional[List[str]] = None):
        """
        Args:
            exclude_columns: List of column names to exclude from scaling
        """
        self.exclude_columns = exclude_columns or []
        self.feature_columns: Optional[List[str]] = None
        self.scaler = None

    @abstractmethod
    def _create_scaler(self):
        """Create the underlying sklearn scaler"""
        pass

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get columns to scale (exclude specified columns).

        Args:
            df: DataFrame

        Returns:
            List of column names to scale
        """
        return [col for col in df.columns if col not in self.exclude_columns]

    def fit(self, X: pd.DataFrame) -> "BaseScaler":
        """
        Fit scaler on training data.

        Args:
            X: Training DataFrame

        Returns:
            self
        """
        if X.empty:
            raise ValueError("DataFrame boş, fit edilemez.")

        # Determine feature columns
        self.feature_columns = self._get_feature_columns(X)

        if not self.feature_columns:
            raise ValueError("Scale edilecek feature bulunamadı.")

        # Create scaler
        self.scaler = self._create_scaler()
        assert self.scaler is not None, "Scaler oluşturulamadı"

        # Fit scaler on feature columns only
        X_features = X[self.feature_columns]
        self.scaler.fit(X_features)

        return self

    def _validate_scaler_state(self) -> None:
        """
        Scaler'ın fit edilip edilmediğini kontrol et.

        Raises:
            ValueError: Scaler fit edilmemişse veya feature columns belirlenmemişse
        """
        if self.scaler is None:
            raise ValueError("Scaler henüz fit edilmedi. Önce fit() çağrılmalı.")

        if self.feature_columns is None:
            raise ValueError("Feature columns belirlenmemiş.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Args:
            X: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        self._validate_scaler_state()
        assert self.scaler is not None, "Scaler None olamaz"

        # Copy DataFrame to avoid modifying original
        X_transformed = X.copy()

        # Transform feature columns
        X_features = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_features)

        # Update DataFrame with scaled values
        X_transformed[self.feature_columns] = X_scaled

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform.

        Args:
            X: DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform (for predictions).

        Args:
            X: DataFrame to inverse transform

        Returns:
            Inverse transformed DataFrame
        """
        self._validate_scaler_state()
        assert self.scaler is not None, "Scaler None olamaz"

        # Copy DataFrame to avoid modifying original
        X_inverse = X.copy()

        # Inverse transform feature columns
        X_features = X[self.feature_columns]
        X_inverse_features = self.scaler.inverse_transform(X_features)

        # Update DataFrame with inverse transformed values
        X_inverse[self.feature_columns] = X_inverse_features

        return X_inverse
