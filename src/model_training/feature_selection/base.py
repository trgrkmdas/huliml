"""
Base feature selector interface - Abstract base class for all feature selectors
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseFeatureSelector(ABC):
    """Base feature selector interface - sklearn transformer pattern"""

    def __init__(self, exclude_columns: Optional[List[str]] = None):
        """
        Args:
            exclude_columns: List of column names to exclude from selection
        """
        self.exclude_columns = exclude_columns or []
        self.selected_features: Optional[List[str]] = None
        self.selector = None

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get columns to select (exclude specified columns).

        Args:
            df: DataFrame

        Returns:
            List of column names to select
        """
        return [col for col in df.columns if col not in self.exclude_columns]

    @abstractmethod
    def _create_selector(self):
        """Create the underlying sklearn selector"""
        pass

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseFeatureSelector":
        """
        Fit selector on training data.

        Args:
            X: Training DataFrame
            y: Target vector (optional, some selectors need it)

        Returns:
            self
        """
        if X.empty:
            raise ValueError("DataFrame boş, fit edilemez.")

        # Determine feature columns
        feature_columns = self._get_feature_columns(X)

        if not feature_columns:
            raise ValueError("Seçilecek feature bulunamadı.")

        # Create selector
        self.selector = self._create_selector()
        assert self.selector is not None, "Selector oluşturulamadı"

        # Fit selector on feature columns only
        X_features = X[feature_columns]

        if y is not None:
            self.selector.fit(X_features, y)
        else:
            self.selector.fit(X_features)

        # Get selected features
        if hasattr(self.selector, "get_support"):
            # sklearn selectors
            support_mask = self.selector.get_support()
            self.selected_features = [
                feature_columns[i]
                for i in range(len(feature_columns))
                if support_mask[i]
            ]
        elif hasattr(self.selector, "selected_features_"):
            # Custom selectors
            self.selected_features = self.selector.selected_features_
        else:
            # Fallback: all features
            self.selected_features = feature_columns

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data (select features).

        Args:
            X: DataFrame to transform

        Returns:
            DataFrame with selected features only
        """
        if self.selector is None:
            raise ValueError("Selector henüz fit edilmedi. Önce fit() çağrılmalı.")

        if self.selected_features is None:
            raise ValueError("Selected features belirlenmemiş.")

        # Return only selected features
        return X[self.selected_features].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform.

        Args:
            X: DataFrame
            y: Target vector (optional)

        Returns:
            DataFrame with selected features
        """
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names"""
        if self.selected_features is None:
            raise ValueError("Selector henüz fit edilmedi.")
        return self.selected_features.copy()

    def get_feature_scores(self) -> Optional[dict]:
        """
        Get feature scores (if available).

        Returns:
            Dictionary of feature scores or None
        """
        if self.selector is None:
            return None

        if hasattr(self.selector, "scores_"):
            feature_columns = self._get_feature_columns(
                pd.DataFrame(columns=self.selected_features or [])
            )
            return dict(zip(feature_columns, self.selector.scores_))

        return None
