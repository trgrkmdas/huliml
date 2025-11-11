"""
Correlation-based feature selection - Remove highly correlated features
"""

import pandas as pd
from typing import List, Optional

from ..base import BaseFeatureSelector


class CorrelationFeatureSelector(BaseFeatureSelector):
    """Correlation-based feature selection - Remove highly correlated features"""

    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "pearson",
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            threshold: Correlation threshold - features with correlation above this will be removed
            method: Correlation method ('pearson', 'kendall', 'spearman')
            exclude_columns: List of column names to exclude from selection
        """
        super().__init__(exclude_columns)
        self.threshold = threshold
        self.method = method
        self.selector = None  # Not using sklearn selector

    def _create_selector(self):
        """Not used - correlation selection is custom"""
        return None

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CorrelationFeatureSelector":
        """
        Fit selector on training data.

        Args:
            X: Training DataFrame
            y: Target vector (not used, kept for interface compatibility)

        Returns:
            self
        """
        if X.empty:
            raise ValueError("DataFrame boş, fit edilemez.")

        # Determine feature columns
        feature_columns = self._get_feature_columns(X)

        if not feature_columns:
            raise ValueError("Seçilecek feature bulunamadı.")

        # Calculate correlation matrix
        X_features = X[feature_columns]
        # Type ignore: pandas corr accepts string method parameter
        corr_matrix = X_features.corr(method=self.method).abs()  # type: ignore[arg-type]

        # Find highly correlated feature pairs
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                # Type ignore: pandas can return various types, but we expect numeric
                if isinstance(corr_value, (int, float)) and corr_value > self.threshold:
                    # Remove the feature with lower index (arbitrary choice)
                    to_remove.add(corr_matrix.columns[j])

        # Selected features are those not in to_remove
        self.selected_features = [
            col for col in feature_columns if col not in to_remove
        ]

        # Create a dummy selector for interface compatibility
        self.selector = type("DummySelector", (), {"get_support": lambda: None})()

        return self
