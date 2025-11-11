"""
Recursive Feature Elimination (RFE) - Model-based feature selection
"""

import pandas as pd
from typing import List, Optional, Any
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..base import BaseFeatureSelector


class RFEFeatureSelector(BaseFeatureSelector):
    """Recursive Feature Elimination - Model-based feature selection"""

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        estimator: Optional[Any] = None,
        cv: Optional[int] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            n_features_to_select: Number of features to select (None for RFECV)
            step: Number of features to remove at each iteration
            estimator: Base estimator (None for default RandomForest)
            cv: Cross-validation folds (None for RFE, int for RFECV)
            exclude_columns: List of column names to exclude from selection
        """
        super().__init__(exclude_columns)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.estimator = estimator
        self.cv = cv

    def _create_estimator(self, task_type: str = "classification"):
        """Create default estimator if not provided"""
        if self.estimator is not None:
            return self.estimator

        # Default estimator
        if task_type == "regression":
            return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    def _create_selector(self, task_type: str = "classification"):
        """Create the underlying sklearn selector"""
        estimator = self._create_estimator(task_type)

        if self.cv is not None:
            # RFECV - automatic feature selection with cross-validation
            return RFECV(
                estimator=estimator,
                step=self.step,
                cv=self.cv,
                scoring="accuracy" if task_type != "regression" else "r2",
            )
        else:
            # RFE - manual feature count
            if self.n_features_to_select is None:
                raise ValueError(
                    "n_features_to_select gerekli RFE için. "
                    "RFECV kullanmak için cv parametresini ayarlayın."
                )
            return RFE(
                estimator=estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.step,
            )

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, task_type: str = "classification"
    ) -> "RFEFeatureSelector":
        """
        Fit selector on training data.

        Args:
            X: Training DataFrame
            y: Target vector (required for RFE)
            task_type: 'classification' or 'regression'

        Returns:
            self
        """
        if X.empty:
            raise ValueError("DataFrame boş, fit edilemez.")

        if y is None:
            raise ValueError("RFE için target vector (y) gerekli.")

        # Determine feature columns
        feature_columns = self._get_feature_columns(X)

        if not feature_columns:
            raise ValueError("Seçilecek feature bulunamadı.")

        # Create selector
        self.selector = self._create_selector(task_type)
        assert self.selector is not None, "Selector oluşturulamadı"

        # Fit selector
        X_features = X[feature_columns]
        self.selector.fit(X_features, y)

        # Get selected features
        support_mask = self.selector.get_support()
        self.selected_features = [
            feature_columns[i] for i in range(len(feature_columns)) if support_mask[i]
        ]

        return self
