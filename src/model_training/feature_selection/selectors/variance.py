"""
Variance threshold feature selection - Remove low variance features
"""

from typing import List, Optional
from sklearn.feature_selection import VarianceThreshold

from ..base import BaseFeatureSelector


class VarianceThresholdSelector(BaseFeatureSelector):
    """Variance threshold feature selection - Remove low variance features"""

    def __init__(
        self,
        threshold: float = 0.0,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            threshold: Features with variance below this threshold will be removed
            exclude_columns: List of column names to exclude from selection
        """
        super().__init__(exclude_columns)
        self.threshold = threshold

    def _create_selector(self):
        """Create the underlying sklearn selector"""
        return VarianceThreshold(threshold=self.threshold)
