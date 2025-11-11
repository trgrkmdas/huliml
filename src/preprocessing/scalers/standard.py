"""
StandardScaler wrapper - Mean=0, Std=1 scaling
"""

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from typing import List, Optional

from ..base import BaseScaler


class StandardScaler(BaseScaler):
    """StandardScaler wrapper - Mean=0, Std=1"""

    def __init__(
        self,
        exclude_columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        """
        Args:
            exclude_columns: List of column names to exclude from scaling
            with_mean: If True, center the data before scaling
            with_std: If True, scale the data to unit variance
        """
        super().__init__(exclude_columns)
        self.with_mean = with_mean
        self.with_std = with_std

    def _create_scaler(self):
        """Create the underlying sklearn StandardScaler"""
        return SklearnStandardScaler(with_mean=self.with_mean, with_std=self.with_std)
