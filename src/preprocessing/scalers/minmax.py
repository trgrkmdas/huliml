"""
MinMaxScaler wrapper - Scale to 0-1 range
"""

from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from typing import List, Optional, Tuple

from ..base import BaseScaler


class MinMaxScaler(BaseScaler):
    """MinMaxScaler wrapper - Scale to 0-1 range"""

    def __init__(
        self,
        exclude_columns: Optional[List[str]] = None,
        feature_range: Tuple[float, float] = (0, 1),
    ):
        """
        Args:
            exclude_columns: List of column names to exclude from scaling
            feature_range: Desired range of transformed data (default: (0, 1))
        """
        super().__init__(exclude_columns)
        self.feature_range = feature_range

    def _create_scaler(self):
        """Create the underlying sklearn MinMaxScaler"""
        return SklearnMinMaxScaler(feature_range=self.feature_range)
