"""
RobustScaler wrapper - Median and IQR based scaling (outlier resistant)
"""

from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from typing import List, Optional, Tuple

from ..base import BaseScaler


class RobustScaler(BaseScaler):
    """RobustScaler wrapper - Median and IQR based (outlier resistant)"""

    def __init__(
        self,
        exclude_columns: Optional[List[str]] = None,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
    ):
        """
        Args:
            exclude_columns: List of column names to exclude from scaling
            quantile_range: Quantile range for IQR calculation (default: (0.25, 0.75))
        """
        super().__init__(exclude_columns)
        self.quantile_range = quantile_range

    def _create_scaler(self):
        """Create the underlying sklearn RobustScaler"""
        return SklearnRobustScaler(quantile_range=self.quantile_range)
