"""
QuantileTransformer wrapper - Uniform or normal distribution transformation
"""

from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from typing import List, Optional

from ..base import BaseScaler


class QuantileTransformer(BaseScaler):
    """QuantileTransformer wrapper - Uniform or normal distribution"""

    def __init__(
        self,
        exclude_columns: Optional[List[str]] = None,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
    ):
        """
        Args:
            exclude_columns: List of column names to exclude from scaling
            n_quantiles: Number of quantiles to be computed
            output_distribution: Output distribution ('uniform' or 'normal')
        """
        super().__init__(exclude_columns)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def _create_scaler(self):
        """Create the underlying sklearn QuantileTransformer"""
        return SklearnQuantileTransformer(
            n_quantiles=self.n_quantiles, output_distribution=self.output_distribution
        )
