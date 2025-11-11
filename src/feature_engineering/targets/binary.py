"""
Binary target variable - Long=1, Short=0
"""

import pandas as pd


class BinaryTarget:
    """Binary target variable sınıfı"""

    @staticmethod
    def create(
        df: pd.DataFrame, future_return: pd.Series, threshold: float
    ) -> pd.DataFrame:
        """
        Binary target variable oluşturur.

        Args:
            df: DataFrame
            future_return: Future return serisi
            threshold: Eşik değeri (örn: 0.01 = %1)

        Returns:
            DataFrame: Binary target eklenmiş DataFrame
        """
        df = df.copy()
        df["target"] = (future_return > threshold).astype(int)
        df["future_return"] = future_return
        return df
