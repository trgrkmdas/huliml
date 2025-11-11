"""
Regression target variable - Direkt return tahmini
"""

import pandas as pd


class RegressionTarget:
    """Regression target variable sınıfı"""

    @staticmethod
    def create(df: pd.DataFrame, future_return: pd.Series) -> pd.DataFrame:
        """
        Regression target variable oluşturur.

        Args:
            df: DataFrame
            future_return: Future return serisi

        Returns:
            DataFrame: Regression target eklenmiş DataFrame
        """
        df = df.copy()
        df["target"] = future_return
        return df
