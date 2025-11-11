"""
Multi-class target variable - Long=1, Short=-1, Hold=0
"""

import pandas as pd


class MulticlassTarget:
    """Multi-class target variable sınıfı"""

    @staticmethod
    def create(
        df: pd.DataFrame,
        future_return: pd.Series,
        threshold: float,
        drop_hold: bool = False,
    ) -> pd.DataFrame:
        """
        Multi-class target variable oluşturur.

        Args:
            df: DataFrame
            future_return: Future return serisi
            threshold: Eşik değeri
            drop_hold: Hold sınıfını çıkar mı?

        Returns:
            DataFrame: Multi-class target eklenmiş DataFrame
        """
        df = df.copy()
        df["target"] = 0  # Hold
        df.loc[future_return > threshold, "target"] = 1  # Long
        df.loc[future_return < -threshold, "target"] = -1  # Short
        df["future_return"] = future_return

        if drop_hold:
            df = df[df["target"] != 0].copy()

        return df
