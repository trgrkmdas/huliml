"""
Volume göstergeleri
"""

import pandas as pd
import pandas_ta as ta


class VolumeIndicators:
    """Volume göstergeleri sınıfı"""

    @staticmethod
    def create_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Volume SMA göstergesini oluşturur.

        Args:
            df: DataFrame (volume sütunu olmalı)
            period: SMA periyodu

        Returns:
            DataFrame: Volume SMA kolonları eklenmiş DataFrame
        """
        df = df.copy()
        df["volume_sma"] = ta.sma(df["volume"], length=period)
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        return df

    @staticmethod
    def create_all(df: pd.DataFrame, volume_sma_period: int = 20) -> pd.DataFrame:
        """
        Tüm volume göstergelerini oluşturur.

        Args:
            df: DataFrame
            volume_sma_period: Volume SMA periyodu

        Returns:
            DataFrame: Tüm volume göstergeleri eklenmiş DataFrame
        """
        return VolumeIndicators.create_volume_sma(df, volume_sma_period)
