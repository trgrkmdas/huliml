"""
Trend göstergeleri - SMA, EMA
"""

import pandas as pd
import pandas_ta as ta
from typing import List


class TrendIndicators:
    """Trend göstergeleri sınıfı"""

    @staticmethod
    def create_sma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Simple Moving Average (SMA) göstergelerini oluşturur.

        Args:
            df: DataFrame (close sütunu olmalı)
            periods: SMA periyotları

        Returns:
            DataFrame: SMA kolonları eklenmiş DataFrame
        """
        df = df.copy()
        for period in periods:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)
        return df

    @staticmethod
    def create_ema(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Exponential Moving Average (EMA) göstergelerini oluşturur.

        Args:
            df: DataFrame (close sütunu olmalı)
            periods: EMA periyotları

        Returns:
            DataFrame: EMA kolonları eklenmiş DataFrame
        """
        df = df.copy()
        for period in periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
        return df

    @staticmethod
    def create_all(
        df: pd.DataFrame, sma_periods: List[int], ema_periods: List[int]
    ) -> pd.DataFrame:
        """
        Tüm trend göstergelerini oluşturur.

        Args:
            df: DataFrame
            sma_periods: SMA periyotları
            ema_periods: EMA periyotları

        Returns:
            DataFrame: Tüm trend göstergeleri eklenmiş DataFrame
        """
        df = TrendIndicators.create_sma(df, sma_periods)
        df = TrendIndicators.create_ema(df, ema_periods)
        return df
