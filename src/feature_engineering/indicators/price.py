"""
Fiyat bazlı feature'lar - Returns, ratios, lag features, rolling statistics
"""

import pandas as pd
from typing import List


class PriceFeatures:
    """Fiyat bazlı feature'lar sınıfı"""

    @staticmethod
    def create_returns(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Fiyat değişimlerini (returns) oluşturur.

        Args:
            df: DataFrame (close sütunu olmalı)
            periods: Returns periyotları

        Returns:
            DataFrame: Returns kolonları eklenmiş DataFrame
        """
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        for period in periods:
            df[f"returns_{period}"] = df["close"].pct_change(period)
        return df

    @staticmethod
    def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fiyat oranlarını oluşturur.

        Args:
            df: DataFrame (high, low, open, close sütunları olmalı)

        Returns:
            DataFrame: Ratio kolonları eklenmiş DataFrame
        """
        df = df.copy()
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        return df

    @staticmethod
    def create_lag_features(df: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
        """
        Lag feature'ları (geçmiş değerler) oluşturur.

        Args:
            df: DataFrame (close, volume sütunları olmalı)
            lag_periods: Lag periyotları

        Returns:
            DataFrame: Lag kolonları eklenmiş DataFrame
        """
        df = df.copy()
        for lag in lag_periods:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        return df

    @staticmethod
    def create_rolling_statistics(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Rolling istatistikleri oluşturur.

        Args:
            df: DataFrame (close sütunu ve returns sütunu olmalı)
            windows: Rolling window boyutları

        Returns:
            DataFrame: Rolling statistics kolonları eklenmiş DataFrame
        """
        df = df.copy()
        # Returns sütunu yoksa oluştur
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()

        for window in windows:
            df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()
            df[f"close_max_{window}"] = df["close"].rolling(window=window).max()
            df[f"close_min_{window}"] = df["close"].rolling(window=window).min()
        return df

    @staticmethod
    def create_all(
        df: pd.DataFrame,
        returns_periods: List[int],
        lag_periods: List[int],
        rolling_windows: List[int],
    ) -> pd.DataFrame:
        """
        Tüm fiyat bazlı feature'ları oluşturur.

        Args:
            df: DataFrame
            returns_periods: Returns periyotları
            lag_periods: Lag periyotları
            rolling_windows: Rolling window boyutları

        Returns:
            DataFrame: Tüm fiyat feature'ları eklenmiş DataFrame
        """
        df = PriceFeatures.create_returns(df, returns_periods)
        df = PriceFeatures.create_ratios(df)
        df = PriceFeatures.create_lag_features(df, lag_periods)
        df = PriceFeatures.create_rolling_statistics(df, rolling_windows)
        return df
