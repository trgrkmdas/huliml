"""
Momentum göstergeleri - RSI, MACD, Stochastic Oscillator
"""

import pandas as pd
import pandas_ta as ta
from typing import List
from ..utils import find_column_by_pattern


class MomentumIndicators:
    """Momentum göstergeleri sınıfı"""

    @staticmethod
    def create_rsi(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Relative Strength Index (RSI) göstergelerini oluşturur.

        Args:
            df: DataFrame (close sütunu olmalı)
            periods: RSI periyotları

        Returns:
            DataFrame: RSI kolonları eklenmiş DataFrame
        """
        df = df.copy()
        if periods:
            # İlk periyot varsayılan 'rsi' adıyla
            df["rsi"] = ta.rsi(df["close"], length=periods[0])
            # Diğer periyotlar
            for period in periods[1:]:
                df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
        return df

    @staticmethod
    def create_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence) göstergesini oluşturur.

        Args:
            df: DataFrame (close sütunu olmalı)
            fast: Fast EMA periyodu
            slow: Slow EMA periyodu
            signal: Signal line periyodu

        Returns:
            DataFrame: MACD kolonları eklenmiş DataFrame
        """
        df = df.copy()
        macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

        if macd is not None and not macd.empty:
            # Kolon adlarını dinamik olarak bul
            macd_col = find_column_by_pattern(
                macd, ["MACD_"] + [f"MACD_{fast}_{slow}_{signal}"]
            )
            macd_signal_col = find_column_by_pattern(macd, ["MACDs"])
            macd_hist_col = find_column_by_pattern(macd, ["MACDh"])

            if macd_col:
                df["macd"] = macd[macd_col]
            if macd_signal_col:
                df["macd_signal"] = macd[macd_signal_col]
            if macd_hist_col:
                df["macd_hist"] = macd[macd_hist_col]

        return df

    @staticmethod
    def create_stochastic(
        df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator göstergesini oluşturur.

        Args:
            df: DataFrame (high, low, close sütunları olmalı)
            k_period: %K periyodu
            d_period: %D periyodu
            smooth_k: %K smoothing periyodu

        Returns:
            DataFrame: Stochastic kolonları eklenmiş DataFrame
        """
        df = df.copy()
        stoch = ta.stoch(
            df["high"],
            df["low"],
            df["close"],
            k=k_period,
            d=d_period,
            smooth_k=smooth_k,
        )

        if stoch is not None and not stoch.empty:
            # Kolon adlarını dinamik olarak bul
            stoch_k_col = find_column_by_pattern(stoch, ["STOCHk", "STOCH"])
            stoch_d_col = find_column_by_pattern(stoch, ["STOCHd"])

            if stoch_k_col:
                df["stoch_k"] = stoch[stoch_k_col]
            if stoch_d_col:
                df["stoch_d"] = stoch[stoch_d_col]

        return df

    @staticmethod
    def create_all(
        df: pd.DataFrame,
        rsi_periods: List[int],
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_k_period: int = 14,
        stoch_d_period: int = 3,
        stoch_smooth: int = 3,
    ) -> pd.DataFrame:
        """
        Tüm momentum göstergelerini oluşturur.

        Args:
            df: DataFrame
            rsi_periods: RSI periyotları
            macd_fast: MACD fast periyodu
            macd_slow: MACD slow periyodu
            macd_signal: MACD signal periyodu

        Returns:
            DataFrame: Tüm momentum göstergeleri eklenmiş DataFrame
        """
        df = MomentumIndicators.create_rsi(df, rsi_periods)
        df = MomentumIndicators.create_macd(df, macd_fast, macd_slow, macd_signal)
        df = MomentumIndicators.create_stochastic(
            df, stoch_k_period, stoch_d_period, stoch_smooth
        )
        return df
