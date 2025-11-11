"""
Volatilite göstergeleri - Bollinger Bands, ATR, ADX
"""

import pandas as pd
import pandas_ta as ta
from ..utils import find_column_by_pattern


class VolatilityIndicators:
    """Volatilite göstergeleri sınıfı"""

    @staticmethod
    def create_bollinger_bands(
        df: pd.DataFrame, length: int = 20, std: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands göstergesini oluşturur.

        Args:
            df: DataFrame (close sütunu olmalı)
            length: BB periyodu
            std: Standart sapma çarpanı

        Returns:
            DataFrame: Bollinger Bands kolonları eklenmiş DataFrame
        """
        df = df.copy()
        bbands = ta.bbands(df["close"], length=length, std=std)  # type: ignore[arg-type]

        if bbands is not None and not bbands.empty:
            # Kolon adlarını dinamik olarak bul
            bb_upper_col = find_column_by_pattern(bbands, ["BBU", "upper"])
            bb_middle_col = find_column_by_pattern(bbands, ["BBM", "middle"])
            bb_lower_col = find_column_by_pattern(bbands, ["BBL", "lower"])

            if bb_upper_col and bb_middle_col and bb_lower_col:
                df["bb_upper"] = bbands[bb_upper_col]
                df["bb_middle"] = bbands[bb_middle_col]
                df["bb_lower"] = bbands[bb_lower_col]
                df["bb_width"] = (bbands[bb_upper_col] - bbands[bb_lower_col]) / bbands[
                    bb_middle_col
                ]
                df["bb_position"] = (df["close"] - bbands[bb_lower_col]) / (
                    bbands[bb_upper_col] - bbands[bb_lower_col]
                )

        return df

    @staticmethod
    def create_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average True Range (ATR) göstergesini oluşturur.

        Args:
            df: DataFrame (high, low, close sütunları olmalı)
            period: ATR periyodu

        Returns:
            DataFrame: ATR kolonu eklenmiş DataFrame
        """
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=period)
        return df

    @staticmethod
    def create_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX) göstergesini oluşturur.

        Args:
            df: DataFrame (high, low, close sütunları olmalı)
            period: ADX periyodu

        Returns:
            DataFrame: ADX kolonları eklenmiş DataFrame
        """
        df = df.copy()
        adx = ta.adx(df["high"], df["low"], df["close"], length=period)

        if adx is not None and not adx.empty:
            # Kolon adlarını dinamik olarak bul
            adx_col = find_column_by_pattern(adx, [f"ADX_{period}", "ADX_"])
            adx_pos_col = find_column_by_pattern(adx, ["DMP"])
            adx_neg_col = find_column_by_pattern(adx, ["DMN"])

            if adx_col:
                df["adx"] = adx[adx_col]
            if adx_pos_col:
                df["adx_pos"] = adx[adx_pos_col]
            if adx_neg_col:
                df["adx_neg"] = adx[adx_neg_col]

        return df

    @staticmethod
    def create_all(
        df: pd.DataFrame,
        bb_length: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        adx_period: int = 14,
    ) -> pd.DataFrame:
        """
        Tüm volatilite göstergelerini oluşturur.

        Args:
            df: DataFrame
            bb_length: Bollinger Bands periyodu
            bb_std: Bollinger Bands standart sapma çarpanı
            atr_period: ATR periyodu
            adx_period: ADX periyodu

        Returns:
            DataFrame: Tüm volatilite göstergeleri eklenmiş DataFrame
        """
        df = VolatilityIndicators.create_bollinger_bands(df, bb_length, bb_std)
        df = VolatilityIndicators.create_atr(df, atr_period)
        df = VolatilityIndicators.create_adx(df, adx_period)
        return df
