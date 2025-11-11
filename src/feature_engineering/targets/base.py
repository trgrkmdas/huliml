"""
Target variable utility fonksiyonları.
"""

import pandas as pd


def calculate_future_return(df: pd.DataFrame, prediction_horizon: int) -> pd.Series:
    """
    Forward-looking return hesaplar.

    Args:
        df: DataFrame (close sütunu olmalı)
        prediction_horizon: Tahmin ufku (saat)

    Returns:
        Series: Future return serisi
    """
    future_price = df["close"].shift(-prediction_horizon)
    future_return = (future_price - df["close"]) / df["close"]
    return future_return
