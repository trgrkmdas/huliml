"""
Zaman bazlı feature'lar.
"""

import pandas as pd


class TimeFeatures:
    """Zaman bazlı feature'lar sınıfı"""

    @staticmethod
    def create(df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman bazlı feature'ları oluşturur.

        Args:
            df: DataFrame (datetime sütunu olmalı)

        Returns:
            DataFrame: Zaman feature'ları eklenmiş DataFrame
        """
        df = df.copy()

        if "datetime" not in df.columns:
            raise ValueError("DataFrame'de 'datetime' sütunu bulunamadı.")

        # Datetime'ı parse et
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        # Zaman feature'ları
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["month"] = df["datetime"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # Saturday=5, Sunday=6

        return df
