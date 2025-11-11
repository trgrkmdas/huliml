"""
Feature Engineering ana sınıfı - Modüler yapı
"""

import pandas as pd
from typing import Optional
import os
from ..config import get_config
from ..logger import get_logger

# Modüller
from .indicators import (
    TrendIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    VolumeIndicators,
    PriceFeatures,
)
from .targets import BinaryTarget, MulticlassTarget, RegressionTarget
from .features import TimeFeatures

# Logger
logger = get_logger("MLProject.FeatureEngineering")


class FeatureEngineer:
    """Feature engineering sınıfı - Modüler yapı"""

    def __init__(self, config=None):
        """
        Args:
            config: Config objesi (None ise get_config() kullanılır)
        """
        self.config = config or get_config()
        self.data = None

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Teknik göstergeleri hesaplar (modüler yapı kullanarak).

        Args:
            df: DataFrame (OHLCV verisi)

        Returns:
            DataFrame: Teknik göstergelerle zenginleştirilmiş DataFrame
        """
        df = df.copy()

        if df.empty:
            raise ValueError("DataFrame boş.")

        # Datetime'ı index yap (pandas_ta için gerekli)
        df = df.set_index("datetime") if "datetime" in df.columns else df

        logger.info("📊 Teknik göstergeler hesaplanıyor...")

        ti_config = self.config.technical_indicators

        # Trend göstergeleri
        if ti_config.sma_periods is not None and ti_config.ema_periods is not None:
            df = TrendIndicators.create_all(
                df, ti_config.sma_periods, ti_config.ema_periods
            )

        # Momentum göstergeleri
        if ti_config.rsi_periods is not None:
            df = MomentumIndicators.create_all(
                df,
                ti_config.rsi_periods,
                ti_config.macd_fast,
                ti_config.macd_slow,
                ti_config.macd_signal,
                ti_config.stoch_k_period,
                ti_config.stoch_d_period,
                ti_config.stoch_smooth,
            )

        # Volatilite göstergeleri
        df = VolatilityIndicators.create_all(
            df,
            ti_config.bb_length,
            ti_config.bb_std,
            ti_config.atr_period,
            ti_config.adx_period,
        )

        # Volume göstergeleri
        df = VolumeIndicators.create_all(df, ti_config.volume_sma_period)

        # Fiyat feature'ları
        if (
            ti_config.returns_periods is not None
            and ti_config.lag_periods is not None
            and ti_config.rolling_windows is not None
        ):
            df = PriceFeatures.create_all(
                df,
                ti_config.returns_periods,
                ti_config.lag_periods,
                ti_config.rolling_windows,
            )

        # Index'i tekrar sütun yap
        df.reset_index(inplace=True)

        logger.info(
            f"✅ Teknik göstergeler hesaplandı. Toplam {len(df.columns)} sütun."
        )

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman bazlı feature'ları oluşturur.

        Args:
            df: DataFrame (datetime sütunu olmalı)

        Returns:
            DataFrame: Zaman feature'larıyla zenginleştirilmiş DataFrame
        """
        logger.info("🕐 Zaman bazlı feature'lar oluşturuluyor...")
        df = TimeFeatures.create(df)
        logger.info("✅ Zaman feature'ları oluşturuldu.")
        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target variable oluşturur (modüler yapı kullanarak).

        Args:
            df: DataFrame (datetime ve close sütunları olmalı)

        Returns:
            DataFrame: Target variable ile zenginleştirilmiş DataFrame
        """
        df = df.copy()

        if "close" not in df.columns:
            raise ValueError("DataFrame'de 'close' sütunu bulunamadı.")

        fe_config = self.config.feature_engineering

        logger.info("🎯 Target variable oluşturuluyor...")
        logger.info(f"   Prediction horizon: {fe_config.prediction_horizon} saat")
        logger.info(f"   Threshold: {fe_config.target_threshold*100:.2f}%")

        # Forward-looking return hesapla
        from .targets.base import calculate_future_return

        future_return = calculate_future_return(df, fe_config.prediction_horizon)

        # Target variable oluştur (strategy pattern)
        if fe_config.target_type == "binary":
            df = BinaryTarget.create(df, future_return, fe_config.target_threshold)
            df = df.iloc[: -fe_config.prediction_horizon].copy()

            logger.info("✅ Binary target oluşturuldu.")
            logger.info(
                f"   Long (1): {df['target'].sum()} satır ({df['target'].sum()/len(df)*100:.2f}%)"
            )
            logger.info(
                f"   Short (0): {(df['target']==0).sum()} satır ({(df['target']==0).sum()/len(df)*100:.2f}%)"
            )

        elif fe_config.target_type == "multi_class":
            df = MulticlassTarget.create(
                df,
                future_return,
                fe_config.target_threshold,
                fe_config.drop_hold_class,
            )
            df = df.iloc[: -fe_config.prediction_horizon].copy()

            logger.info("✅ Multi-class target oluşturuldu.")
            logger.info(f"   Long (1): {(df['target']==1).sum()} satır")
            logger.info(f"   Short (-1): {(df['target']==-1).sum()} satır")
            logger.info(f"   Hold (0): {(df['target']==0).sum()} satır")

        elif fe_config.target_type == "regression":
            df = RegressionTarget.create(df, future_return)
            df = df.iloc[: -fe_config.prediction_horizon].copy()
            logger.info("✅ Regression target oluşturuldu.")

        else:
            raise ValueError(f"Geçersiz target_type: {fe_config.target_type}")

        logger.info(f"📊 {len(df)} satır kaldı (target oluşturma sonrası).")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi temizler (missing values, duplicates).

        Args:
            df: DataFrame

        Returns:
            DataFrame: Temizlenmiş DataFrame
        """
        df = df.copy()

        fe_config = self.config.feature_engineering

        logger.info("🧹 Veri temizleniyor...")

        initial_rows = len(df)

        # Duplicate kontrolü
        duplicates = df.duplicated(subset=["datetime"]).sum()
        if duplicates > 0:
            logger.warning(f"⚠️  {duplicates} duplicate satır bulundu, çıkarılıyor...")
            df = df.drop_duplicates(subset=["datetime"])

        # Missing values
        if fe_config.drop_na:
            nan_before = df.isnull().sum().sum()
            df = df.dropna()
            nan_after = df.isnull().sum().sum()
            logger.info(f"   NaN değerler temizlendi: {nan_before} → {nan_after}")

        final_rows = len(df)
        logger.info(
            f"✅ Veri temizlendi: {initial_rows} → {final_rows} satır ({final_rows-initial_rows} satır çıkarıldı)."
        )

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm feature'ları oluşturur (ana metod - pipeline).

        Args:
            df: DataFrame (OHLCV verisi)

        Returns:
            DataFrame: Tüm feature'larla zenginleştirilmiş DataFrame
        """
        logger.info("🚀 Feature engineering başlatılıyor...")
        logger.info(f"📊 Başlangıç: {len(df)} satır, {len(df.columns)} sütun")

        # Teknik göstergeler
        df = self.create_technical_indicators(df)

        # Zaman feature'ları (opsiyonel)
        fe_config = self.config.feature_engineering
        if fe_config.include_time_features:
            df = self.create_time_features(df)

        # Target variable
        df = self.create_target_variable(df)

        # Veri temizleme
        df = self.clean_data(df)

        # Preprocessing (scaling) - Opsiyonel, genelde train/test split'ten sonra kullanılmalı
        fe_config = self.config.feature_engineering
        if fe_config.enable_scaling_in_pipeline:
            df = self.scale_features(df)

        self.data = df  # type: ignore[assignment]

        logger.info("✅ Feature engineering tamamlandı!")
        logger.info(f"📊 Sonuç: {len(df)} satır, {len(df.columns)} sütun")

        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature scaling (opsiyonel, genelde train/test split'ten sonra kullanılmalı).

        ⚠️  DİKKAT: Bu metod tüm veri üzerinde fit eder, data leakage riski taşır.
        Production için train/test split'ten sonra Preprocessor kullanılmalı.

        Args:
            df: DataFrame

        Returns:
            DataFrame: Scaled DataFrame
        """
        from ..preprocessing import Preprocessor

        logger.info("🔧 Feature scaling yapılıyor (pipeline içinde)...")
        logger.warning(
            "⚠️  DİKKAT: Bu yaklaşım data leakage riski taşır. "
            "Production için train/test split'ten sonra scaling yapılmalı."
        )

        preprocessor = Preprocessor(config=self.config)
        df_scaled = preprocessor.fit_transform(df)

        return df_scaled

    def save_processed_data(
        self, df: Optional[pd.DataFrame] = None, filepath: Optional[str] = None
    ) -> str:
        """
        Processed data'yı CSV olarak kaydeder.

        Args:
            df: DataFrame (None ise self.data kullanılır)
            filepath: Kayıt yolu (None ise otomatik oluşturulur)

        Returns:
            str: Kaydedilen dosya yolu
        """
        if df is None:
            df = self.data

        if df is None or df.empty:
            raise ValueError("Kaydedilecek veri bulunamadı.")

        if filepath is None:
            # Config'den dizin yolunu al
            processed_data_dir = self.config.paths.processed_data_dir
            os.makedirs(processed_data_dir, exist_ok=True)

            # Dosya adı oluştur
            start_date = df["datetime"].min().strftime("%Y%m%d")
            end_date = df["datetime"].max().strftime("%Y%m%d")
            fe_config = self.config.feature_engineering
            filepath = (
                processed_data_dir
                / f"processed_{start_date}_{end_date}_h{fe_config.prediction_horizon}_t{fe_config.target_threshold}.csv"
            )
            filepath = str(filepath)

        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False)
        logger.info(f"💾 Processed data kaydedildi: {filepath}")
        logger.info(f"📁 Dosya boyutu: {os.path.getsize(filepath) / 1024:.2f} KB")

        return filepath

    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """
        Processed data'yı CSV'den yükler.

        Args:
            filepath: Dosya yolu

        Returns:
            DataFrame: Yüklenen veri
        """
        self.data = pd.read_csv(filepath)  # type: ignore[assignment]
        if self.data is not None and "datetime" in self.data.columns:
            self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        logger.info(f"📂 Processed data yüklendi: {filepath}")
        if self.data is not None:
            logger.info(f"📊 {len(self.data)} satır, {len(self.data.columns)} sütun")
            return self.data  # type: ignore[no-any-return]
        else:
            raise ValueError("Veri yüklenemedi")
