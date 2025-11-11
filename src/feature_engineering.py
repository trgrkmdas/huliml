"""
Feature engineering modülü - Modüler yapı kullanarak.
Bu dosya geriye dönük uyumluluk için korunuyor.
Yeni kod feature_engineering/ klasöründeki modülleri kullanmalı.
"""

# Geriye dönük uyumluluk için eski import'u koruyoruz
from .feature_engineering.base import FeatureEngineer
from .feature_engineering.base import logger
from .config import get_config
import os
import pandas as pd


def main():
    """Feature engineering pipeline - Raw data'dan processed data oluşturur"""

    config = get_config()

    logger.info("=" * 60)
    logger.info("🚀 FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Raw data yükle (son çekilen veriyi kullan)
    raw_data_dir = config.paths.raw_data_dir
    raw_files = sorted(raw_data_dir.glob("*.csv"), key=os.path.getmtime, reverse=True)

    if not raw_files:
        logger.error("❌ Raw data bulunamadı. Önce veri çekin.")
        return

    raw_file = raw_files[0]
    logger.info(f"📂 Raw data yükleniyor: {raw_file}")

    df_raw = pd.read_csv(raw_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])

    logger.info(f"📊 Raw data: {len(df_raw)} satır, {len(df_raw.columns)} sütun")

    # Feature engineering
    fe = FeatureEngineer(config)
    df_processed = fe.create_features(df_raw)

    # Kaydet
    filepath = fe.save_processed_data(df_processed)

    logger.info("\n" + "=" * 60)
    logger.info("✅ FEATURE ENGINEERING TAMAMLANDI!")
    logger.info("=" * 60)
    logger.info("\n📊 Özet:")
    logger.info(f"   Raw data: {len(df_raw)} satır")
    logger.info(f"   Processed data: {len(df_processed)} satır")
    logger.info(f"   Feature sayısı: {len(df_processed.columns)}")
    logger.info(f"   Kayıt yolu: {filepath}")


if __name__ == "__main__":
    main()
