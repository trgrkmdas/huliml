"""
End-to-end ML Pipeline Runner
Tüm adımları sırayla çalıştırır: Data Collection → Feature Engineering → Model Training
"""

import sys
from pathlib import Path

# Proje root'unu Python path'ine ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_collection import BitcoinDataCollector
from src.feature_engineering.base import FeatureEngineer
from src.model_training.trainer import ModelTrainer
from src.config import get_config
from src.logger import get_logger
import pandas as pd

logger = get_logger("MLProject.Pipeline")


def run_data_collection(config):
    """Veri toplama adımı"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 STEP 1: DATA COLLECTION")
    logger.info("=" * 60)

    try:
        collector = BitcoinDataCollector()

        # Veri çek: Config'den tarih aralığı ve interval alınır
        logger.info("🔄 Bitcoin verileri Binance'den çekiliyor...")
        collector.fetch_data(
            interval=config.data_collection.main_interval,
            start_date=config.data_collection.main_start_date,
            end_date=config.data_collection.main_end_date,
        )

        # Kaydet (config'den dizin yolu kullanılır)
        filepath = collector.save_data()
        logger.info(f"💾 Veri kaydedildi: {filepath}")

        if collector.data is not None:
            logger.info(
                f"📊 Veri özeti: {len(collector.data)} satır, {len(collector.data.columns)} sütun"
            )

        logger.info("✅ Data collection tamamlandı!")
        return True
    except Exception as e:
        logger.error(f"❌ Data collection hatası: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def run_feature_engineering(config):
    """Feature engineering adımı"""
    logger.info("\n" + "=" * 60)
    logger.info("🔧 STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    try:
        # Raw data yükle (son çekilen veriyi kullan)
        raw_data_dir = config.paths.raw_data_dir
        raw_files = sorted(
            raw_data_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True
        )

        if not raw_files:
            logger.error("❌ Raw data bulunamadı. Önce veri çekin.")
            return False

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

        logger.info("✅ Feature engineering tamamlandı!")
        logger.info(f"📊 Özet:")
        logger.info(f"   Raw data: {len(df_raw)} satır")
        logger.info(f"   Processed data: {len(df_processed)} satır")
        logger.info(f"   Feature sayısı: {len(df_processed.columns)}")
        logger.info(f"   Kayıt yolu: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Feature engineering hatası: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def run_model_training(config):
    """Model training adımı"""
    logger.info("\n" + "=" * 60)
    logger.info("🤖 STEP 3: MODEL TRAINING")
    logger.info("=" * 60)

    try:
        # Processed data yükle (son işlenen veriyi kullan)
        processed_data_dir = config.paths.processed_data_dir
        processed_files = sorted(
            processed_data_dir.glob("*.csv"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not processed_files:
            logger.error(
                "❌ Processed data bulunamadı. Önce feature engineering yapın."
            )
            return False

        processed_file = processed_files[0]
        logger.info(f"📂 Processed data yükleniyor: {processed_file}")

        df_processed = pd.read_csv(processed_file)
        df_processed["datetime"] = pd.to_datetime(df_processed["datetime"])

        logger.info(
            f"📊 Processed data: {len(df_processed)} satır, {len(df_processed.columns)} sütun"
        )

        # Model training
        trainer = ModelTrainer(config)
        trainer.train(df_processed)

        # Model kaydet
        if config.model.save_model:
            filepath = trainer.save_model()
            logger.info(f"\n💾 Model kaydedildi: {filepath}")

        logger.info("✅ Model training tamamlandı!")
        return True
    except Exception as e:
        logger.error(f"❌ Model training hatası: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def run_full_pipeline(enable_data_collection=False, skip_feature_engineering=False):
    """
    Tüm pipeline'ı çalıştır

    Args:
        enable_data_collection: True ise data collection adımını çalıştır (varsayılan: False)
        skip_feature_engineering: True ise feature engineering adımını atla
    """
    config = get_config()

    logger.info("=" * 60)
    logger.info("🚀 FULL ML PIPELINE")
    logger.info("=" * 60)
    logger.info(f"   Enable Data Collection: {enable_data_collection}")
    logger.info(f"   Skip Feature Engineering: {skip_feature_engineering}")
    logger.info("=" * 60)

    success = True

    # 1. Data Collection (varsayılan olarak kapalı - manuel veri çekme için)
    if enable_data_collection:
        if not run_data_collection(config):
            logger.error("❌ Pipeline data collection adımında durdu!")
            return False
    else:
        logger.info("\n⏭️  Data collection adımı atlandı (veri çekme manuel yapılmalı)")

    # 2. Feature Engineering
    if not skip_feature_engineering:
        if not run_feature_engineering(config):
            logger.error("❌ Pipeline feature engineering adımında durdu!")
            return False
    else:
        logger.info(
            "\n⏭️  Feature engineering adımı atlandı (skip_feature_engineering=True)"
        )

    # 3. Model Training
    if not run_model_training(config):
        logger.error("❌ Pipeline model training adımında durdu!")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✅ FULL PIPELINE COMPLETED!")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full ML Pipeline")
    parser.add_argument(
        "--enable-data-collection",
        action="store_true",
        help="Enable data collection step (default: disabled - manual data collection)",
    )
    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help="Skip feature engineering step",
    )

    args = parser.parse_args()

    success = run_full_pipeline(
        enable_data_collection=args.enable_data_collection,
        skip_feature_engineering=args.skip_feature_engineering,
    )

    sys.exit(0 if success else 1)
