"""
Model training modülü - Modüler yapı kullanarak.
Bu dosya geriye dönük uyumluluk için korunuyor.
Yeni kod model_training/ klasöründeki modülleri kullanmalı.
"""

# Geriye dönük uyumluluk için eski import'u koruyoruz
from .model_training.trainer import ModelTrainer
from .model_training import logger
from .config import get_config
import pandas as pd


def main():
    """Model training pipeline - Processed data'dan model oluşturur"""

    config = get_config()

    logger.info("=" * 60)
    logger.info("🚀 MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Processed data yükle (son işlenen veriyi kullan)
    processed_data_dir = config.paths.processed_data_dir
    processed_files = sorted(
        processed_data_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True
    )

    if not processed_files:
        logger.error("❌ Processed data bulunamadı. Önce feature engineering yapın.")
        return

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

    logger.info("\n" + "=" * 60)
    logger.info("✅ MODEL TRAINING TAMAMLANDI!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
