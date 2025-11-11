"""
Bitcoin fiyat verilerini Binance API'den toplama modülü.
"""

import pandas as pd
import ccxt
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Optional
import os
import time
from .config import get_config
from .logger import get_logger

# Logger'ı modül seviyesinde oluştur
logger = get_logger("MLProject.DataCollection")


class BitcoinDataCollector:
    """Bitcoin fiyat verilerini Binance'den toplayan sınıf."""

    def __init__(self, symbol: Optional[str] = None):
        """
        Args:
            symbol: Bitcoin sembolü (None ise config'den alınır)
        """
        self.config = get_config()
        self.symbol = symbol or self.config.data_collection.default_symbol

        # Exchange oluştur
        exchange_class = getattr(ccxt, self.config.exchange.name)
        exchange_params = {
            "enableRateLimit": self.config.exchange.enable_rate_limit,
            "options": {"defaultType": self.config.exchange.default_type},
        }

        # API keys varsa ekle
        if self.config.exchange.api_key:
            exchange_params["apiKey"] = self.config.exchange.api_key
        if self.config.exchange.api_secret:
            exchange_params["secret"] = self.config.exchange.api_secret
        if self.config.exchange.sandbox:
            exchange_params["sandbox"] = True

        self.exchange = exchange_class(exchange_params)
        self.data = None

    def _convert_interval(self, interval: str) -> str:
        """
        Standart interval formatını Binance formatına çevirir.

        Args:
            interval: Standart format (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)

        Returns:
            str: Binance formatı
        """
        interval_map = self.config.data_collection.supported_intervals
        default_interval = self.config.data_collection.default_interval
        return interval_map.get(interval, default_interval)

    def fetch_data(
        self,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Bitcoin fiyat verilerini Binance API'den çeker.

        Args:
            interval: Veri aralığı (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w) (None ise config'den alınır)
            start_date: Başlangıç tarihi (YYYY-MM-DD veya timestamp)
            end_date: Bitiş tarihi (YYYY-MM-DD veya timestamp)
            limit: Her istekte çekilecek maksimum veri sayısı (None ise config'den alınır)

        Returns:
            DataFrame: OHLCV verileri
        """
        # Config'den varsayılan değerleri al
        interval = interval or self.config.data_collection.default_interval
        limit = limit or self.config.data_collection.default_limit

        binance_interval = self._convert_interval(interval)

        # Tarihleri timestamp'e çevir
        if start_date:
            if isinstance(start_date, str):
                start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
            else:
                start_timestamp = int(start_date * 1000)
        else:
            # Config'den varsayılan gün sayısını al
            days_back = self.config.data_collection.default_days_back
            start_timestamp = int(
                (datetime.now() - timedelta(days=days_back)).timestamp() * 1000
            )

        if end_date:
            if isinstance(end_date, str):
                end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
            else:
                end_timestamp = int(end_date * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)

        logger.info("🔄 Binance'den veri çekiliyor...")
        logger.info(f"📊 Sembol: {self.symbol}")
        logger.info(f"⏱️  Interval: {interval} ({binance_interval})")
        logger.info(
            f"📅 Tarih aralığı: {datetime.fromtimestamp(start_timestamp/1000)} - {datetime.fromtimestamp(end_timestamp/1000)}"
        )

        all_ohlcv = []
        current_timestamp = start_timestamp

        # Config'den rate limiting değerlerini al
        request_delay = self.config.data_collection.request_delay
        if request_delay is None:
            request_delay = 0.1  # Varsayılan değer

        while current_timestamp < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, binance_interval, since=current_timestamp, limit=limit
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Son çekilen verinin timestamp'ini al
                last_timestamp = ohlcv[-1][0]

                # Eğer aynı timestamp'te kalıyorsak, bir sonraki interval'e geç
                if last_timestamp == current_timestamp:
                    # Interval'e göre timestamp artır
                    interval_ms = self._get_interval_ms(binance_interval)
                    current_timestamp = last_timestamp + interval_ms
                else:
                    current_timestamp = last_timestamp + 1  # +1 ms

                # Rate limiting
                time.sleep(request_delay)

                # İlerleme göster
                threshold = self.config.data_collection.progress_print_threshold
                if len(all_ohlcv) % threshold == 0:
                    logger.debug(f"   📥 {len(all_ohlcv)} mum çekildi...")

            except ccxt.NetworkError as e:
                # Ağ bağlantı hatası
                logger.warning(f"⚠️  Ağ hatası: {e}")
                logger.info("   Bekleniyor...")
                time.sleep(self.config.data_collection.error_retry_delay)
                continue
            except ccxt.ExchangeError as e:
                # Exchange API hatası (rate limit, invalid request, etc.)
                logger.warning(f"⚠️  Exchange hatası: {e}")
                logger.info("   Bekleniyor...")
                time.sleep(self.config.data_collection.error_retry_delay)
                continue
            except ccxt.RequestTimeout as e:
                # İstek zaman aşımı
                logger.warning(f"⚠️  İstek zaman aşımı: {e}")
                logger.info("   Bekleniyor...")
                time.sleep(self.config.data_collection.error_retry_delay)
                continue
            except ConnectionError as e:
                # Python ConnectionError
                logger.warning(f"⚠️  Bağlantı hatası: {e}")
                logger.info("   Bekleniyor...")
                time.sleep(self.config.data_collection.error_retry_delay)
                continue
            except TimeoutError as e:
                # Python TimeoutError
                logger.warning(f"⚠️  Zaman aşımı: {e}")
                logger.info("   Bekleniyor...")
                time.sleep(self.config.data_collection.error_retry_delay)
                continue
            except ValueError as e:
                # Geçersiz parametreler
                logger.error(f"❌ Geçersiz parametre: {e}")
                raise  # ValueError'ı yeniden fırlat çünkü bu düzeltilemez bir hata
            except Exception as e:
                # Diğer beklenmeyen hatalar için genel exception (fallback)
                logger.warning(f"⚠️  Beklenmeyen hata: {type(e).__name__}: {e}")
                logger.info("   Bekleniyor...")
                time.sleep(self.config.data_collection.error_retry_delay)
                continue

        if not all_ohlcv:
            raise ValueError("Veri çekilemedi. Lütfen parametreleri kontrol edin.")

        # DataFrame oluştur
        df = pd.DataFrame(
            all_ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )

        # Timestamp'i datetime'a çevir
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

        # Duplicate'leri temizle
        df = df.drop_duplicates(subset=["datetime"])

        # Tarihe göre sırala
        df = df.sort_values("datetime").reset_index(drop=True)

        # End date'e kadar filtrele
        df = df[df["datetime"] <= pd.Timestamp(end_timestamp, unit="ms")]

        self.data = df  # type: ignore[assignment]

        logger.info(f"✅ {len(df)} satır veri çekildi.")
        logger.info(
            f"📅 Tarih aralığı: {df['datetime'].min()} - {df['datetime'].max()}"
        )

        return df

    def _get_interval_ms(self, interval: str) -> int:
        """Interval'i milisaniyeye çevirir."""
        interval_ms_map = self.config.data_collection.interval_ms_map
        default_ms = self.config.data_collection.default_interval_ms
        return interval_ms_map.get(interval, default_ms)

    def calculate_technical_indicators(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Teknik göstergeleri hesaplar.

        Args:
            df: DataFrame (None ise self.data kullanılır)

        Returns:
            DataFrame: Teknik göstergelerle zenginleştirilmiş DataFrame
        """
        if df is None:
            if self.data is None:
                raise ValueError(
                    "Veri bulunamadı. Önce fetch_data() metodunu çalıştırın."
                )
            df = self.data.copy()

        if df is None or df.empty:
            raise ValueError("Veri bulunamadı. Önce fetch_data() metodunu çalıştırın.")

        # Datetime'ı index yap (pandas_ta için gerekli)
        df = df.set_index("datetime") if "datetime" in df.columns else df

        logger.info("📊 Teknik göstergeler hesaplanıyor...")

        ti_config = self.config.technical_indicators

        # Trend göstergeleri - SMA
        for period in ti_config.sma_periods:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)

        # Trend göstergeleri - EMA
        for period in ti_config.ema_periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # Momentum göstergeleri - RSI
        if ti_config.rsi_periods is not None:
            for idx, period in enumerate(ti_config.rsi_periods):
                if idx == 0:  # İlk değer varsayılan 'rsi' adıyla
                    df["rsi"] = ta.rsi(df["close"], length=period)
                else:
                    df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

        # MACD
        macd = ta.macd(
            df["close"],
            fast=ti_config.macd_fast,
            slow=ti_config.macd_slow,
            signal=ti_config.macd_signal,
        )
        if macd is not None and not macd.empty:
            df["macd"] = macd[
                f"MACD_{ti_config.macd_fast}_{ti_config.macd_slow}_{ti_config.macd_signal}"
            ]
            df["macd_signal"] = macd[
                f"MACDs_{ti_config.macd_fast}_{ti_config.macd_slow}_{ti_config.macd_signal}"
            ]
            df["macd_hist"] = macd[
                f"MACDh_{ti_config.macd_fast}_{ti_config.macd_slow}_{ti_config.macd_signal}"
            ]

        # Bollinger Bands
        # Note: pandas_ta bbands std parametresi dict bekliyor ama float da kabul ediyor
        bbands = ta.bbands(
            df["close"], length=ti_config.bb_length, std=ti_config.bb_std  # type: ignore[arg-type]
        )
        if bbands is not None and not bbands.empty:
            df["bb_upper"] = bbands[f"BBU_{ti_config.bb_length}_{ti_config.bb_std}"]
            df["bb_middle"] = bbands[f"BBM_{ti_config.bb_length}_{ti_config.bb_std}"]
            df["bb_lower"] = bbands[f"BBL_{ti_config.bb_length}_{ti_config.bb_std}"]
            df["bb_width"] = (
                bbands[f"BBU_{ti_config.bb_length}_{ti_config.bb_std}"]
                - bbands[f"BBL_{ti_config.bb_length}_{ti_config.bb_std}"]
            ) / bbands[f"BBM_{ti_config.bb_length}_{ti_config.bb_std}"]
            df["bb_position"] = (
                df["close"] - bbands[f"BBL_{ti_config.bb_length}_{ti_config.bb_std}"]
            ) / (
                bbands[f"BBU_{ti_config.bb_length}_{ti_config.bb_std}"]
                - bbands[f"BBL_{ti_config.bb_length}_{ti_config.bb_std}"]
            )

        # Stochastic Oscillator
        stoch = ta.stoch(
            df["high"],
            df["low"],
            df["close"],
            k=ti_config.stoch_k_period,
            d=ti_config.stoch_d_period,
            smooth_k=ti_config.stoch_smooth,
        )
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch[
                f"STOCHk_{ti_config.stoch_k_period}_{ti_config.stoch_smooth}_{ti_config.stoch_d_period}"
            ]
            df["stoch_d"] = stoch[
                f"STOCHd_{ti_config.stoch_k_period}_{ti_config.stoch_smooth}_{ti_config.stoch_d_period}"
            ]

        # ATR (Average True Range) - Volatilite göstergesi
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"], length=ti_config.atr_period
        )

        # ADX (Average Directional Index) - Trend gücü
        adx = ta.adx(df["high"], df["low"], df["close"], length=ti_config.adx_period)
        if adx is not None and not adx.empty:
            df["adx"] = adx[f"ADX_{ti_config.adx_period}"]
            df["adx_pos"] = adx[f"DMP_{ti_config.adx_period}"]
            df["adx_neg"] = adx[f"DMN_{ti_config.adx_period}"]

        # Volume göstergeleri
        df["volume_sma"] = ta.sma(df["volume"], length=ti_config.volume_sma_period)
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Fiyat değişimleri
        df["returns"] = df["close"].pct_change()
        for period in ti_config.returns_periods:
            df[f"returns_{period}"] = df["close"].pct_change(period)

        # Yüksek/Düşük oranları
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # Fiyat pozisyonu (günlük range içindeki konumu)
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # Lag features (geçmiş değerler)
        for lag in ti_config.lag_periods:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        # Rolling istatistikler
        for window in ti_config.rolling_windows:
            df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()
            df[f"close_max_{window}"] = df["close"].rolling(window=window).max()
            df[f"close_min_{window}"] = df["close"].rolling(window=window).min()

        # Index'i tekrar sütun yap
        df.reset_index(inplace=True)

        # NaN değerleri temizle
        df = df.dropna()

        logger.info(f"✅ {len(df.columns)} sütunlu veri seti hazırlandı.")
        logger.info(f"📊 {len(df)} satır veri kaldı (NaN temizleme sonrası).")

        self.data = df  # type: ignore[assignment]
        return df

    def save_data(self, filepath: Optional[str] = None) -> str:
        """
        Veriyi CSV olarak data/raw klasörüne kaydeder.

        Args:
            filepath: Kayıt yolu (None ise otomatik oluşturulur)

        Returns:
            str: Kaydedilen dosya yolu
        """
        if self.data is None or self.data.empty:
            raise ValueError("Kaydedilecek veri bulunamadı.")

        if filepath is None:
            # Config'den dizin yolunu al
            raw_data_dir = self.config.paths.raw_data_dir
            os.makedirs(raw_data_dir, exist_ok=True)
            symbol_clean = self.symbol.replace("/", "_")
            start_date = self.data["datetime"].min().strftime("%Y%m%d")
            end_date = self.data["datetime"].max().strftime("%Y%m%d")
            filepath = raw_data_dir / f"{symbol_clean}_{start_date}_{end_date}.csv"
            filepath = str(filepath)

        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.data.to_csv(filepath, index=False)
        logger.info(f"💾 Veri kaydedildi: {filepath}")
        logger.info(f"📁 Dosya boyutu: {os.path.getsize(filepath) / 1024:.2f} KB")

        return filepath

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        CSV dosyasından veri yükler.

        Args:
            filepath: Dosya yolu

        Returns:
            DataFrame: Yüklenen veri
        """
        self.data = pd.read_csv(filepath)  # type: ignore[assignment]
        if self.data is not None and "datetime" in self.data.columns:
            self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        logger.info(f"📂 Veri yüklendi: {filepath}")
        if self.data is not None:
            logger.info(f"📊 {len(self.data)} satır, {len(self.data.columns)} sütun")
            return self.data
        else:
            raise ValueError("Veri yüklenemedi")


def main():
    """Bitcoin verilerini çeker - tarih aralığı ve interval config'den alınır"""
    config = get_config()

    # Config'den varsayılan sembol kullanılır
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

    logger.info("\n✅ Veri toplama işlemi tamamlandı!")
    if collector.data is not None:
        logger.info("\n📊 Veri özeti:")
        logger.debug(f"\n{collector.data.describe()}")
        logger.info("\n📈 İlk 5 satır:")
        logger.debug(f"\n{collector.data.head()}")
        logger.info("\n📈 Son 5 satır:")
        logger.debug(f"\n{collector.data.tail()}")


if __name__ == "__main__":
    main()
