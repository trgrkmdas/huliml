"""
Config dosyasını nasıl kullanacağınıza dair örnekler.
"""

from .config import get_config, Config

# Örnek 1: Varsayılan config'i kullanma
config = get_config()
print(f"Varsayılan sembol: {config.data_collection.default_symbol}")
print(f"Varsayılan interval: {config.data_collection.default_interval}")

# Örnek 2: Config değerlerini değiştirme
config.data_collection.default_symbol = "ETH/USDT"
config.data_collection.default_interval = "1h"
config.data_collection.default_days_back = 365  # 1 yıl

# Örnek 3: Teknik gösterge parametrelerini özelleştirme
config.technical_indicators.sma_periods = [10, 20, 50, 100, 200]
config.technical_indicators.rsi_periods = [14, 21]

# Örnek 4: Model parametrelerini değiştirme
config.model.lightgbm_params["learning_rate"] = 0.01
config.model.lightgbm_params["num_leaves"] = 50

# Örnek 5: Yeni bir config instance'ı oluşturma
custom_config = Config()
custom_config.data_collection.default_symbol = "BNB/USDT"
custom_config.exchange.name = "binance"

# Örnek 6: Environment variables kullanımı
# Terminal'de şu komutları çalıştırın:
# export BINANCE_API_KEY="your_key"
# export BINANCE_API_SECRET="your_secret"
# export BINANCE_SANDBOX="true"
#
# Config otomatik olarak bu değerleri yükleyecektir
