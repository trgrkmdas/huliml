# Feature Scaling/Normalization Analizi ve Planlama

## 📊 Mevcut Durum Analizi

### Proje Özeti
- **Proje Tipi**: Bitcoin/USDT kripto para trading sinyali tahmin (Binary Classification: Long/Short)
- **Model**: LightGBM (Gradient Boosting)
- **Feature Sayısı**: ~64 feature (target ve datetime hariç)

### Mevcut Feature Kategorileri

#### 1. **OHLCV Verileri** (Scale edilmeli)
- `open`, `high`, `low`, `close`: ~46,000 - 70,000 arası
- `volume`: Binlerce değer

#### 2. **Trend Göstergeleri** (Scale edilmeli)
- `sma_7`, `sma_14`, `sma_21`, `sma_50`, `sma_200`: Fiyat ölçeğinde (~46k-70k)
- `ema_12`, `ema_26`: Fiyat ölçeğinde

#### 3. **Momentum Göstergeleri** (Kısmen scale edilmeli)
- `rsi`, `rsi_7`, `rsi_21`: 0-100 arası (zaten normalize)
- `macd`, `macd_signal`, `macd_hist`: Yüzlerce değer (scale edilmeli)
- `stoch_k`, `stoch_d`: 0-100 arası (zaten normalize)

#### 4. **Volatilite Göstergeleri** (Scale edilmeli)
- `bb_upper`, `bb_middle`, `bb_lower`: Fiyat ölçeğinde
- `bb_width`: 0.05-0.07 gibi küçük değerler (scale edilmeli)
- `bb_position`: 0-1 arası (zaten normalize)
- `atr`: Yüzlerce değer
- `adx`, `adx_pos`, `adx_neg`: 0-100 arası (zaten normalize)

#### 5. **Volume Göstergeleri** (Scale edilmeli)
- `volume_sma`: Binlerce değer
- `volume_ratio`: Oran (scale edilebilir ama gerekli değil)

#### 6. **Fiyat Feature'ları** (Kısmen scale edilmeli)
- `returns`, `returns_5`, `returns_10`, `returns_20`: -0.1 ile 0.1 arası (çok küçük)
- `high_low_ratio`, `close_open_ratio`: Oranlar (~1.0 civarı)
- `price_position`: 0-1 arası (zaten normalize)
- `close_lag_*`: Fiyat ölçeğinde
- `volume_lag_*`: Volume ölçeğinde
- `volatility_*`: Çok küçük değerler (0.002-0.008 arası)
- `close_max_*`, `close_min_*`: Fiyat ölçeğinde

#### 7. **Zaman Feature'ları** (Scale edilmemeli)
- `hour`: 0-23 (categorical)
- `day_of_week`: 0-6 (categorical)
- `month`: 1-12 (categorical)
- `is_weekend`: 0-1 (binary)

#### 8. **Target Variables** (Scale edilmemeli)
- `target`: Binary (0/1)
- `future_return`: Regression target

### Ölçek Farklılıkları

| Feature Kategorisi | Ölçek Aralığı | Örnek Değerler |
|-------------------|---------------|----------------|
| Fiyatlar | 46,000 - 70,000 | close, open, high, low |
| Volume | 500 - 5,000 | volume, volume_sma |
| Returns | -0.1 - 0.1 | returns, returns_5, returns_10 |
| Volatility | 0.002 - 0.008 | volatility_7, volatility_14 |
| RSI/ADX | 0 - 100 | rsi, adx, stoch_k |
| Ratios | 0.9 - 1.1 | high_low_ratio, close_open_ratio |
| MACD | -500 - 500 | macd, macd_signal |
| BB Width | 0.05 - 0.07 | bb_width |

## 🎯 Feature Scaling Gerekli mi?

### LightGBM için Durum
- **Tree-based modeller** (LightGBM, XGBoost, Random Forest) genelde scaling'e ihtiyaç duymaz
- Ancak bazı durumlarda faydalı olabilir:
  - Feature importance karşılaştırmaları
  - Hyperparameter tuning stabilitesi
  - Model interpretability

### Gelecek Modeller için Durum
- **Neural Networks**: Kesinlikle gerekli
- **SVM**: Kesinlikle gerekli
- **Logistic Regression**: Kesinlikle gerekli
- **K-Means Clustering**: Kesinlikle gerekli
- **PCA**: Kesinlikle gerekli

### Sonuç
✅ **Feature scaling eklenmeli** çünkü:
1. Gelecekteki model değişikliklerine hazırlıklı olur
2. Feature importance karşılaştırmalarını daha anlamlı hale getirir
3. Model interpretability'yi artırır
4. Production pipeline'ında tutarlılık sağlar

## 🏗️ Modüler Mimari Planı

### Klasör Yapısı
```
src/
├── preprocessing/
│   ├── __init__.py
│   ├── base.py              # BaseScaler abstract class
│   ├── scalers/
│   │   ├── __init__.py
│   │   ├── standard.py      # StandardScaler wrapper
│   │   ├── minmax.py        # MinMaxScaler wrapper
│   │   ├── robust.py        # RobustScaler wrapper
│   │   └── quantile.py       # QuantileTransformer wrapper
│   └── preprocessor.py      # Ana Preprocessor sınıfı
```

### Sınıf Tasarımı

#### 1. BaseScaler (Abstract Base Class)
```python
class BaseScaler(ABC):
    """Base scaler interface"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'BaseScaler':
        """Fit scaler on training data"""
        
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform"""
        
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform (for predictions)"""
```

#### 2. Concrete Scalers
- `StandardScaler`: Mean=0, Std=1 (outlier'lara hassas)
- `MinMaxScaler`: 0-1 arası (outlier'lara hassas)
- `RobustScaler`: Median ve IQR kullanır (outlier'lara dayanıklı) ⭐ **Önerilen**
- `QuantileTransformer`: Uniform veya normal dağılım (non-linear)

#### 3. Preprocessor Sınıfı
```python
class Preprocessor:
    """Ana preprocessing sınıfı"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.scaler = None
        self.feature_columns = None
        self.excluded_columns = None
        
    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """Fit scaler on training data"""
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data (train/test)"""
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform"""
        
    def save_scaler(self, filepath: str) -> None:
        """Save scaler for production"""
        
    def load_scaler(self, filepath: str) -> None:
        """Load scaler for production"""
```

### Config Entegrasyonu

```python
@dataclass
class PreprocessingConfig:
    """Preprocessing konfigürasyonu"""
    
    # Scaling ayarları
    enable_scaling: bool = True
    scaler_type: str = "robust"  # 'standard', 'minmax', 'robust', 'quantile'
    
    # Hangi kolonlar scale edilmeyecek
    exclude_columns: List[str] = field(default_factory=lambda: [
        'datetime',
        'target',
        'future_return',
        'hour',
        'day_of_week',
        'month',
        'is_weekend',
        'rsi', 'rsi_7', 'rsi_21',  # Zaten 0-100 arası
        'stoch_k', 'stoch_d',  # Zaten 0-100 arası
        'adx', 'adx_pos', 'adx_neg',  # Zaten 0-100 arası
        'bb_position',  # Zaten 0-1 arası
        'price_position',  # Zaten 0-1 arası
    ])
    
    # Scaler parametreleri
    robust_quantile_range: Tuple[float, float] = (0.25, 0.75)
    standard_with_mean: bool = True
    standard_with_std: bool = True
    minmax_feature_range: Tuple[float, float] = (0, 1)
```

## 🔄 Pipeline Entegrasyonu

### Mevcut Pipeline
```
Data Collection → Feature Engineering → Model Training
```

### Yeni Pipeline
```
Data Collection → Feature Engineering → Preprocessing (Scaling) → Model Training
```

### Kullanım Senaryoları

#### Senaryo 1: Feature Engineering Sonrası
```python
# Feature engineering
fe = FeatureEngineer()
df_features = fe.create_features(df_raw)

# Preprocessing (scaling)
preprocessor = Preprocessor()
df_scaled = preprocessor.fit_transform(df_features)

# Model training
X = df_scaled.drop(['target', 'datetime'], axis=1)
y = df_scaled['target']
```

#### Senaryo 2: Train/Test Split Sonrası
```python
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit scaler on training data only
preprocessor = Preprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)  # Only transform, don't fit!
```

## 📈 Beklenen Sonuçlar ve Faydalar

### 1. Model Performansı
- **LightGBM**: Minimal etki (tree-based olduğu için)
- **Gelecek modeller**: Kritik fayda (neural network, SVM vb.)

### 2. Feature Importance
- Daha anlamlı karşılaştırmalar
- Ölçek farklılıklarından kaynaklanan yanlış yorumlamaların önlenmesi

### 3. Hyperparameter Tuning
- Daha stabil sonuçlar
- Learning rate gibi parametrelerin daha tutarlı çalışması

### 4. Model Interpretability
- Feature'ların etkilerini daha iyi anlama
- SHAP değerleri gibi interpretability tool'ları için hazırlık

### 5. Production Hazırlığı
- Scaler'ların kaydedilmesi ve yüklenmesi
- Yeni veriler için tutarlı preprocessing

### 6. Gelecek Hazırlığı
- Neural network, SVM gibi modellere geçiş kolaylaşır
- Ensemble modeller için hazırlık

## ⚠️ Dikkat Edilmesi Gerekenler

### 1. Data Leakage Önleme
- ✅ Scaler'ı **sadece training data** üzerinde fit et
- ✅ Test data'yı **sadece transform** et (fit etme!)
- ✅ Production'da aynı scaler'ı kullan

### 2. Target Variable
- ❌ Target variable'ı **asla scale etme**
- ❌ `future_return` gibi regression target'ları scale etme

### 3. Categorical Features
- ❌ Datetime, hour, day_of_week gibi categorical feature'ları scale etme
- ✅ One-hot encoding veya label encoding kullan

### 4. Zaten Normalize Olan Feature'lar
- RSI, ADX, Stochastic gibi 0-100 arası feature'lar opsiyonel
- Returns gibi zaten küçük değerler opsiyonel
- Ancak tutarlılık için hepsini scale etmek de mantıklı

### 5. Outlier Handling
- RobustScaler outlier'lara dayanıklı (önerilen)
- StandardScaler ve MinMaxScaler outlier'lara hassas

## 🎯 Önerilen Yaklaşım

### 1. Scaler Seçimi
**RobustScaler** önerilir çünkü:
- Outlier'lara dayanıklı (kripto piyasasında önemli)
- Median ve IQR kullanır (daha robust)
- Tree-based modeller için yeterli

### 2. Exclude Listesi
```python
exclude_columns = [
    'datetime',
    'target',
    'future_return',
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
]
```

### 3. Pipeline Sırası
1. Feature Engineering
2. Train/Test Split
3. Fit scaler on training data
4. Transform both train and test
5. Model Training

## 📝 Implementation Checklist

- [ ] `src/preprocessing/` klasör yapısını oluştur
- [ ] `BaseScaler` abstract class'ı oluştur
- [ ] `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer` wrapper'ları oluştur
- [ ] `Preprocessor` ana sınıfını oluştur
- [ ] `PreprocessingConfig` config'e ekle
- [ ] Unit testler yaz
- [ ] Feature engineering pipeline'ına entegre et
- [ ] Model training pipeline'ına entegre et
- [ ] Scaler save/load fonksiyonlarını ekle
- [ ] Dokümantasyon güncelle

## 🔍 Özeleştiri

### Detaylı İncelenmesi Gerekenler
1. ✅ Feature ölçekleri analiz edildi
2. ✅ Mevcut pipeline yapısı incelendi
3. ✅ Config yapısı incelendi
4. ✅ Model tipi (LightGBM) dikkate alındı
5. ⚠️ Model training modülü henüz yok - bu entegrasyon için önemli
6. ⚠️ Production deployment senaryosu düşünülmeli
7. ⚠️ Backtesting modülü ile uyumluluk kontrol edilmeli

### Şüpheli Durumlar
- Model training modülü henüz yok, bu yüzden entegrasyon tam planlanamadı
- Backtesting modülü var mı kontrol edilmeli (scaling'in geri alınması gerekebilir)
- Production'da scaler'ların nasıl yönetileceği detaylandırılmalı

