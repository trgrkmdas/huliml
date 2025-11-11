# Feature Scaling/Normalization - Özet Rapor

## 📋 Özet

Bu doküman, MLProject için feature scaling/normalization modülünün analizi, planlaması ve implementasyon stratejisini içerir.

## 🔍 Mevcut Durum

### Proje Özellikleri
- **Model**: LightGBM (Gradient Boosting - Tree-based)
- **Problem**: Binary Classification (Long/Short sinyali tahmini)
- **Feature Sayısı**: ~64 feature
- **Veri Tipi**: Bitcoin/USDT kripto para verisi

### Feature Kategorileri ve Ölçekleri

| Kategori | Örnek Feature'lar | Ölçek Aralığı | Scale Edilmeli? |
|----------|-------------------|---------------|-----------------|
| **OHLCV** | open, high, low, close | 46,000 - 70,000 | ✅ Evet |
| **Volume** | volume, volume_sma | 500 - 5,000 | ✅ Evet |
| **Trend** | sma_*, ema_* | 46,000 - 70,000 | ✅ Evet |
| **Momentum** | macd, macd_signal | -500 - 500 | ✅ Evet |
| **Momentum (Norm)** | rsi, stoch_k | 0 - 100 | ⚠️ Opsiyonel |
| **Volatilite** | volatility_* | 0.002 - 0.008 | ✅ Evet |
| **Volatilite (Norm)** | bb_position | 0 - 1 | ⚠️ Opsiyonel |
| **Returns** | returns, returns_* | -0.1 - 0.1 | ⚠️ Opsiyonel |
| **Ratios** | high_low_ratio | 0.9 - 1.1 | ⚠️ Opsiyonel |
| **Zaman** | hour, day_of_week | Categorical | ❌ Hayır |
| **Target** | target, future_return | Binary/Regression | ❌ Hayır |

## ❓ Feature Scaling Gerekli mi?

### LightGBM için
- **Kısa cevap**: Hayır, zorunlu değil
- **Uzun cevap**: Evet, eklenmeli çünkü:
  1. ✅ Gelecekteki model değişikliklerine hazırlık
  2. ✅ Feature importance karşılaştırmalarını daha anlamlı hale getirir
  3. ✅ Hyperparameter tuning stabilitesi
  4. ✅ Model interpretability artışı
  5. ✅ Production pipeline tutarlılığı

### Gelecek Modeller için
- **Neural Networks**: ✅ Kesinlikle gerekli
- **SVM**: ✅ Kesinlikle gerekli
- **Logistic Regression**: ✅ Kesinlikle gerekli
- **K-Means Clustering**: ✅ Kesinlikle gerekli

## 🎯 Ne İşe Yarayacak?

### 1. Model Performansı
- **LightGBM**: Minimal etki (tree-based olduğu için)
- **Gelecek modeller**: Kritik fayda

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

## 🏗️ Modüler Mimari

### Klasör Yapısı
```
src/preprocessing/
├── __init__.py
├── base.py              # BaseScaler abstract class
├── preprocessor.py      # Ana Preprocessor sınıfı
└── scalers/
    ├── __init__.py
    ├── standard.py      # StandardScaler
    ├── minmax.py        # MinMaxScaler
    ├── robust.py        # RobustScaler ⭐ Önerilen
    └── quantile.py      # QuantileTransformer
```

### Scaler Seçimi

#### ⭐ RobustScaler (Önerilen)
- **Neden?**: Outlier'lara dayanıklı (kripto piyasasında önemli)
- **Nasıl çalışır?**: Median ve IQR kullanır
- **Avantajlar**: 
  - Outlier'lara dayanıklı
  - Tree-based modeller için yeterli
  - Daha robust istatistikler

#### Alternatifler
- **StandardScaler**: Mean=0, Std=1 (outlier'lara hassas)
- **MinMaxScaler**: 0-1 arası (outlier'lara hassas)
- **QuantileTransformer**: Uniform/normal dağılım (non-linear)

## 🔄 Pipeline Entegrasyonu

### Mevcut Pipeline
```
Data Collection → Feature Engineering → Model Training
```

### Yeni Pipeline
```
Data Collection → Feature Engineering → Preprocessing (Scaling) → Model Training
```

### Kullanım Örneği
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

### 4. Exclude Columns
Varsayılan exclude listesi:
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

## 📈 Beklenen Sonuçlar

### Performans
- LightGBM için minimal etki
- Gelecek modeller için kritik fayda
- Feature importance daha anlamlı

### Kod Kalitesi
- Modüler yapı
- Test edilebilir
- Genişletilebilir
- Production-ready

### Kullanıcı Deneyimi
- Kolay kullanım
- Config ile kontrol
- Otomatik exclude columns
- Clear logging

## 📝 Implementation Checklist

- [ ] `src/preprocessing/` klasör yapısını oluştur
- [ ] `BaseScaler` abstract class'ı oluştur
- [ ] `RobustScaler` wrapper'ı oluştur (öncelikli)
- [ ] `Preprocessor` ana sınıfını oluştur
- [ ] `PreprocessingConfig` config'e ekle
- [ ] Unit testler yaz
- [ ] Feature engineering pipeline'ına entegre et
- [ ] Model training pipeline'ına entegre et
- [ ] Scaler save/load fonksiyonlarını ekle
- [ ] Dokümantasyon güncelle

## 📚 Detaylı Dokümanlar

1. **FEATURE_SCALING_ANALYSIS.md**: Detaylı analiz ve planlama
2. **FEATURE_SCALING_IMPLEMENTATION_PLAN.md**: Implementation detayları ve kod örnekleri

## 🎯 Sonuç

Feature scaling/normalization modülü:
- ✅ **Gerekli**: Gelecek modeller ve production için kritik
- ✅ **Faydalı**: Feature importance ve interpretability için önemli
- ✅ **Modüler**: Mevcut yapıya uyumlu, genişletilebilir
- ✅ **Production-ready**: Save/load functionality ile hazır

**Önerilen Yaklaşım**: RobustScaler kullanarak modüler bir preprocessing modülü oluşturmak.

