# Early Stopping Kullanım Kılavuzu

## 🎯 Early Stopping Nedir?

Early stopping, model eğitimi sırasında validation set performansı iyileşmediğinde eğitimi durduran bir tekniktir. Overfitting'i önler ve eğitim süresini optimize eder.

## ✅ Nasıl Çalışır?

```
Eğitim başlar
  ↓
Her iterasyonda validation set'te değerlendirilir
  ↓
Validation loss iyileşmiyor mu?
  ├─ Evet → Early stopping devreye girer, eğitim durur
  └─ Hayır → Eğitim devam eder
  ↓
En iyi model (en düşük validation loss) kaydedilir
```

## 🚀 Kullanım

### Basit Kullanım (Config ile)

```python
from src.config import get_config
from src.model_training import ModelTrainer

config = get_config()

# Early stopping aktif et
config.model.early_stopping_rounds = 50  # 50 iterasyon iyileşme yoksa dur
config.model.validation_size = 0.1  # %10 validation set (gerekli!)

# Model training
trainer = ModelTrainer(config)
trainer.train(df_features)
```

### Early Stopping'i Kapatma

```python
# Early stopping'i kapat
config.model.early_stopping_rounds = None

# veya validation_size=0 yap
config.model.validation_size = 0
```

### Özelleştirme

```python
config.model.early_stopping_rounds = 100  # 100 iterasyon bekle
config.model.early_stopping_verbose = True  # Mesajları göster
config.model.validation_size = 0.15  # %15 validation set
```

## ⚙️ Config Ayarları

```python
# src/config.py - ModelConfig
early_stopping_rounds: Optional[int] = 50  # None ise kapalı
early_stopping_verbose: bool = True  # Mesajları göster
validation_size: float = 0.1  # %10 validation set (early stopping için gerekli)
```

## 📊 Örnek Çıktı

### Early Stopping Aktif

```
🚀 Model eğitimi başlatılıyor...
⏹️  Early stopping aktif: 50 round
[100]	training's binary_logloss: 0.12345	validation's binary_logloss: 0.12567
[200]	training's binary_logloss: 0.11234	validation's binary_logloss: 0.11890
[250]	training's binary_logloss: 0.10987	validation's binary_logloss: 0.11567
[300]	training's binary_logloss: 0.10890	validation's binary_logloss: 0.11523
[350]	training's binary_logloss: 0.10765	validation's binary_logloss: 0.11545  ← İyileşme yok!
Early stopping, best iteration is:
[300]	training's binary_logloss: 0.10890	validation's binary_logloss: 0.11523
✅ Model eğitimi tamamlandı
```

### Early Stopping Kapalı

```
🚀 Model eğitimi başlatılıyor...
[100]	training's binary_logloss: 0.12345
[200]	training's binary_logloss: 0.11234
...
[1000]	training's binary_logloss: 0.09876  ← Tüm iterasyonlar tamamlandı
✅ Model eğitimi tamamlandı
```

## 💡 Öneriler

### Early Stopping Rounds Seçimi

```python
# Küçük veri setleri (< 10k satır)
early_stopping_rounds = 30-50

# Orta veri setleri (10k-50k satır)
early_stopping_rounds = 50-100

# Büyük veri setleri (> 50k satır)
early_stopping_rounds = 100-200
```

### Validation Size Seçimi

```python
# Küçük veri setleri
validation_size = 0.15-0.2  # %15-20

# Büyük veri setleri
validation_size = 0.1  # %10 yeterli
```

## ⚠️ Önemli Notlar

### 1. Validation Set Gerekli

Early stopping için **mutlaka validation set gerekli**:

```python
# ✅ Doğru
config.model.validation_size = 0.1  # %10 validation
config.model.early_stopping_rounds = 50

# ❌ Yanlış - Early stopping çalışmaz
config.model.validation_size = 0
config.model.early_stopping_rounds = 50  # Uyarı verilir ama çalışmaz
```

### 2. Overfitting Önleme

Early stopping overfitting'i önler:

```python
# Early stopping YOK:
# Training loss: 0.05 (çok düşük - overfitting!)
# Validation loss: 0.15 (yüksek - overfitting!)

# Early stopping VAR:
# Training loss: 0.08 (daha yüksek ama gerçekçi)
# Validation loss: 0.12 (daha düşük - overfitting yok!)
```

### 3. Eğitim Süresi

Early stopping eğitim süresini optimize eder:

```python
# Early stopping YOK: 1000 iterasyon (uzun sürer)
# Early stopping VAR: 300 iterasyon (hızlı, en iyi model)
```

## 🔍 Debugging

### Early Stopping Çalışmıyor?

```python
# 1. Validation set kontrolü
print(f"Validation size: {config.model.validation_size}")
# 0 ise early stopping çalışmaz!

# 2. Early stopping rounds kontrolü
print(f"Early stopping rounds: {config.model.early_stopping_rounds}")
# None ise early stopping kapalı

# 3. Verbose kontrolü
config.model.early_stopping_verbose = True
# Mesajları görmek için
```

## ✅ Sonuç

Early stopping:
- ✅ **Overfitting önler**
- ✅ **Eğitim süresini optimize eder**
- ✅ **En iyi modeli otomatik seçer**
- ✅ **LightGBM için kritik**

**Kullanım**: Validation set + early_stopping_rounds ayarla, gerisi otomatik!

