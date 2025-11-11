# Walk-Forward Validation Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### Expanding Window (Varsayılan)

```python
from src.config import get_config
from src.model_training import ModelTrainer

config = get_config()
config.model.use_walk_forward = True
config.model.walk_forward_type = "expanding"  # Varsayılan
config.model.walk_forward_n_splits = 5

trainer = ModelTrainer(config)
trainer.train(df_features)
```

### Rolling Window

```python
config.model.use_walk_forward = True
config.model.walk_forward_type = "rolling"
config.model.walk_forward_window_size = 10000  # Son 10k satır
config.model.walk_forward_n_splits = 5

trainer = ModelTrainer(config)
trainer.train(df_features)
```

## 📊 İki Türü Karşılaştırma

### Senaryo: Her İkisini de Test Et

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
trainer.prepare_data(df_features)
trainer.split_data(X, y)

# 1. Expanding Window Test
print("=" * 60)
print("EXPANDING WINDOW TEST")
print("=" * 60)
results_expanding = trainer.walk_forward_validation(
    window_type="expanding",
    n_splits=5,
    test_size=2000
)

# 2. Rolling Window Test
print("\n" + "=" * 60)
print("ROLLING WINDOW TEST")
print("=" * 60)
results_rolling = trainer.walk_forward_validation(
    window_type="rolling",
    n_splits=5,
    test_size=2000,
    window_size=10000  # Son 10k satır
)

# 3. Karşılaştır
print("\n" + "=" * 60)
print("KARŞILAŞTIRMA")
print("=" * 60)
print(f"Expanding - Mean: {results_expanding['mean_score']:.4f} ± {results_expanding['std_score']:.4f}")
print(f"Rolling   - Mean: {results_rolling['mean_score']:.4f} ± {results_rolling['std_score']:.4f}")

# En iyisini seç
if results_expanding['mean_score'] > results_rolling['mean_score']:
    print("✅ Expanding window daha iyi performans gösterdi")
    best_type = "expanding"
else:
    print("✅ Rolling window daha iyi performans gösterdi")
    best_type = "rolling"
```

## ⚙️ Config Ayarları

### Expanding Window

```python
config.model.use_walk_forward = True
config.model.walk_forward_type = "expanding"
config.model.walk_forward_n_splits = 5
config.model.walk_forward_test_size = 2000  # Her fold'ta test boyutu
config.model.walk_forward_gap = 100  # Purged walk-forward için gap
```

### Rolling Window

```python
config.model.use_walk_forward = True
config.model.walk_forward_type = "rolling"
config.model.walk_forward_n_splits = 5
config.model.walk_forward_test_size = 2000  # Her fold'ta test boyutu
config.model.walk_forward_window_size = 10000  # Pencere boyutu (önemli!)
config.model.walk_forward_gap = 100  # Purged walk-forward için gap
```

## 🎯 Ne Zaman Hangi Türü Kullanmalı?

### Expanding Window Kullan:
- ✅ Küçük-orta veri setleri (< 20,000 satır)
- ✅ Uzun vadeli trend analizi
- ✅ Tüm geçmiş pattern'leri öğrenmek istiyorsanız
- ✅ Model'in tüm geçmişten öğrenmesini istiyorsanız

### Rolling Window Kullan:
- ✅ **Büyük veri setleri (> 50,000 satır)** ⭐
- ✅ **Market regime değişiklikleri sık** ⭐
- ✅ **Eski veriler gereksiz olabilir** ⭐
- ✅ **Trend değişikliklerine hızlı adapte olmak istiyorsanız** ⭐
- ✅ **Hesaplama hızı önemli** ⭐

## 📊 Örnek: Büyük Veri Seti (50,000 satır)

### Expanding Window

```python
config.model.walk_forward_type = "expanding"
config.model.walk_forward_n_splits = 5
config.model.walk_forward_test_size = 2000

# Fold 1: Train [0-10k] → Test [10k-12k]
# Fold 2: Train [0-12k] → Test [12k-14k]
# Fold 3: Train [0-14k] → Test [14k-16k]
# Fold 4: Train [0-16k] → Test [16k-18k]
# Fold 5: Train [0-18k] → Test [18k-20k]

# Son fold'ta 18k satır train (ilk 10k çok eski olabilir!)
```

### Rolling Window

```python
config.model.walk_forward_type = "rolling"
config.model.walk_forward_n_splits = 5
config.model.walk_forward_test_size = 2000
config.model.walk_forward_window_size = 10000  # Son 10k satır

# Fold 1: Train [0-10k] → Test [10k-12k]
# Fold 2: Train [2k-12k] → Test [12k-14k]  # İlk 2k çıkarıldı
# Fold 3: Train [4k-14k] → Test [14k-16k]  # İlk 4k çıkarıldı
# Fold 4: Train [6k-16k] → Test [16k-18k]  # İlk 6k çıkarıldı
# Fold 5: Train [8k-18k] → Test [18k-20k]  # İlk 8k çıkarıldı

# Her fold'ta sadece son 10k satır (daha güncel!)
```

## 🔧 Manuel Kullanım

### Expanding Window

```python
trainer = ModelTrainer()
trainer.prepare_data(df_features)
trainer.split_data(X, y)

results = trainer.walk_forward_validation(
    window_type="expanding",
    n_splits=5,
    test_size=2000,
    gap=100
)
```

### Rolling Window

```python
results = trainer.walk_forward_validation(
    window_type="rolling",
    n_splits=5,
    test_size=2000,
    window_size=10000,  # Önemli!
    gap=100
)
```

## 📈 Sonuçları İnceleme

```python
results = trainer.walk_forward_validation()

# Fold skorları
print("Fold Scores:", results['fold_scores'])

# İstatistikler
print(f"Mean: {results['mean_score']:.4f}")
print(f"Std: {results['std_score']:.4f}")
print(f"Min: {results['min_score']:.4f}")
print(f"Max: {results['max_score']:.4f}")

# Her fold'un detaylı metrikleri
for i, metrics in enumerate(results['fold_metrics']):
    print(f"\nFold {i+1} Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value:.4f}")
```

## 💡 İpuçları

### Window Size Seçimi (Rolling)

```python
# Veri boyutuna göre öneriler:
n_samples = len(X_train)

if n_samples < 10000:
    window_size = n_samples // 2  # %50
elif n_samples < 50000:
    window_size = 10000  # Sabit 10k
else:
    window_size = 20000  # Sabit 20k veya %20
```

### Test Size Seçimi

```python
# Veri boyutuna göre:
if n_samples < 10000:
    test_size = 500  # Küçük veri setleri
elif n_samples < 50000:
    test_size = 2000  # Orta veri setleri
else:
    test_size = 5000  # Büyük veri setleri
```

### Gap Seçimi (Purged Walk-Forward)

```python
# Time series için gap önerileri:
# - 1 saatlik veri için: gap = 0-10
# - 1 günlük veri için: gap = 1-5
# - Büyük veri setleri için: gap = 50-200

gap = 100  # Genelde 50-200 arası iyi çalışır
```

## ✅ Özet

- ✅ **Expanding Window**: Küçük veri setleri, uzun vadeli analiz
- ✅ **Rolling Window**: Büyük veri setleri, hızlı adaptasyon
- ✅ **Her İkisini de Test Et**: Performansı karşılaştır
- ✅ **Config ile Kontrol**: Kolay kullanım
- ✅ **Manuel Override**: İstediğiniz zaman parametreleri değiştirebilirsiniz

