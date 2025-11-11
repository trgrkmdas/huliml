# Walk-Forward Validation - Time Series için Kritik

## 🎯 Neden Walk-Forward Validation?

### Time Series Verilerinde Sorun

```
Normal Train/Test Split:
[Train: 2020-2023] [Test: 2024]
  ↓
Problem: Model 2024'ü tahmin ederken 2024 verilerini "görmüş" gibi davranır
```

### Walk-Forward Validation (Gerçek Dünya Senaryosu)

```
Walk-Forward (Expanding Window):
[Train: 2020] → [Test: 2021]
[Train: 2020-2021] → [Test: 2022]
[Train: 2020-2022] → [Test: 2023]
[Train: 2020-2023] → [Test: 2024]

Her adımda:
- Sadece geçmiş verilerle eğitilir
- Geleceği tahmin eder
- Gerçek dünya senaryosunu simüle eder
```

## 📊 Walk-Forward Validation Türleri

### 1. Expanding Window (Genişleyen Pencere)

```
Fold 1: [Train: 1000] → [Test: 200]
Fold 2: [Train: 1200] → [Test: 200]
Fold 3: [Train: 1400] → [Test: 200]
Fold 4: [Train: 1600] → [Test: 200]
Fold 5: [Train: 1800] → [Test: 200]

✅ Avantaj: Daha fazla veri kullanır
⚠️ Dezavantaj: Eski veriler modeli etkileyebilir
```

### 2. Rolling Window (Sabit Pencere)

```
Fold 1: [Train: 1000] → [Test: 200]
Fold 2: [Train: 200-1200] → [Test: 200]
Fold 3: [Train: 400-1400] → [Test: 200]
Fold 4: [Train: 600-1600] → [Test: 200]
Fold 5: [Train: 800-1800] → [Test: 200]

✅ Avantaj: Sadece son verileri kullanır (trend değişikliklerine adapte olur)
⚠️ Dezavantaj: Eski verileri kaybeder
```

### 3. Purged Walk-Forward (Gap ile)

```
Fold 1: [Train: 1000] → [Gap: 50] → [Test: 200]
Fold 2: [Train: 1250] → [Gap: 50] → [Test: 200]

✅ Avantaj: Data leakage önler (önemli!)
⚠️ Dezavantaj: Daha az veri kullanır
```

## 🚀 Büyük Veri Setleri için Neden Önemli?

### Senaryo: 50,000 satır Bitcoin verisi

#### Normal Train/Test Split:
```python
Train: 40,000 satır (2020-2023)
Test: 10,000 satır (2024)

Problem:
- Model 2024'ün başındaki pattern'leri öğrenmiş olabilir
- 2024'ün sonundaki yeni trend'leri yakalayamaz
- Market regime değişikliklerini test edemez
```

#### Walk-Forward Validation:
```python
Fold 1: Train [0-10k] → Test [10k-12k]    # 2020 → 2021 başı
Fold 2: Train [0-12k] → Test [12k-14k]   # 2020-2021 → 2021 sonu
Fold 3: Train [0-14k] → Test [14k-16k]   # 2020-2022 → 2022 başı
...
Fold 10: Train [0-40k] → Test [40k-42k]  # 2020-2023 → 2024 başı

Avantajlar:
✅ Her zaman sadece geçmiş verilerle eğitilir
✅ Farklı market regime'lerini test eder
✅ Model'in zaman içindeki performansını görürsünüz
✅ Gerçek trading senaryosunu simüle eder
```

## 💡 Kripto Verisi için Özel Önemi

### 1. Market Regime Değişiklikleri

```
2020: Bull market başlangıcı
2021: Bull market zirvesi
2022: Bear market
2023: Recovery
2024: Yeni trend?

Walk-forward her regime'i test eder!
```

### 2. Model Drift Tespiti

```python
Fold 1: Accuracy 0.85  # 2020-2021
Fold 2: Accuracy 0.82  # 2020-2022
Fold 3: Accuracy 0.75  # 2020-2023 ⚠️ Düşüş!
Fold 4: Accuracy 0.70  # 2020-2024 ⚠️ Daha da düşüş!

→ Model drift var! Model güncellenmeli.
```

### 3. Gerçek Trading Senaryosu

```
Gerçek trading'de:
- Her gün yeni veri gelir
- Model sürekli güncellenir
- Geleceği tahmin eder

Walk-forward bunu simüle eder!
```

## 🔧 Implementation Önerisi

### Büyük Veri Setleri için Optimize Edilmiş

```python
class WalkForwardValidator:
    """Walk-forward validation for time series"""
    
    def __init__(
        self,
        initial_train_size: int = 10000,  # İlk train set boyutu
        test_size: int = 2000,             # Her fold'ta test boyutu
        step_size: int = 1000,            # Her fold'ta ne kadar ilerle
        gap: int = 0,                     # Train-test arası gap (purged)
        expanding: bool = True,            # Expanding mi rolling mi?
    ):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.gap = gap
        self.expanding = expanding
    
    def split(self, X, y):
        """Walk-forward splits oluştur"""
        n_samples = len(X)
        splits = []
        
        train_start = 0
        train_end = self.initial_train_size
        
        while train_end + self.gap + self.test_size <= n_samples:
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            splits.append({
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            })
            
            # Sonraki fold için train_end'i güncelle
            if self.expanding:
                # Expanding: Train set büyür
                train_end += self.step_size
            else:
                # Rolling: Train set kayar
                train_start += self.step_size
                train_end += self.step_size
        
        return splits
```

## 📊 Kullanım Örneği

### Büyük Veri Seti (50,000 satır)

```python
from src.model_training import ModelTrainer
from sklearn.model_selection import TimeSeriesSplit

# Veri hazırla
trainer = ModelTrainer()
X, y = trainer.prepare_data(df_features)

# Walk-forward validation
tscv = TimeSeriesSplit(
    n_splits=10,           # 10 fold
    test_size=2000,        # Her fold'ta 2000 satır test
    gap=100,               # 100 satır gap (purged)
)

scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}:")
    print(f"  Train: {len(train_idx)} satır")
    print(f"  Test: {len(test_idx)} satır")
    
    # Her fold için model eğit
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Preprocessing
    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train_fold)
    X_test_scaled = preprocessor.transform(X_test_fold)
    
    # Model eğit
    model = LightGBMModel()
    model.fit(X_train_scaled, y_train_fold)
    
    # Değerlendir
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test_fold, y_pred)
    scores.append(score)
    print(f"  Accuracy: {score:.4f}")

print(f"\nWalk-Forward Results:")
print(f"  Mean: {np.mean(scores):.4f}")
print(f"  Std: {np.std(scores):.4f}")
print(f"  Min: {np.min(scores):.4f}")
print(f"  Max: {np.max(scores):.4f}")
```

## 🎯 Önerilen Yaklaşım

### Büyük Veri Setleri için:

1. **Walk-Forward Validation** (Model seçimi için)
   - TimeSeriesSplit kullan
   - 5-10 fold
   - Her fold'ta 1000-5000 satır test

2. **Final Train/Test Split** (Final model için)
   - Son %20'yi test olarak sakla
   - Walk-forward'dan öğrenilen parametrelerle final model eğit

### Hibrit Yaklaşım:

```python
# 1. Walk-Forward ile model seçimi ve hyperparameter tuning
walk_forward_scores = walk_forward_validation(X_train, y_train)

# 2. En iyi parametrelerle final model eğit
best_model = train_final_model(X_train, y_train, best_params)

# 3. Test set'te final değerlendirme
final_score = evaluate(best_model, X_test, y_test)
```

## ✅ Sonuç

**Büyük veri setleri için bile Walk-Forward Validation önerilir çünkü:**

1. ✅ **Time Series**: Geçmiş verilerle geleceği tahmin ediyoruz
2. ✅ **Market Regime**: Farklı market koşullarını test eder
3. ✅ **Model Drift**: Zaman içindeki performans değişikliklerini gösterir
4. ✅ **Gerçekçilik**: Gerçek trading senaryosunu simüle eder
5. ✅ **Güvenilirlik**: Tek bir split'e bağlı kalmaz

**Özellikle kripto verisi için kritik!**

