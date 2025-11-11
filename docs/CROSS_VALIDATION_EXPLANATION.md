# Cross Validation Açıklaması

## ❓ Cross Validation Nedir?

Cross Validation (Çapraz Doğrulama), model performansını daha güvenilir bir şekilde değerlendirmek için kullanılan bir tekniktir.

## 🔄 Mevcut Durum: Train/Test Split

### Şu An Ne Yapılıyor?

```
Tüm Veri (1000 satır)
  ↓
Train/Test Split
  ├─ Train: 800 satır (80%)
  └─ Test: 200 satır (20%)
      ↓
Model Train → Test Evaluation
```

**Sorunlar:**
- ❌ Tek bir split'e bağlı (split'e göre sonuçlar değişebilir)
- ❌ Küçük veri setlerinde güvenilir değil
- ❌ Model'in farklı veri bölümlerinde nasıl performans gösterdiği bilinmiyor

## ✅ Cross Validation Nasıl Çalışır?

### K-Fold Cross Validation (En Yaygın)

```
Tüm Veri (1000 satır)
  ↓
5-Fold Cross Validation:
  
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]

Her fold için:
  - Model eğitilir
  - Test fold'unda değerlendirilir
  - Metrikler kaydedilir

Sonuç: 5 farklı metrik → Ortalama ve standart sapma
```

### Stratified K-Fold (Classification için)

- Class dağılımını korur
- Her fold'ta aynı class oranları olur
- Binary ve multi-class için önemli

## 🎯 Ne İşe Yarar?

### 1. **Daha Güvenilir Performans Tahmini**

```python
# Train/Test Split (Tek sonuç)
Accuracy: 0.85  # Bu sonuç güvenilir mi?

# Cross Validation (5-Fold)
Fold 1: 0.84
Fold 2: 0.86
Fold 3: 0.85
Fold 4: 0.83
Fold 5: 0.87
─────────────────
Mean: 0.85 ± 0.014  # Daha güvenilir!
```

### 2. **Overfitting Tespiti**

```python
# Eğer fold'lar arasında büyük fark varsa:
Fold 1: 0.95  # Overfitting olabilir
Fold 2: 0.75
Fold 3: 0.80
Fold 4: 0.90
Fold 5: 0.70
─────────────────
Mean: 0.82 ± 0.10  # Yüksek varyans = Overfitting riski
```

### 3. **Hyperparameter Tuning**

```python
# Farklı parametreleri test etmek için:
for params in param_grid:
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_score = cv_scores.mean()
    # En iyi parametreleri seç
```

### 4. **Küçük Veri Setlerinde Kritik**

```python
# 100 satır veri:
# Train/Test Split (80/20):
#   Train: 80 satır → Çok az!
#   Test: 20 satır → Güvenilir değil

# Cross Validation (5-Fold):
#   Her fold: 80 train, 20 test
#   5 farklı değerlendirme → Daha güvenilir
```

## 📊 Karşılaştırma

| Özellik | Train/Test Split | Cross Validation |
|---------|------------------|------------------|
| **Güvenilirlik** | ⚠️ Tek sonuç | ✅ Ortalama + Standart sapma |
| **Veri Kullanımı** | ⚠️ Test set kullanılmaz | ✅ Tüm veri kullanılır |
| **Overfitting Tespiti** | ❌ Zor | ✅ Kolay |
| **Hesaplama Maliyeti** | ✅ Düşük | ⚠️ Yüksek (K katı) |
| **Küçük Veri Setleri** | ❌ Uygun değil | ✅ Uygun |

## 🎯 Ne Zaman Kullanılır?

### Cross Validation Kullan:
- ✅ Küçük veri setleri (< 1000 satır)
- ✅ Model performansını güvenilir değerlendirmek istediğinizde
- ✅ Hyperparameter tuning yaparken
- ✅ Overfitting riskini kontrol etmek istediğinizde
- ✅ Model karşılaştırması yaparken

### Train/Test Split Yeterli:
- ✅ Büyük veri setleri (> 10,000 satır)
- ✅ Hızlı prototyping
- ✅ Production model eğitimi (final model)
- ✅ Hesaplama kaynağı sınırlı

## 💡 Önerilen Yaklaşım

### Hibrit Yaklaşım (En İyi)

```
1. Train/Test Split (80/20)
   └─ Test set: Final değerlendirme için sakla

2. Train set üzerinde Cross Validation
   └─ Model seçimi ve hyperparameter tuning

3. Final model'i tüm train set ile eğit
   └─ Test set'te final değerlendirme
```

## 🔧 Sklearn Cross Validation Fonksiyonları

### 1. cross_val_score (Basit)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 2. cross_validate (Detaylı)

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X_train, y_train, 
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)
```

### 3. StratifiedKFold (Classification)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=skf)
```

### 4. TimeSeriesSplit (Time Series için)

```python
from sklearn.model_selection import TimeSeriesSplit

# Kripto verisi için önemli!
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv)
```

## 🚀 Projeye Eklenmeli mi?

### ✅ Evet, Eklenmeli Çünkü:

1. **Kripto Verisi**: Time series olduğu için TimeSeriesSplit önemli
2. **Model Güvenilirliği**: Daha güvenilir performans değerlendirmesi
3. **Hyperparameter Tuning**: Gelecekte gerekli olacak
4. **Modüler Yapı**: Mevcut yapıya kolay entegre edilebilir

### 📝 Önerilen Implementation:

```python
# ModelTrainer'a eklenebilir:
def cross_validate(self, cv=5, scoring='accuracy'):
    """Cross validation yap"""
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    
    # Time series için TimeSeriesSplit kullan
    tscv = TimeSeriesSplit(n_splits=cv)
    
    scores = cross_val_score(
        self.model,
        self.X_train_scaled,
        self.y_train,
        cv=tscv,
        scoring=scoring
    )
    
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }
```

## 📊 Sonuç

- **Mevcut Durum**: Sadece train/test split var
- **Cross Validation**: Yok, ama eklenebilir
- **Öneri**: TimeSeriesSplit ile cross validation eklenmeli
- **Fayda**: Daha güvenilir model değerlendirmesi

