# Feature Scaling Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### Temel Kullanım

```python
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer

# Feature engineering
fe = FeatureEngineer()
df_features = fe.create_features(df_raw)

# Preprocessing (scaling)
preprocessor = Preprocessor()
df_scaled = preprocessor.fit_transform(df_features)

# Model training için hazır
X = df_scaled.drop(['target', 'datetime'], axis=1)
y = df_scaled['target']
```

### Train/Test Split Senaryosu (Önerilen)

```python
from sklearn.model_selection import train_test_split
from src.preprocessing import Preprocessor

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler on training data only
preprocessor = Preprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)  # Only transform!

# Model training
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
```

## ⚙️ Config Ayarları

### Scaler Tipi Değiştirme

```python
from src.config import get_config

config = get_config()
config.preprocessing.scaler_type = "standard"  # veya "minmax", "quantile"
```

### Exclude Columns Ekleme

```python
config.preprocessing.exclude_columns.append("custom_column")
```

### Scaling'i Devre Dışı Bırakma

```python
config.preprocessing.enable_scaling = False
```

## 🔧 Scaler Parametreleri

### RobustScaler (Varsayılan)

```python
config.preprocessing.robust_quantile_range = (0.25, 0.75)  # IQR range
```

### StandardScaler

```python
config.preprocessing.standard_with_mean = True
config.preprocessing.standard_with_std = True
```

### MinMaxScaler

```python
config.preprocessing.minmax_feature_range = (0, 1)  # Output range
```

### QuantileTransformer

```python
config.preprocessing.quantile_n_quantiles = 1000
config.preprocessing.quantile_output_distribution = "uniform"  # veya "normal"
```

## 💾 Scaler Kaydetme ve Yükleme

### Kaydetme

```python
preprocessor = Preprocessor()
preprocessor.fit(df_train)

# Otomatik yol (models/scaler.pkl)
filepath = preprocessor.save_scaler()

# Özel yol
filepath = preprocessor.save_scaler("custom/path/scaler.pkl")
```

### Yükleme (Production)

```python
preprocessor = Preprocessor()
preprocessor.load_scaler("models/scaler.pkl")

# Yeni veriyi transform et
df_new_scaled = preprocessor.transform(df_new)
```

## 🎯 Direkt Scaler Kullanımı

### RobustScaler

```python
from src.preprocessing import RobustScaler

scaler = RobustScaler(
    exclude_columns=["target", "datetime"],
    quantile_range=(0.25, 0.75)
)

df_scaled = scaler.fit_transform(df)
df_original = scaler.inverse_transform(df_scaled)
```

### Diğer Scaler'lar

```python
from src.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

# StandardScaler
scaler = StandardScaler(exclude_columns=["target"], with_mean=True, with_std=True)

# MinMaxScaler
scaler = MinMaxScaler(exclude_columns=["target"], feature_range=(0, 1))

# QuantileTransformer
scaler = QuantileTransformer(
    exclude_columns=["target"],
    n_quantiles=1000,
    output_distribution="uniform"
)
```

## 📊 Örnek: Feature Engineering Pipeline ile Entegrasyon

```python
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split

# 1. Feature Engineering
fe = FeatureEngineer()
df_features = fe.create_features(df_raw)

# 2. Train/Test Split
X = df_features.drop(['target', 'datetime'], axis=1)
y = df_features['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing (Scaling)
preprocessor = Preprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# 4. Model Training
from lightgbm import LGBMClassifier

model = LGBMClassifier()
model.fit(X_train_scaled, y_train)

# 5. Prediction
predictions = model.predict(X_test_scaled)

# 6. Scaler'ı kaydet (production için)
preprocessor.save_scaler()
```

## ⚠️ Önemli Notlar

### 1. Data Leakage Önleme
- ✅ Scaler'ı **sadece training data** üzerinde fit et
- ✅ Test data'yı **sadece transform** et (fit etme!)
- ✅ Production'da aynı scaler'ı kullan

### 2. Target Variable
- ❌ Target variable'ı **asla scale etme**
- ❌ `future_return` gibi regression target'ları scale etme

### 3. Categorical Features
- ❌ Datetime, hour, day_of_week gibi categorical feature'ları scale etme
- ✅ Bu feature'lar otomatik olarak exclude edilir

### 4. Exclude Columns
Varsayılan exclude listesi:
- `datetime`
- `target`
- `future_return`
- `hour`
- `day_of_week`
- `month`
- `is_weekend`

## 🔍 Debugging

### Hangi Feature'lar Scale Ediliyor?

```python
preprocessor = Preprocessor()
preprocessor.fit(df_train)

print("Scale edilen feature'lar:")
print(preprocessor.feature_columns)

print("\nExclude edilen feature'lar:")
print(preprocessor.excluded_columns)
```

### Scaler İstatistikleri

```python
# RobustScaler için
scaler = preprocessor.scaler
print(f"Center (median): {scaler.scaler.center_}")
print(f"Scale (IQR): {scaler.scaler.scale_}")
```

## 📈 Beklenen Sonuçlar

### LightGBM için
- Minimal performans etkisi (tree-based olduğu için)
- Feature importance daha anlamlı
- Hyperparameter tuning daha stabil

### Gelecek Modeller için
- Neural Networks: Kritik fayda
- SVM: Kritik fayda
- Logistic Regression: Kritik fayda

## 🐛 Sorun Giderme

### Hata: "Scaler henüz fit edilmedi"
```python
# Çözüm: Önce fit() çağır
preprocessor.fit(df_train)
df_scaled = preprocessor.transform(df_train)
```

### Hata: "Feature columns belirlenmemiş"
```python
# Çözüm: fit() çağrıldığından emin ol
preprocessor.fit(df_train)
```

### Scaler kaydedilemiyor
```python
# Çözüm: models/ klasörünün var olduğundan emin ol
import os
os.makedirs("models", exist_ok=True)
preprocessor.save_scaler()
```

