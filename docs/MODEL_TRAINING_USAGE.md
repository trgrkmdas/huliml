# Model Training Modülü Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### Basit Kullanım

```python
from src.model_training import ModelTrainer
from src.feature_engineering import FeatureEngineer

# 1. Feature Engineering
fe = FeatureEngineer()
df_features = fe.create_features(df_raw)

# 2. Model Training (Full Pipeline)
trainer = ModelTrainer()
trainer.train(df_features)

# Model otomatik olarak kaydedilir (config'e göre)
```

### Detaylı Kullanım

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()

# Adım adım pipeline
X, y = trainer.prepare_data(df_features)
trainer.split_data(X, y)
trainer.apply_preprocessing()
trainer.train_model()

# Evaluation
metrics = trainer.evaluate()

# Model kaydet
trainer.save_model()
```

## 📊 Pipeline Akışı

```
Feature Engineering
  ↓
Model Training Pipeline:
  1. Prepare Data (X, y separation) ✅
  2. Train/Test Split ✅
  3. Preprocessing (Scaling) ✅
  4. Model Training ✅
  5. Evaluation ✅
  6. Save Model & Preprocessor ✅
```

## ⚙️ Config Ayarları

### Model Tipi

```python
from src.config import get_config

config = get_config()
config.model.model_type = "lightgbm"  # Şu an için sadece lightgbm
```

### Train/Test Split

```python
config.model.test_size = 0.2  # %20 test
config.model.validation_size = 0.1  # %10 validation
config.model.random_seed = 42
```

### Preprocessing

```python
config.model.use_preprocessing = True  # Scaling kullanılsın mı?
```

### Model Saving

```python
config.model.save_model = True  # Model kaydedilsin mi?
```

### LightGBM Parametreleri

```python
config.model.lightgbm_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}
```

## 🎯 Kullanım Senaryoları

### Senaryo 1: Tam Pipeline

```python
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

# Feature engineering
fe = FeatureEngineer()
df_features = fe.create_features(df_raw)

# Model training (tüm pipeline otomatik)
trainer = ModelTrainer()
trainer.train(df_features)
```

### Senaryo 2: Production (Model Yükleme)

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
trainer.load_model("models/model_lightgbm.pkl")

# Yeni veri için prediction
predictions = trainer.predict(df_new)
probabilities = trainer.predict_proba(df_new)
```

### Senaryo 3: Custom Evaluation

```python
from src.model_training import ModelTrainer, ModelEvaluator

trainer = ModelTrainer()
trainer.train(df_features)

# Custom evaluation
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(
    y_true=trainer.y_test.values,
    y_pred=trainer.predict(trainer.X_test),
    y_proba=trainer.predict_proba(trainer.X_test),
)

# Classification report
evaluator.print_classification_report(
    trainer.y_test.values,
    trainer.predict(trainer.X_test),
)
```

## 📈 Evaluation Metrikleri

### Classification Metrikleri

- `accuracy`: Doğruluk
- `precision`: Kesinlik
- `recall`: Duyarlılık
- `f1_score`: F1 skoru
- `roc_auc`: ROC AUC (binary classification için)
- `confusion_matrix`: Karışıklık matrisi

### Regression Metrikleri

- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `r2_score`: R² skoru

## 💾 Model Kaydetme ve Yükleme

### Kaydetme

```python
trainer = ModelTrainer()
trainer.train(df_features)

# Otomatik yol (models/model_lightgbm.pkl)
filepath = trainer.save_model()

# Özel yol
filepath = trainer.save_model("custom/path/model.pkl")
```

### Yükleme

```python
trainer = ModelTrainer()
trainer.load_model("models/model_lightgbm.pkl")

# Preprocessor otomatik yüklenir (scaler.pkl)
```

## 🔍 Feature Importance

```python
trainer = ModelTrainer()
trainer.train(df_features)

# Feature importance
importance = trainer.model.get_feature_importance()

# En önemli feature'lar
sorted_importance = sorted(
    importance.items(), key=lambda x: x[1], reverse=True
)
for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score:.4f}")
```

## 📝 Main Script Kullanımı

```bash
# Feature engineering
python -m src.feature_engineering

# Model training
python -m src.model_training
```

## ⚠️ Önemli Notlar

### 1. Data Leakage Önleme
- ✅ Train/test split otomatik yapılır
- ✅ Scaler sadece training data'da fit edilir
- ✅ Test data sadece transform edilir

### 2. Preprocessing Entegrasyonu
- ✅ Otomatik scaling (config'e göre)
- ✅ Preprocessor model ile birlikte kaydedilir
- ✅ Production'da otomatik yüklenir

### 3. Model Tipi
- ✅ Şu an için sadece LightGBM
- ✅ Classification ve Regression desteklenir
- ✅ Gelecekte diğer modeller eklenecek

## 🐛 Sorun Giderme

### Hata: "Model henüz eğitilmedi"
```python
# Çözüm: Önce train() çağır
trainer.train(df_features)
```

### Hata: "Processed data bulunamadı"
```python
# Çözüm: Önce feature engineering yap
fe = FeatureEngineer()
df_features = fe.create_features(df_raw)
```

### Model kaydedilemiyor
```python
# Çözüm: models/ klasörünün var olduğundan emin ol
import os
os.makedirs("models", exist_ok=True)
trainer.save_model()
```

## 📊 Beklenen Sonuçlar

### Pipeline Çıktıları

1. **Model**: Eğitilmiş model
2. **Preprocessor**: Fit edilmiş scaler
3. **Evaluation Metrics**: Test set metrikleri
4. **Feature Importance**: Feature önem skorları
5. **Saved Files**: Model ve preprocessor pickle dosyaları

### Log Çıktıları

```
🚀 MODEL TRAINING PIPELINE
📊 Veri hazırlanıyor...
✂️  Train/Test split yapılıyor...
🔧 Preprocessing uygulanıyor...
🚀 Model eğitimi başlatılıyor...
📊 MODEL EVALUATION
✅ MODEL TRAINING TAMAMLANDI!
```

