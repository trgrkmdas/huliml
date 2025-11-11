# Feature Importance ve SHAP Analysis Kullanım Kılavuzu

## Genel Bakış

Model analysis modülü, feature importance ve SHAP değerleri ile model interpretability sağlar. Modüler yapıda tasarlanmıştır ve ModelTrainer pipeline'ına entegre edilmiştir.

## Özellikler

- ✅ **Feature Importance** - Model feature importance analizi ve görselleştirme
- ✅ **SHAP Analysis** - SHAP değerleri ile model interpretability
- ✅ **Otomatik Görselleştirme** - Plot'lar otomatik kaydedilir
- ✅ **CSV Export** - Feature importance CSV'ye kaydedilir
- ✅ **Modüler Yapı** - Bağımsız kullanılabilir

## Hızlı Başlangıç

### 1. Config'de Aktif Et

```python
from src.config import get_config

config = get_config()

# Feature importance aktif (varsayılan: True)
config.model.analysis.enable_feature_importance = True
config.model.analysis.feature_importance_top_n = 20

# SHAP analysis aktif et
config.model.analysis.enable_shap = True
config.model.analysis.shap_explainer_type = "tree"  # LightGBM için
config.model.analysis.shap_sample_size = 1000
```

### 2. Model Training ile Kullan

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer(config=config)
trainer.train(df, target_column="target")

# Analysis otomatik yapılacak ve sonuçlar kaydedilecek
```

## Feature Importance

### Otomatik Kullanım

Feature importance varsayılan olarak aktif. Training sonrası otomatik olarak:
- Top N feature importance loglanır
- Plot kaydedilir (`models/feature_importance.png`)
- CSV kaydedilir (`models/feature_importance.csv`)

### Manuel Kullanım

```python
from src.model_training.analysis import FeatureImportanceAnalyzer

# Analyzer oluştur
analyzer = FeatureImportanceAnalyzer(
    model=trainer.model,
    feature_names=list(trainer.X_train_scaled.columns),
)

# Importance al
importance_dict = analyzer.get_importance()

# Top features
top_features = analyzer.get_top_features(top_n=20)
print(top_features)

# Görselleştir
analyzer.plot_horizontal(top_n=20, save_path="importance.png")

# CSV kaydet
analyzer.save_to_csv("importance.csv")
```

### Config Parametreleri

```python
config.model.analysis.enable_feature_importance = True
config.model.analysis.feature_importance_top_n = 20
config.model.analysis.feature_importance_save_plot = True
config.model.analysis.feature_importance_save_csv = True
config.model.analysis.feature_importance_plot_type = "horizontal"  # veya "vertical"
```

## SHAP Analysis

### Gereksinimler

```bash
pip install shap>=0.43.0
```

### Otomatik Kullanım

```python
config.model.analysis.enable_shap = True
config.model.analysis.shap_explainer_type = "tree"  # LightGBM için
config.model.analysis.shap_sample_size = 1000
config.model.analysis.shap_top_n = 20
config.model.analysis.shap_plot_types = ["summary", "waterfall", "dependence"]
```

### Manuel Kullanım

```python
from src.model_training.analysis import SHAPAnalyzer

# Analyzer oluştur
shap_analyzer = SHAPAnalyzer(
    model=trainer.model,
    X=trainer.X_train_scaled.head(1000),  # Background data
    feature_names=list(trainer.X_train_scaled.columns),
)

# SHAP değerlerini hesapla
shap_values = shap_analyzer.calculate_shap_values(
    explainer_type="tree"
)

# Summary plot
shap_analyzer.plot_summary(
    top_n=20,
    plot_type="bar",
    save_path="shap_summary.png",
)

# Waterfall plot (tek örnek)
shap_analyzer.plot_waterfall(
    instance_idx=0,
    save_path="shap_waterfall.png",
)

# Dependence plot
shap_analyzer.plot_dependence(
    feature="feature_name",
    interaction_feature="interaction_feature",  # Opsiyonel
    save_path="shap_dependence.png",
)

# Feature importance from SHAP
shap_importance = shap_analyzer.get_feature_importance_from_shap()
print(shap_importance)

# SHAP değerlerini kaydet
shap_analyzer.save_shap_values("shap_values.npy")
```

### SHAP Explainer Tipleri

1. **TreeExplainer** (`"tree"`) - LightGBM, XGBoost için (hızlı)
2. **KernelExplainer** (`"kernel"`) - Herhangi bir model için (yavaş)
3. **LinearExplainer** (`"linear"`) - Linear modeller için

### Config Parametreleri

```python
config.model.analysis.enable_shap = True
config.model.analysis.shap_explainer_type = "tree"
config.model.analysis.shap_sample_size = 1000  # Background data boyutu
config.model.analysis.shap_top_n = 20
config.model.analysis.shap_save_plots = True
config.model.analysis.shap_save_values = False
config.model.analysis.shap_plot_types = ["summary"]  # veya ["summary", "waterfall", "dependence"]
```

## Çıktılar

### Feature Importance

- **Plot**: `models/feature_importance.png`
- **CSV**: `models/feature_importance.csv`
- **Log**: Top N feature importance listesi

### SHAP Analysis

- **Summary Plot**: `models/shap_plots/shap_summary.png`
- **Waterfall Plot**: `models/shap_plots/shap_waterfall.png`
- **Dependence Plots**: `models/shap_plots/shap_dependence_{feature}.png`
- **SHAP Values**: `models/shap_values.npy` (opsiyonel)

## Örnek: Tam Pipeline

```python
from src.config import get_config
from src.model_training import ModelTrainer

# Config yükle
config = get_config()

# Analysis ayarları
config.model.analysis.enable_feature_importance = True
config.model.analysis.feature_importance_top_n = 20
config.model.analysis.feature_importance_save_plot = True
config.model.analysis.feature_importance_save_csv = True

config.model.analysis.enable_shap = True
config.model.analysis.shap_explainer_type = "tree"
config.model.analysis.shap_sample_size = 1000
config.model.analysis.shap_top_n = 20
config.model.analysis.shap_plot_types = ["summary", "waterfall"]

# Model training
trainer = ModelTrainer(config=config)
trainer.train(df, target_column="target")

# Analysis sonuçlarına erişim
if trainer.feature_importance_analyzer:
    top_features = trainer.feature_importance_analyzer.get_top_features(top_n=10)
    print(top_features)

if trainer.shap_analyzer:
    shap_importance = trainer.shap_analyzer.get_feature_importance_from_shap()
    print(shap_importance)
```

## Best Practices

1. **Feature Importance** - Her zaman aktif tutun, hızlı ve bilgilendirici
2. **SHAP Analysis** - Büyük veri setlerinde örnekleme kullanın (`shap_sample_size`)
3. **TreeExplainer** - LightGBM için en hızlı ve doğru seçenek
4. **Plot Kaydetme** - Production'da `show=False` kullanın
5. **SHAP Values** - Gerekirse kaydedin, tekrar hesaplamadan kullanın

## Troubleshooting

### SHAP Import Hatası

```bash
pip install shap>=0.43.0
```

### SHAP Çok Yavaş

- `shap_sample_size` azaltın (örn: 500)
- `shap_explainer_type="tree"` kullanın (LightGBM için)
- Sadece `summary` plot kullanın

### Feature Importance Plot Görünmüyor

- `feature_importance_save_plot = True` kontrol edin
- `models/` klasörünün yazılabilir olduğundan emin olun

### SHAP Background Data Çok Büyük

- `shap_sample_size` parametresini kullanın
- Veya manuel olarak background data'yı örnekleyin

## API Referansı

### FeatureImportanceAnalyzer

- `get_importance(importance_type="gain")` - Feature importance dict
- `get_top_features(top_n=20)` - Top N features DataFrame
- `plot_importance(top_n, figsize, save_path, show)` - Dikey bar chart
- `plot_horizontal(top_n, figsize, save_path, show)` - Yatay bar chart
- `save_to_csv(file_path, top_n)` - CSV'ye kaydet
- `get_summary()` - Özet istatistikler

### SHAPAnalyzer

- `calculate_shap_values(X_explain, explainer_type, max_evals)` - SHAP değerlerini hesapla
- `plot_summary(top_n, plot_type, show, save_path)` - Summary plot
- `plot_waterfall(instance_idx, X_explain, show, save_path)` - Waterfall plot
- `plot_dependence(feature, interaction_feature, show, save_path)` - Dependence plot
- `get_feature_importance_from_shap()` - SHAP'ten feature importance
- `save_shap_values(file_path)` - SHAP değerlerini kaydet

