# Hyperparameter Tuning Kullanım Kılavuzu

## Genel Bakış

Hyperparameter tuning modülü, Optuna (Bayesian optimization), RandomizedSearchCV ve GridSearchCV desteği ile modüler bir yapı sunar.

## Özellikler

- ✅ **Optuna** (Bayesian optimization) - En gelişmiş ve hızlı
- ✅ **RandomizedSearchCV** - Hızlı random sampling
- ✅ **GridSearchCV** - Exhaustive search
- ✅ **Incremental Tuning** - İki aşamalı (coarse → fine)
- ✅ **Time Series Support** - TimeSeriesSplit ile CV
- ✅ **Pruning** - Optuna ile erken durdurma

## Hızlı Başlangıç

### 1. Config'de Tuning Aktif Et

```python
from src.config import get_config

config = get_config()

# Tuning'i aktif et
config.model.tuning.enable_tuning = True
config.model.tuning.tuning_method = "optuna"  # 'optuna', 'randomized', 'grid'
config.model.tuning.optuna_n_trials = 100
```

### 2. Parametre Space Tanımla

#### Optuna Formatı (Önerilen)

```python
config.model.tuning.param_grids = {
    "num_leaves": {
        "type": "int",
        "low": 15,
        "high": 127,
        "step": 1,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.01,
        "high": 0.2,
        "log": True,  # Log scale
    },
    "feature_fraction": {
        "type": "float",
        "low": 0.7,
        "high": 1.0,
    },
    "bagging_fraction": {
        "type": "float",
        "low": 0.7,
        "high": 1.0,
    },
    "min_child_samples": {
        "type": "int",
        "low": 10,
        "high": 50,
        "step": 5,
    },
}
```

#### RandomizedSearchCV/GridSearchCV Formatı

```python
config.model.tuning.param_grids = {
    "num_leaves": [31, 50, 100, 127],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [10, 20, 30, 50],
}
```

### 3. Model Training ile Kullan

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer(config=config)
trainer.train(df, target_column="target")
```

Tuning otomatik olarak yapılacak ve best parametreler modele uygulanacak.

## Detaylı Kullanım

### Manuel Tuning

```python
from src.model_training import ModelTrainer
from src.model_training.tuning import HyperparameterTuner
from src.model_training.models import LightGBMModel

# Model oluştur
model = LightGBMModel(config=config)

# Tuner oluştur
tuner = HyperparameterTuner(config=config)

# Parametre space
param_space = {
    "num_leaves": {
        "type": "int",
        "low": 15,
        "high": 127,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.01,
        "high": 0.2,
        "log": True,
    },
}

# Tuning yap
best_params = tuner.tune(
    model,
    X_train,
    y_train,
    param_space=param_space,
)

# Best parametreleri modele uygula
tuner.apply_best_params(model)

# Model'i eğit
model.fit(X_train, y_train)
```

### Incremental Tuning (İki Aşamalı)

```python
config.model.tuning.use_incremental = True
config.model.tuning.coarse_n_iter = 20  # Coarse aşama
config.model.tuning.fine_n_iter = 50    # Fine aşama

trainer = ModelTrainer(config=config)
trainer.train(df, target_column="target")
```

Veya manuel:

```python
tuner = HyperparameterTuner(config=config)

# Coarse space (geniş aralıklar)
coarse_space = {
    "num_leaves": {
        "type": "int",
        "low": 10,
        "high": 200,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.001,
        "high": 0.5,
        "log": True,
    },
}

# Fine space None - otomatik oluşturulacak
best_params = tuner.incremental_tune(
    model,
    X_train,
    y_train,
    coarse_space,
    fine_space=None,  # Otomatik oluşturulur
)
```

## Config Parametreleri

### TuningConfig

```python
@dataclass
class TuningConfig:
    # Tuning ayarları
    enable_tuning: bool = False
    tuning_method: str = "optuna"  # 'grid', 'randomized', 'optuna'

    # RandomizedSearchCV parametreleri
    randomized_n_iter: int = 50
    randomized_cv: int = 5

    # GridSearchCV parametreleri
    grid_cv: int = 5

    # Optuna parametreleri
    optuna_n_trials: int = 100
    optuna_timeout: Optional[int] = None  # Saniye
    study_name: Optional[str] = None
    direction: str = "maximize"  # 'maximize' or 'minimize'
    use_pruning: bool = True

    # Performans
    n_jobs: int = -1  # -1 = tüm core'lar
    scoring: str = "accuracy"

    # Incremental tuning
    use_incremental: bool = False
    coarse_n_iter: int = 20
    fine_n_iter: int = 50

    # Parametre grid'leri
    param_grids: Optional[Dict[str, Any]] = None
```

## Optuna Özellikleri

### Pruning

Optuna, umut vermeyen trial'ları erken durdurur:

```python
config.model.tuning.use_pruning = True
```

### Timeout

Maksimum süre belirle:

```python
config.model.tuning.optuna_timeout = 3600  # 1 saat
```

### Study Name

Optuna study'yi kaydetmek için:

```python
config.model.tuning.study_name = "lightgbm_tuning_v1"
```

## Scoring Metrikleri

```python
# Classification
config.model.tuning.scoring = "accuracy"
config.model.tuning.scoring = "f1"
config.model.tuning.scoring = "roc_auc"
config.model.tuning.scoring = "precision"
config.model.tuning.scoring = "recall"

# Regression
config.model.tuning.scoring = "r2"
config.model.tuning.scoring = "neg_mean_squared_error"
config.model.tuning.scoring = "neg_mean_absolute_error"
```

## Best Practices

1. **Optuna kullan** - En hızlı ve etkili
2. **Time series için TimeSeriesSplit** - Otomatik aktif
3. **Pruning aktif et** - Süreyi kısaltır
4. **Incremental tuning** - Büyük space'lerde kullan
5. **Timeout belirle** - Sonsuz çalışmayı önle

## Örnek: Tam Pipeline

```python
from src.config import get_config
from src.model_training import ModelTrainer

# Config yükle
config = get_config()

# Tuning ayarları
config.model.tuning.enable_tuning = True
config.model.tuning.tuning_method = "optuna"
config.model.tuning.optuna_n_trials = 100
config.model.tuning.use_pruning = True
config.model.tuning.optuna_timeout = 3600  # 1 saat

# Parametre space
config.model.tuning.param_grids = {
    "num_leaves": {
        "type": "int",
        "low": 15,
        "high": 127,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.01,
        "high": 0.2,
        "log": True,
    },
    "feature_fraction": {
        "type": "float",
        "low": 0.7,
        "high": 1.0,
    },
    "bagging_fraction": {
        "type": "float",
        "low": 0.7,
        "high": 1.0,
    },
    "min_child_samples": {
        "type": "int",
        "low": 10,
        "high": 50,
        "step": 5,
    },
}

# Model training
trainer = ModelTrainer(config=config)
trainer.train(df, target_column="target")

# Best parametreleri görüntüle
if trainer.tuner:
    print("Best params:", trainer.tuner.get_best_params())
    print("Best score:", trainer.tuner.get_best_score())
```

## Troubleshooting

### Optuna Import Hatası

```bash
pip install optuna>=3.0.0
```

### Tuning Çok Uzun Sürüyor

- `optuna_n_trials` azalt
- `use_pruning = True` yap
- `optuna_timeout` belirle
- `n_jobs` artır (ama Optuna için genelde 1)

### Best Score Düşük

- Parametre space'i genişlet
- `n_trials` artır
- Incremental tuning kullan

