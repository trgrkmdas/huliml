# Hyperparameter Tuning Mimari Tasarımı

## 🎯 Genel Bakış

Modern, modüler ve yüksek performanslı hyperparameter tuning sistemi.

## 📁 Klasör Yapısı

```
src/model_training/
├── tuning/
│   ├── __init__.py
│   ├── base.py              # BaseTuner abstract class
│   ├── grid_search.py       # GridSearchCV wrapper
│   ├── randomized_search.py # RandomizedSearchCV wrapper
│   ├── optuna_tuner.py      # Optuna wrapper (opsiyonel)
│   └── tuner.py             # HyperparameterTuner ana sınıfı
```

## 🏗️ Mimari Tasarım

### 1. BaseTuner (Abstract Base Class)

**Dosya**: `src/model_training/tuning/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTuner(ABC):
    """Base tuner interface"""
    
    @abstractmethod
    def tune(self, model, X, y, param_grid, **kwargs):
        """Perform hyperparameter tuning"""
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters"""
        pass
    
    @abstractmethod
    def get_best_score(self) -> float:
        """Get best score"""
        pass
```

### 2. RandomizedSearchTuner (Öncelikli - Hızlı)

**Dosya**: `src/model_training/tuning/randomized_search.py`

```python
from sklearn.model_selection import RandomizedSearchCV
from .base import BaseTuner

class RandomizedSearchTuner(BaseTuner):
    """RandomizedSearchCV wrapper - Hızlı ve etkili"""
    
    def __init__(self, n_iter=50, cv=5, n_jobs=-1, scoring='accuracy'):
        # RandomizedSearchCV ile tuning
        # Parallel processing desteği
        # Caching desteği
```

### 3. GridSearchTuner

**Dosya**: `src/model_training/tuning/grid_search.py`

```python
from sklearn.model_selection import GridSearchCV
from .base import BaseTuner

class GridSearchTuner(BaseTuner):
    """GridSearchCV wrapper - Exhaustive search"""
    
    def __init__(self, cv=5, n_jobs=-1, scoring='accuracy'):
        # GridSearchCV ile tuning
        # Küçük parametre space'leri için
```

### 4. OptunaTuner (Opsiyonel - En İyi)

**Dosya**: `src/model_training/tuning/optuna_tuner.py`

```python
import optuna
from .base import BaseTuner

class OptunaTuner(BaseTuner):
    """Optuna wrapper - Bayesian optimization"""
    
    def __init__(self, n_trials=100, timeout=None):
        # Optuna ile tuning
        # En gelişmiş ve hızlı
        # Ek paket gerektirir: optuna
```

### 5. HyperparameterTuner (Ana Sınıf)

**Dosya**: `src/model_training/tuning/tuner.py`

```python
class HyperparameterTuner:
    """Ana hyperparameter tuning sınıfı"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.tuner = None  # BaseTuner instance
        self.best_params = None
        self.best_score = None
        self.tuning_history = []
    
    def tune(self, model, X, y, param_grid, **kwargs):
        """Hyperparameter tuning yap"""
        # Tuning metodunu seç (config'den)
        # Tuning yap
        # Best parametreleri kaydet
        # History kaydet
    
    def get_best_params(self) -> Dict[str, Any]:
        """Best parametreleri döndür"""
    
    def incremental_tune(self, model, X, y, coarse_grid, fine_grid):
        """İki aşamalı tuning (coarse → fine)"""
        # 1. Coarse search (geniş aralıklar)
        # 2. Fine search (dar aralıklar, best çevresinde)
```

## ⚡ Performans Optimizasyonları

### 1. Parallel Processing

```python
# n_jobs=-1: Tüm CPU core'ları kullan
tuner = RandomizedSearchTuner(n_jobs=-1)
```

### 2. Caching Mekanizması

```python
class TuningCache:
    """Tuning sonuçlarını cache'le"""
    
    def __init__(self):
        self.cache = {}  # {param_hash: score}
    
    def get(self, params):
        """Cache'den al"""
        param_hash = self._hash_params(params)
        return self.cache.get(param_hash)
    
    def set(self, params, score):
        """Cache'e kaydet"""
        param_hash = self._hash_params(params)
        self.cache[param_hash] = score
```

### 3. Early Stopping Entegrasyonu

```python
# Her tuning denemesinde early stopping kullan
# Gereksiz iterasyonları önle
model.fit(X, y, early_stopping_rounds=50, ...)
```

### 4. Incremental Tuning

```python
# İki aşamalı tuning
# 1. Coarse: Geniş aralıklar, az iterasyon
coarse_grid = {
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
}

# 2. Fine: Dar aralıklar, çok iterasyon (best çevresinde)
fine_grid = {
    'num_leaves': [best_num_leaves-10, best_num_leaves, best_num_leaves+10],
    'learning_rate': [best_lr-0.01, best_lr, best_lr+0.01],
}
```

### 5. Time-Based Pruning (Optuna)

```python
# Zaman limiti ile tuning
study.optimize(objective, timeout=3600)  # 1 saat
```

## 🔄 Walk-Forward Validation Entegrasyonu

### Yaklaşım 1: Tüm Fold'lar için Ortak Tuning (Hızlı)

```python
# Önce tüm fold'lar için ortak tuning
best_params = tuner.tune(model, X_train, y_train, param_grid)

# Sonra best parametrelerle walk-forward validation
trainer.walk_forward_validation()
```

### Yaklaşım 2: Her Fold için Ayrı Tuning (Daha Doğru)

```python
# Her fold için ayrı tuning (yavaş ama daha doğru)
for fold in walk_forward_splits:
    best_params_fold = tuner.tune(model, X_train_fold, y_train_fold, param_grid)
```

### Önerilen: Hibrit Yaklaşım

```python
# 1. Coarse tuning (tüm fold'lar için ortak, hızlı)
coarse_best = tuner.tune(model, X_train, y_train, coarse_grid)

# 2. Fine tuning (best çevresinde)
fine_best = tuner.incremental_tune(model, X_train, y_train, coarse_best, fine_grid)

# 3. Walk-forward validation (best parametrelerle)
trainer.walk_forward_validation()
```

## ⚙️ Config Entegrasyonu

```python
@dataclass
class TuningConfig:
    """Hyperparameter tuning konfigürasyonu"""
    
    # Tuning ayarları
    enable_tuning: bool = False
    tuning_method: str = "randomized"  # 'grid', 'randomized', 'optuna'
    
    # RandomizedSearchCV parametreleri
    randomized_n_iter: int = 50  # Deneme sayısı
    randomized_cv: int = 5  # Cross-validation fold sayısı
    
    # GridSearchCV parametreleri
    grid_cv: int = 5
    
    # Optuna parametreleri
    optuna_n_trials: int = 100
    optuna_timeout: Optional[int] = None  # Saniye cinsinden
    
    # Performans
    n_jobs: int = -1  # Parallel processing (-1 = tüm core'lar)
    use_cache: bool = True  # Caching aktif mi?
    
    # Incremental tuning
    use_incremental: bool = True  # İki aşamalı tuning
    coarse_n_iter: int = 20  # Coarse aşama iterasyon sayısı
    fine_n_iter: int = 50  # Fine aşama iterasyon sayısı
    
    # Parametre grid'leri (LightGBM için örnek)
    param_grids: Optional[Dict[str, List[Any]]] = None
    
    def __post_init__(self):
        if self.param_grids is None:
            # LightGBM için varsayılan grid
            self.param_grids = {
                'num_leaves': [15, 31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'min_child_samples': [10, 20, 30],
            }
```

## 🚀 Kullanım Senaryoları

### Senaryo 1: Basit Tuning (RandomizedSearch)

```python
from src.model_training import ModelTrainer
from src.model_training.tuning import HyperparameterTuner

trainer = ModelTrainer()
trainer.prepare_data(df_features)
trainer.split_data(X, y)
trainer.apply_preprocessing()

# Tuning
tuner = HyperparameterTuner()
best_params = tuner.tune(
    trainer.model,
    trainer.X_train_scaled,
    trainer.y_train,
    param_grid=config.model.tuning.param_grids
)

# Best parametrelerle model eğit
trainer.model.set_params(**best_params)
trainer.train_model()
```

### Senaryo 2: Incremental Tuning

```python
# Coarse tuning
coarse_grid = {
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
}

coarse_best = tuner.tune(model, X, y, coarse_grid, n_iter=20)

# Fine tuning (best çevresinde)
fine_grid = {
    'num_leaves': [coarse_best['num_leaves']-10, 
                   coarse_best['num_leaves'], 
                   coarse_best['num_leaves']+10],
    'learning_rate': [coarse_best['learning_rate']-0.01,
                     coarse_best['learning_rate'],
                     coarse_best['learning_rate']+0.01],
}

fine_best = tuner.tune(model, X, y, fine_grid, n_iter=50)
```

### Senaryo 3: Walk-Forward ile Tuning

```python
# 1. Tuning (tüm train set üzerinde)
best_params = tuner.tune(model, X_train, y_train, param_grid)

# 2. Best parametrelerle walk-forward validation
trainer.model.set_params(**best_params)
results = trainer.walk_forward_validation()
```

## 📊 Performans Metrikleri

```python
tuning_results = {
    'best_params': {...},
    'best_score': 0.85,
    'tuning_time': 120.5,  # saniye
    'n_trials': 50,
    'improvement': 0.05,  # Baseline'a göre iyileşme
    'tuning_history': [...],  # Her deneme için skor
}
```

## ✅ Sonuç

**Mimari Özellikleri:**
- ✅ Modüler (BaseTuner, concrete implementasyonlar)
- ✅ Yüksek performanslı (parallel, caching, incremental)
- ✅ Mevcut sistemle entegre (ModelTrainer, Config)
- ✅ Walk-forward validation uyumlu
- ✅ Farklı tuning stratejileri (Grid, Randomized, Optuna)

**Performans Optimizasyonları:**
- ✅ Parallel processing (n_jobs=-1)
- ✅ Caching mekanizması
- ✅ Early stopping entegrasyonu
- ✅ Incremental tuning
- ✅ Time-based pruning (Optuna)

