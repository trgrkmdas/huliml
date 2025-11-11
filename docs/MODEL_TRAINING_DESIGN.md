# Model Training Modülü Tasarım Dokümanı

## 🎯 Genel Bakış

Modern, modüler ve mevcut yapıyla uyumlu model training modülü tasarımı.

## 📁 Klasör Yapısı

```
src/model_training/
├── __init__.py
├── base.py              # BaseModel abstract class
├── trainer.py           # ModelTrainer ana sınıfı (pipeline)
├── evaluator.py         # ModelEvaluator sınıfı
├── utils.py             # Yardımcı fonksiyonlar
└── models/
    ├── __init__.py
    ├── lightgbm.py      # LightGBM wrapper
    └── sklearn_models.py # Sklearn modelleri (gelecek)
```

## 🏗️ Mimari Tasarım

### 1. BaseModel (Abstract Base Class)

**Dosya**: `src/model_training/base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Base model interface"""
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities (for classification)"""
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """Get feature importance"""
        pass
```

### 2. LightGBM Model Wrapper

**Dosya**: `src/model_training/models/lightgbm.py`

```python
from lightgbm import LGBMClassifier, LGBMRegressor
from ..base import BaseModel

class LightGBMModel(BaseModel):
    """LightGBM model wrapper"""
    
    def __init__(self, params=None, task_type="classification"):
        # Initialize LightGBM model
        # Handle classification vs regression
```

### 3. ModelTrainer (Ana Pipeline Sınıfı)

**Dosya**: `src/model_training/trainer.py`

```python
class ModelTrainer:
    """Ana model training pipeline sınıfı"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
    
    def prepare_data(self, df):
        """X ve y'yi ayır, train/test split yap"""
        pass
    
    def train(self, df):
        """Full training pipeline"""
        # 1. Prepare data
        # 2. Train/test split
        # 3. Preprocessing (scaling)
        # 4. Model training
        # 5. Evaluation
        pass
    
    def save_model(self, filepath=None):
        """Model ve preprocessor'ı kaydet"""
        pass
    
    def load_model(self, filepath):
        """Model ve preprocessor'ı yükle"""
        pass
```

### 4. ModelEvaluator

**Dosya**: `src/model_training/evaluator.py`

```python
class ModelEvaluator:
    """Model evaluation sınıfı"""
    
    def evaluate(self, y_true, y_pred, y_proba=None):
        """Comprehensive evaluation"""
        # Classification metrics
        # Regression metrics
        # Feature importance
        pass
    
    def plot_confusion_matrix(self):
        """Confusion matrix visualization"""
        pass
    
    def plot_feature_importance(self):
        """Feature importance visualization"""
        pass
```

## 🔄 Pipeline Akışı

```
Feature Engineering
  ↓
Model Training Pipeline:
  1. Prepare Data (X, y separation)
  2. Train/Test Split
  3. Preprocessing (Scaling) ⭐
  4. Model Training
  5. Evaluation
  6. Save Model & Preprocessor
```

## ⚙️ Config Entegrasyonu

```python
@dataclass
class ModelConfig:
    model_type: str = "lightgbm"
    lightgbm_params: Dict[str, Any] = ...
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42
    use_preprocessing: bool = True  # Preprocessing kullanılsın mı?
    save_model: bool = True  # Model kaydedilsin mi?
```

## 📊 Özellikler

1. **Modüler Yapı**: Her model tipi ayrı modül
2. **Config-Based**: Tüm ayarlar config'den
3. **Preprocessing Entegrasyonu**: Otomatik scaling
4. **Train/Test Split**: Otomatik split
5. **Model Evaluation**: Comprehensive metrics
6. **Save/Load**: Model ve preprocessor birlikte
7. **Logging**: Tüm adımlar loglanır

## 🎯 Kullanım Senaryoları

### Senaryo 1: Basit Kullanım

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
trainer.train(df_features)
```

### Senaryo 2: Detaylı Kullanım

```python
trainer = ModelTrainer()
trainer.prepare_data(df_features)
trainer.train()
trainer.evaluate()
trainer.save_model()
```

### Senaryo 3: Production

```python
trainer = ModelTrainer()
trainer.load_model("models/model.pkl")
predictions = trainer.predict(df_new)
```

