# Feature Scaling Implementation Plan

## 🎯 Genel Bakış

Bu doküman, feature scaling/normalization modülünün modüler ve modern bir şekilde implementasyon planını içerir.

## 📁 Klasör Yapısı

```
src/
├── preprocessing/
│   ├── __init__.py
│   ├── base.py              # BaseScaler abstract class
│   ├── scalers/
│   │   ├── __init__.py
│   │   ├── standard.py      # StandardScaler wrapper
│   │   ├── minmax.py        # MinMaxScaler wrapper
│   │   ├── robust.py        # RobustScaler wrapper
│   │   └── quantile.py      # QuantileTransformer wrapper
│   └── preprocessor.py      # Ana Preprocessor sınıfı
```

## 🏗️ Detaylı Tasarım

### 1. BaseScaler (Abstract Base Class)

**Dosya**: `src/preprocessing/base.py`

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class BaseScaler(ABC):
    """Base scaler interface - sklearn transformer pattern"""
    
    def __init__(self, exclude_columns: Optional[List[str]] = None):
        self.exclude_columns = exclude_columns or []
        self.feature_columns: Optional[List[str]] = None
        self.scaler = None
        
    @abstractmethod
    def _create_scaler(self):
        """Create the underlying sklearn scaler"""
        pass
    
    def fit(self, X: pd.DataFrame) -> 'BaseScaler':
        """Fit scaler on training data"""
        # Determine feature columns
        # Fit scaler
        return self
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        pass
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform"""
        return self.fit(X).transform(X)
    
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform (for predictions)"""
        pass
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to scale (exclude specified columns)"""
        return [col for col in df.columns if col not in self.exclude_columns]
```

### 2. Concrete Scalers

#### StandardScaler Wrapper
**Dosya**: `src/preprocessing/scalers/standard.py`

```python
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from ..base import BaseScaler
import pandas as pd

class StandardScaler(BaseScaler):
    """StandardScaler wrapper - Mean=0, Std=1"""
    
    def __init__(self, exclude_columns=None, with_mean=True, with_std=True):
        super().__init__(exclude_columns)
        self.with_mean = with_mean
        self.with_std = with_std
    
    def _create_scaler(self):
        return SklearnStandardScaler(with_mean=self.with_mean, with_std=self.with_std)
    
    # Implement fit, transform, inverse_transform
```

#### RobustScaler Wrapper (Önerilen)
**Dosya**: `src/preprocessing/scalers/robust.py`

```python
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from ..base import BaseScaler
import pandas as pd

class RobustScaler(BaseScaler):
    """RobustScaler wrapper - Median and IQR based (outlier resistant)"""
    
    def __init__(self, exclude_columns=None, quantile_range=(0.25, 0.75)):
        super().__init__(exclude_columns)
        self.quantile_range = quantile_range
    
    def _create_scaler(self):
        return SklearnRobustScaler(quantile_range=self.quantile_range)
    
    # Implement fit, transform, inverse_transform
```

#### MinMaxScaler Wrapper
**Dosya**: `src/preprocessing/scalers/minmax.py`

```python
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from ..base import BaseScaler
import pandas as pd

class MinMaxScaler(BaseScaler):
    """MinMaxScaler wrapper - Scale to 0-1 range"""
    
    def __init__(self, exclude_columns=None, feature_range=(0, 1)):
        super().__init__(exclude_columns)
        self.feature_range = feature_range
    
    def _create_scaler(self):
        return SklearnMinMaxScaler(feature_range=self.feature_range)
    
    # Implement fit, transform, inverse_transform
```

#### QuantileTransformer Wrapper
**Dosya**: `src/preprocessing/scalers/quantile.py`

```python
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from ..base import BaseScaler
import pandas as pd

class QuantileTransformer(BaseScaler):
    """QuantileTransformer wrapper - Uniform or normal distribution"""
    
    def __init__(self, exclude_columns=None, n_quantiles=1000, output_distribution='uniform'):
        super().__init__(exclude_columns)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
    
    def _create_scaler(self):
        return SklearnQuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution
        )
    
    # Implement fit, transform, inverse_transform
```

### 3. Preprocessor (Ana Sınıf)

**Dosya**: `src/preprocessing/preprocessor.py`

```python
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, List
from ..config import get_config
from ..logger import get_logger
from .scalers import RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer

logger = get_logger("MLProject.Preprocessing")

class Preprocessor:
    """Ana preprocessing sınıfı - Scaling işlemlerini yönetir"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.scaler = None
        self.feature_columns: Optional[List[str]] = None
        self.excluded_columns: Optional[List[str]] = None
        
    def _create_scaler(self):
        """Config'e göre scaler oluştur"""
        preprocess_config = self.config.preprocessing
        exclude_cols = preprocess_config.exclude_columns
        
        scaler_type = preprocess_config.scaler_type.lower()
        
        if scaler_type == "standard":
            return StandardScaler(
                exclude_columns=exclude_cols,
                with_mean=preprocess_config.standard_with_mean,
                with_std=preprocess_config.standard_with_std
            )
        elif scaler_type == "minmax":
            return MinMaxScaler(
                exclude_columns=exclude_cols,
                feature_range=preprocess_config.minmax_feature_range
            )
        elif scaler_type == "robust":
            return RobustScaler(
                exclude_columns=exclude_cols,
                quantile_range=preprocess_config.robust_quantile_range
            )
        elif scaler_type == "quantile":
            return QuantileTransformer(
                exclude_columns=exclude_cols,
                n_quantiles=preprocess_config.quantile_n_quantiles,
                output_distribution=preprocess_config.quantile_output_distribution
            )
        else:
            raise ValueError(f"Geçersiz scaler_type: {scaler_type}")
    
    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """Fit scaler on training data"""
        if not self.config.preprocessing.enable_scaling:
            logger.info("⚠️  Scaling devre dışı, atlanıyor...")
            return self
        
        logger.info("🔧 Scaler fit ediliyor...")
        self.scaler = self._create_scaler()
        self.scaler.fit(df)
        self.feature_columns = self.scaler.feature_columns
        self.excluded_columns = self.scaler.exclude_columns
        
        logger.info(f"✅ Scaler fit edildi. {len(self.feature_columns)} feature scale edilecek.")
        logger.info(f"   Excluded columns: {len(self.excluded_columns)}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data (train/test)"""
        if not self.config.preprocessing.enable_scaling:
            return df
        
        if self.scaler is None:
            raise ValueError("Scaler henüz fit edilmedi. Önce fit() çağrılmalı.")
        
        logger.info("🔄 Veri transform ediliyor...")
        df_scaled = self.scaler.transform(df)
        logger.info("✅ Veri transform edildi.")
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform"""
        return self.fit(df).transform(df)
    
    def save_scaler(self, filepath: Optional[str] = None) -> str:
        """Save scaler for production"""
        if self.scaler is None:
            raise ValueError("Scaler henüz fit edilmedi.")
        
        if filepath is None:
            models_dir = self.config.paths.models_dir
            filepath = models_dir / "scaler.pkl"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'excluded_columns': self.excluded_columns
            }, f)
        
        logger.info(f"💾 Scaler kaydedildi: {filepath}")
        return str(filepath)
    
    def load_scaler(self, filepath: str) -> 'Preprocessor':
        """Load scaler for production"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.excluded_columns = data['excluded_columns']
        
        logger.info(f"📂 Scaler yüklendi: {filepath}")
        return self
```

### 4. Config Entegrasyonu

**Dosya**: `src/config.py` (ekleme)

```python
@dataclass
class PreprocessingConfig:
    """Preprocessing konfigürasyonu"""
    
    # Scaling ayarları
    enable_scaling: bool = True
    scaler_type: str = "robust"  # 'standard', 'minmax', 'robust', 'quantile'
    
    # Hangi kolonlar scale edilmeyecek
    exclude_columns: List[str] = field(default_factory=lambda: [
        'datetime',
        'target',
        'future_return',
        'hour',
        'day_of_week',
        'month',
        'is_weekend',
    ])
    
    # RobustScaler parametreleri
    robust_quantile_range: Tuple[float, float] = (0.25, 0.75)
    
    # StandardScaler parametreleri
    standard_with_mean: bool = True
    standard_with_std: bool = True
    
    # MinMaxScaler parametreleri
    minmax_feature_range: Tuple[float, float] = (0, 1)
    
    # QuantileTransformer parametreleri
    quantile_n_quantiles: int = 1000
    quantile_output_distribution: str = 'uniform'  # 'uniform' or 'normal'

# Config sınıfına ekleme
@dataclass
class Config:
    # ... mevcut alanlar ...
    preprocessing: Optional[PreprocessingConfig] = None
    
    def __post_init__(self):
        # ... mevcut kod ...
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
```

## 🔄 Pipeline Entegrasyonu

### Senaryo 1: Feature Engineering Sonrası

```python
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor

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

### Senaryo 2: Train/Test Split Sonrası (Önerilen)

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

## 📝 Implementation Adımları

### Adım 1: Klasör Yapısını Oluştur
```bash
mkdir -p src/preprocessing/scalers
touch src/preprocessing/__init__.py
touch src/preprocessing/base.py
touch src/preprocessing/preprocessor.py
touch src/preprocessing/scalers/__init__.py
touch src/preprocessing/scalers/standard.py
touch src/preprocessing/scalers/minmax.py
touch src/preprocessing/scalers/robust.py
touch src/preprocessing/scalers/quantile.py
```

### Adım 2: BaseScaler'ı Implement Et
- Abstract base class
- Common functionality
- Interface definition

### Adım 3: Concrete Scalers'ı Implement Et
- StandardScaler
- RobustScaler (öncelikli)
- MinMaxScaler
- QuantileTransformer

### Adım 4: Preprocessor'ı Implement Et
- Ana sınıf
- Config entegrasyonu
- Save/load functionality

### Adım 5: Config'e PreprocessingConfig Ekle
- PreprocessingConfig dataclass
- Config sınıfına entegrasyon

### Adım 6: Unit Testler Yaz
- BaseScaler testleri
- Her scaler için testler
- Preprocessor testleri
- Integration testleri

### Adım 7: Dokümantasyon
- Docstring'ler
- README güncellemesi
- Usage examples

### Adım 8: Integration
- Feature engineering pipeline'ına entegre et
- Model training pipeline'ına entegre et (gelecekte)

## ✅ Test Senaryoları

### Test 1: Basic Functionality
```python
def test_basic_scaling():
    df = create_test_dataframe()
    preprocessor = Preprocessor()
    df_scaled = preprocessor.fit_transform(df)
    
    assert df_scaled.shape == df.shape
    assert 'target' not in preprocessor.scaler.feature_columns
```

### Test 2: Train/Test Split
```python
def test_train_test_split():
    X_train, X_test = train_test_split(X, test_size=0.2)
    
    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Test data should not affect scaler
    assert preprocessor.scaler is not None
```

### Test 3: Exclude Columns
```python
def test_exclude_columns():
    df = create_test_dataframe()
    preprocessor = Preprocessor()
    df_scaled = preprocessor.fit_transform(df)
    
    # Excluded columns should not be scaled
    assert df_scaled['target'].equals(df['target'])
    assert df_scaled['datetime'].equals(df['datetime'])
```

### Test 4: Save/Load
```python
def test_save_load():
    preprocessor = Preprocessor()
    preprocessor.fit(df_train)
    preprocessor.save_scaler('test_scaler.pkl')
    
    preprocessor2 = Preprocessor()
    preprocessor2.load_scaler('test_scaler.pkl')
    df_scaled = preprocessor2.transform(df_test)
    
    assert preprocessor2.scaler is not None
```

## 🎯 Öncelik Sırası

1. **Yüksek Öncelik**:
   - BaseScaler abstract class
   - RobustScaler (en önemli)
   - Preprocessor ana sınıf
   - Config entegrasyonu

2. **Orta Öncelik**:
   - StandardScaler, MinMaxScaler
   - Save/load functionality
   - Unit testler

3. **Düşük Öncelik**:
   - QuantileTransformer
   - Advanced features
   - Integration testleri

## 📊 Beklenen Sonuçlar

### Performans Metrikleri
- LightGBM için minimal etki (tree-based)
- Gelecek modeller için kritik fayda
- Feature importance daha anlamlı

### Kod Kalitesi
- Modüler yapı
- Test edilebilir
- Genişletilebilir
- Production-ready

### Kullanıcı Deneyimi
- Kolay kullanım
- Config ile kontrol
- Otomatik exclude columns
- Clear logging

