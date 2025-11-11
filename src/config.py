"""
Proje genelinde kullanılan tüm konfigürasyon değerleri.
Hardcoded değerler buradan yönetilir.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import os
from pathlib import Path


@dataclass
class ExchangeConfig:
    """Exchange API konfigürasyonu"""

    name: str = "binance"
    enable_rate_limit: bool = True
    default_type: str = "spot"  # spot, future, delivery
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = False


@dataclass
class DataCollectionConfig:
    """Veri toplama konfigürasyonu"""

    # Varsayılan değerler
    default_symbol: str = "BTC/USDT"
    default_interval: str = "1d"
    default_limit: int = 1000
    default_days_back: int = 730  # 2 yıl

    # Rate limiting
    max_requests_per_second: int = 10
    request_delay: Optional[float] = None  # None ise otomatik hesaplanır

    # Error handling
    error_retry_delay: float = 5.0  # saniye

    # Progress reporting
    progress_print_threshold: int = 1000

    # Desteklenen interval'ler
    supported_intervals: Optional[Dict[str, str]] = None

    # Interval'lerin milisaniye karşılıkları
    interval_ms_map: Optional[Dict[str, int]] = None
    default_interval_ms: int = 24 * 60 * 60 * 1000  # 1 gün (varsayılan)

    # main() fonksiyonu için varsayılan değerler
    main_interval: str = "1h"
    main_start_date: str = "2024-01-01"
    main_end_date: Optional[str] = "2025-11-01"

    def __post_init__(self):
        if self.supported_intervals is None:
            self.supported_intervals = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
                "1w": "1w",
            }

        if self.interval_ms_map is None:
            self.interval_ms_map = {
                "1m": 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "30m": 30 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000,
                "1w": 7 * 24 * 60 * 60 * 1000,
            }

        if self.request_delay is None:
            self.request_delay = 1.0 / self.max_requests_per_second


@dataclass
class TechnicalIndicatorsConfig:
    """Teknik gösterge parametreleri"""

    # SMA (Simple Moving Average) periyotları
    sma_periods: Optional[List[int]] = None

    # EMA (Exponential Moving Average) periyotları
    ema_periods: Optional[List[int]] = None

    # RSI (Relative Strength Index) periyotları
    rsi_periods: Optional[List[int]] = None

    # MACD parametreleri
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands parametreleri
    bb_length: int = 20
    bb_std: float = 2.0

    # Stochastic Oscillator parametreleri
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth: int = 3

    # ATR (Average True Range) periyodu
    atr_period: int = 14

    # ADX (Average Directional Index) periyodu
    adx_period: int = 14

    # Volume SMA periyodu
    volume_sma_period: int = 20

    # Returns hesaplama periyotları
    returns_periods: Optional[List[int]] = None

    # Lag features (geçmiş değerler)
    lag_periods: Optional[List[int]] = None

    # Rolling window'lar
    rolling_windows: Optional[List[int]] = None

    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [7, 14, 21, 50, 200]

        if self.ema_periods is None:
            self.ema_periods = [12, 26]

        if self.rsi_periods is None:
            self.rsi_periods = [14, 7, 21]

        if self.returns_periods is None:
            self.returns_periods = [5, 10, 20]

        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10]

        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 21]


@dataclass
class PathsConfig:
    """Dosya yolu konfigürasyonu"""

    # Proje root dizini
    project_root: Optional[Path] = None

    # Veri dizinleri
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Model dizinleri
    models_dir: str = "models"

    # Sonuç dizinleri
    results_dir: str = "results"

    # Notebook dizinleri
    notebooks_dir: str = "notebooks"

    def __post_init__(self):
        if self.project_root is None:
            # Proje root'unu otomatik bul
            current_file = Path(__file__).resolve()
            # src/config.py -> MLProject/
            self.project_root = current_file.parent.parent

        # Tüm dizinleri Path objelerine çevir
        self.data_dir = self.project_root / self.data_dir
        self.raw_data_dir = self.project_root / self.raw_data_dir
        self.processed_data_dir = self.project_root / self.processed_data_dir
        self.models_dir = self.project_root / self.models_dir
        self.results_dir = self.project_root / self.results_dir
        self.notebooks_dir = self.project_root / self.notebooks_dir


@dataclass
class LoggingConfig:
    """Logging konfigürasyonu"""

    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering konfigürasyonu"""

    # Target variable parametreleri
    prediction_horizon: int = 1  # Kaç saat sonrasını tahmin edeceğiz
    target_threshold: float = 0.005  # %0.5 minimum return threshold
    target_type: str = "binary"  # 'binary', 'multi_class', 'regression'
    drop_hold_class: bool = False  # Hold sınıfını çıkar mı? (multi_class için)

    # Feature seçenekleri
    include_time_features: bool = True  # Zaman bazlı feature'lar
    include_candlestick_patterns: bool = False  # Mum formasyonları (gelecek)

    # Data cleaning
    drop_na: bool = True  # NaN'ları sil
    handle_outliers: bool = False  # Outlier handling (gelecek)

    # Feature selection
    feature_selection: bool = False  # Feature selection yapılacak mı (gelecek)

    # Preprocessing (scaling) - Opsiyonel, genelde train/test split'ten sonra kullanılmalı
    enable_scaling_in_pipeline: bool = (
        False  # Feature engineering pipeline'ında scaling yapılsın mı?
    )


@dataclass
class PreprocessingConfig:
    """Preprocessing konfigürasyonu"""

    # Scaling ayarları
    enable_scaling: bool = True
    scaler_type: str = "robust"  # 'standard', 'minmax', 'robust', 'quantile'

    # Hangi kolonlar scale edilmeyecek
    exclude_columns: List[str] = field(
        default_factory=lambda: [
            "datetime",
            "target",
            "future_return",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
        ]
    )

    # RobustScaler parametreleri
    robust_quantile_range: Tuple[float, float] = (0.25, 0.75)

    # StandardScaler parametreleri
    standard_with_mean: bool = True
    standard_with_std: bool = True

    # MinMaxScaler parametreleri
    minmax_feature_range: Tuple[float, float] = (0, 1)

    # QuantileTransformer parametreleri
    quantile_n_quantiles: int = 1000
    quantile_output_distribution: str = "uniform"  # 'uniform' or 'normal'


@dataclass
class TuningConfig:
    """Hyperparameter tuning konfigürasyonu"""

    # Tuning ayarları
    enable_tuning: bool = True  # Hyperparameter tuning aktif
    tuning_method: str = "optuna"  # 'grid', 'randomized', 'optuna'

    # RandomizedSearchCV parametreleri
    randomized_n_iter: int = 50  # Deneme sayısı
    randomized_cv: int = 5  # Cross-validation fold sayısı

    # GridSearchCV parametreleri
    grid_cv: int = 5

    # Optuna parametreleri
    optuna_n_trials: int = (
        50  # Bayesian optimization için deneme sayısı (hızlı test için 50)
    )
    optuna_timeout: Optional[int] = None  # Saniye cinsinden
    study_name: Optional[str] = None
    direction: str = "maximize"  # 'maximize' or 'minimize'
    use_pruning: bool = True  # Pruning kullanılsın mı?

    # Performans
    n_jobs: int = -1  # Parallel processing (-1 = tüm core'lar)
    scoring: str = "accuracy"  # Scoring metric

    # Incremental tuning
    use_incremental: bool = False  # İki aşamalı tuning
    coarse_n_iter: int = 20  # Coarse aşama iterasyon sayısı
    fine_n_iter: int = 50  # Fine aşama iterasyon sayısı
    fine_space_range_ratio: float = 0.2  # Fine tuning için ±%20 aralık

    # Cross-validation ayarları
    use_time_series_split: bool = (
        True  # Time series için TimeSeriesSplit kullanılsın mı?
    )

    # Grid search ayarları
    grid_combination_threshold: int = 1000  # Bu değerin üzerinde uyarı verilir

    # Verbose ayarları
    verbose_level: int = 1  # Genel verbose seviyesi (0=sessiz, 1=normal, 2=detaylı)

    # Parametre grid'leri (LightGBM için örnek)
    param_grids: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.param_grids is None:
            # LightGBM için varsayılan Optuna format
            self.param_grids = {
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


@dataclass
class FeatureSelectionConfig:
    """Feature selection konfigürasyonu"""

    # Feature selection ayarları
    enable_selection: bool = True  # Feature selection yapılsın mı?
    method: str = "univariate"  # 'univariate', 'rfe', 'variance', 'correlation'

    # Exclude columns (feature selection'dan hariç tutulacak)
    exclude_columns: Optional[List[str]] = None

    # Univariate selection parametreleri
    univariate_method: str = "k_best"  # 'k_best' veya 'percentile'
    k: int = 30  # SelectKBest için feature sayısı
    percentile: int = 10  # SelectPercentile için yüzde
    score_func: str = (
        "f_classif"  # 'f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression', 'chi2'
    )

    # RFE parametreleri
    n_features_to_select: Optional[int] = None  # None ise RFECV kullanılır
    rfe_step: int = 1  # Her iterasyonda çıkarılacak feature sayısı
    rfe_cv: Optional[int] = None  # Cross-validation folds (None ise RFE, int ise RFECV)

    # Variance threshold parametreleri
    variance_threshold: float = 0.0  # Düşük varyanslı feature'ları çıkar

    # Correlation parametreleri
    correlation_threshold: float = 0.95  # Yüksek korelasyonlu feature'ları çıkar
    correlation_method: str = "pearson"  # 'pearson', 'kendall', 'spearman'

    def __post_init__(self):
        if self.exclude_columns is None:
            self.exclude_columns = []


@dataclass
class AnalysisConfig:
    """Model analysis konfigürasyonu"""

    # Feature importance
    enable_feature_importance: bool = True  # Feature importance analizi yapılsın mı?
    feature_importance_top_n: int = 20  # Kaç feature gösterilecek
    feature_importance_save_plot: bool = True  # Plot kaydedilsin mi?
    feature_importance_save_csv: bool = True  # CSV kaydedilsin mi?
    feature_importance_plot_type: str = "horizontal"  # 'horizontal' veya 'vertical'

    # SHAP analysis
    enable_shap: bool = False  # SHAP analizi yapılsın mı?
    shap_explainer_type: str = "tree"  # 'tree', 'kernel', 'linear'
    shap_sample_size: int = 1000  # SHAP için örnekleme boyutu (None ise hepsi)
    shap_top_n: int = 20  # Summary plot'ta kaç feature gösterilecek
    shap_save_plots: bool = True  # SHAP plot'ları kaydedilsin mi?
    shap_save_values: bool = False  # SHAP değerleri kaydedilsin mi?
    shap_plot_types: List[str] = None  # ['summary', 'waterfall', 'dependence']

    def __post_init__(self):
        if self.shap_plot_types is None:
            self.shap_plot_types = ["summary"]


@dataclass
class ModelConfig:
    """Model eğitimi konfigürasyonu"""

    # Model tipi
    model_type: str = "lightgbm"

    # LightGBM parametreleri
    lightgbm_params: Optional[Dict[str, Any]] = None

    # Train/test split
    test_size: float = 0.2
    validation_size: float = 0.1

    # Random seed
    random_seed: int = 42

    # Early stopping
    early_stopping_rounds: Optional[int] = 50  # None ise early stopping kapalı
    early_stopping_verbose: bool = True  # Early stopping mesajlarını göster
    early_stopping_verbose_level: int = (
        100  # Early stopping verbose seviyesi (verbose=True ise)
    )

    # Preprocessing
    use_preprocessing: bool = True  # Preprocessing (scaling) kullanılsın mı?

    # Model saving
    save_model: bool = True  # Model kaydedilsin mi?

    # Cross validation / Walk-forward validation
    use_walk_forward: bool = False  # Walk-forward validation kullanılsın mı?
    walk_forward_type: str = "expanding"  # 'expanding' veya 'rolling'
    walk_forward_n_splits: int = 5  # Walk-forward fold sayısı
    walk_forward_test_size: Optional[int] = (
        None  # Her fold'ta test boyutu (None ise otomatik)
    )
    walk_forward_window_size: Optional[int] = (
        None  # Rolling window için pencere boyutu (None ise otomatik)
    )
    walk_forward_gap: int = 0  # Train-test arası gap (purged walk-forward için)
    walk_forward_window_multiplier: int = (
        5  # Rolling window için test_size'ın kaç katı olacak
    )
    walk_forward_window_percentage: float = (
        0.2  # Rolling window için toplam verinin yüzde kaçı olacak
    )

    # Hyperparameter tuning
    tuning: Optional["TuningConfig"] = None

    # Model analysis
    analysis: Optional["AnalysisConfig"] = None

    # Feature selection
    feature_selection: Optional["FeatureSelectionConfig"] = None

    def __post_init__(self):
        if self.lightgbm_params is None:
            self.lightgbm_params = {
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


@dataclass
class Config:
    """Ana konfigürasyon sınıfı - tüm alt konfigürasyonları içerir"""

    exchange: Optional[ExchangeConfig] = None
    data_collection: Optional[DataCollectionConfig] = None
    technical_indicators: Optional[TechnicalIndicatorsConfig] = None
    feature_engineering: Optional[FeatureEngineeringConfig] = None
    preprocessing: Optional[PreprocessingConfig] = None
    paths: Optional[PathsConfig] = None
    model: Optional[ModelConfig] = None
    logging: Optional[LoggingConfig] = None

    def __post_init__(self):
        if self.exchange is None:
            self.exchange = ExchangeConfig()

        if self.data_collection is None:
            self.data_collection = DataCollectionConfig()

        if self.technical_indicators is None:
            self.technical_indicators = TechnicalIndicatorsConfig()

        if self.feature_engineering is None:
            self.feature_engineering = FeatureEngineeringConfig()

        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()

        if self.paths is None:
            self.paths = PathsConfig()

        if self.model is None:
            self.model = ModelConfig()

        # ModelConfig içindeki tuning'i initialize et
        if self.model.tuning is None:
            self.model.tuning = TuningConfig()

        # ModelConfig içindeki analysis'i initialize et
        if self.model.analysis is None:
            self.model.analysis = AnalysisConfig()

        # ModelConfig içindeki feature_selection'ı initialize et
        if self.model.feature_selection is None:
            self.model.feature_selection = FeatureSelectionConfig()

        if self.logging is None:
            self.logging = LoggingConfig()

        # Mypy için assert'ler - __post_init__ sonrası None olamazlar
        assert self.exchange is not None
        assert self.data_collection is not None
        assert self.technical_indicators is not None
        assert self.feature_engineering is not None
        assert self.preprocessing is not None
        assert self.paths is not None
        assert self.model is not None
        assert self.logging is not None

        # Environment variables'dan değerleri yükle
        self._load_from_env()

    def _load_from_env(self):
        """Environment variables'dan konfigürasyon yükle"""
        # Exchange API keys
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if api_key:
            self.exchange.api_key = api_key
        if api_secret:
            self.exchange.api_secret = api_secret

        # Sandbox mode
        sandbox = os.getenv("BINANCE_SANDBOX", "false").lower() == "true"
        self.exchange.sandbox = sandbox


# Global config instance
config = Config()


# Kolay erişim için kısayollar
def get_config() -> Config:
    """Global config instance'ı döndürür"""
    return config


def reload_config():
    """Config'i yeniden yükle"""
    global config
    config = Config()
    return config
