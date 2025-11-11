"""
Model training ana sınıfı - Pipeline yönetimi
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..config import get_config
from ..logger import get_logger
from ..preprocessing import Preprocessor
from .models.lightgbm import LightGBMModel
from .base import BaseModel
from .evaluator import ModelEvaluator
from .tuning import HyperparameterTuner
from .analysis import (
    FeatureImportanceAnalyzer,
    SHAPAnalyzer,
)  # Deprecated metodlar için gerekli
from .constants import (
    LOG_SEPARATOR,
    MODEL_TYPE_LIGHTGBM,
    TASK_TYPE_REGRESSION,
    TASK_TYPE_CLASSIFICATION,
    COLUMN_TARGET,
)
from .data_preparator import DataPreparator
from .walk_forward_validator import WalkForwardValidator
from .model_analyzer import ModelAnalyzer
from .model_persistence import ModelPersistence
from .feature_selection import FeatureSelector

logger = get_logger("MLProject.ModelTraining")


class ModelTrainer:
    """Ana model training pipeline sınıfı"""

    def __init__(self, config=None):
        """
        Args:
            config: Config objesi (None ise get_config() kullanılır)
        """
        self.config = config or get_config()

        # Config değerlerini instance variable'lara kaydet (config erişimini azaltmak için)
        self._model_config = self.config.model
        self._fe_config = self.config.feature_engineering
        self._preprocessing_config = self.config.preprocessing
        self._paths_config = self.config.paths
        self._tuning_config = self._model_config.tuning
        self._analysis_config = self._model_config.analysis
        self._feature_selection_config = self._model_config.feature_selection

        # Helper sınıfları oluştur
        self._data_preparator = DataPreparator(
            self._preprocessing_config, self._fe_config, self._model_config
        )
        self._walk_forward_validator = WalkForwardValidator(
            self._model_config, self._fe_config, self.config
        )
        self._model_analyzer = ModelAnalyzer(
            self._analysis_config, self._model_config, self._paths_config
        )
        self._model_persistence = ModelPersistence(
            self._paths_config, self._model_config
        )

        # Model ve components
        self.model: Optional[BaseModel] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.feature_columns: Optional[list] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.tuner: Optional[HyperparameterTuner] = None
        self.feature_selector: Optional[FeatureSelector] = None
        self.feature_importance_analyzer: Optional[FeatureImportanceAnalyzer] = None
        self.shap_analyzer: Optional[SHAPAnalyzer] = None

        # Data splits
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Scaled data
        self.X_train_scaled: Optional[pd.DataFrame] = None
        self.X_test_scaled: Optional[pd.DataFrame] = None

    def _create_model(self) -> BaseModel:
        """Config'e göre model oluştur"""
        model_type = self._model_config.model_type.lower()

        if model_type == MODEL_TYPE_LIGHTGBM:
            # Task type'ı belirle
            task_type = (
                TASK_TYPE_REGRESSION
                if self._fe_config.target_type == TASK_TYPE_REGRESSION
                else TASK_TYPE_CLASSIFICATION
            )

            params = self._model_config.lightgbm_params or {}
            return LightGBMModel(params=params, task_type=task_type)
        else:
            raise ValueError(f"Geçersiz model_type: {model_type}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = COLUMN_TARGET,
        exclude_columns: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        X ve y'yi ayır.

        Args:
            df: DataFrame
            target_column: Target column adı
            exclude_columns: Exclude edilecek kolonlar

        Returns:
            Tuple of (X, y)
        """
        X, y, feature_columns = self._data_preparator.prepare_data(
            df, target_column, exclude_columns
        )
        self.feature_columns = feature_columns
        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Train/test split yap.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test size ratio
            random_state: Random seed
        """
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self._data_preparator.split_data(X, y, test_size, random_state)

    def apply_preprocessing(self) -> None:
        """Preprocessing (scaling) uygula"""
        if not self._model_config.use_preprocessing:
            logger.info("⚠️  Preprocessing devre dışı, atlanıyor...")
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
            return

        logger.info("🔧 Preprocessing uygulanıyor...")

        self.preprocessor = Preprocessor(config=self.config)
        self.X_train_scaled = self.preprocessor.fit_transform(self.X_train)
        self.X_test_scaled = self.preprocessor.transform(self.X_test)

        logger.info("✅ Preprocessing tamamlandı")

    def apply_feature_selection(self) -> None:
        """Feature selection uygula"""
        if self.X_train_scaled is None or self.y_train is None:
            raise ValueError(
                "Feature selection için önce preprocessing yapılmalı. "
                "apply_preprocessing() çağrılmalı."
            )

        logger.info("\n" + LOG_SEPARATOR)
        logger.info("🔍 FEATURE SELECTION")
        logger.info(LOG_SEPARATOR)

        # Task type belirle
        task_type = (
            TASK_TYPE_REGRESSION
            if self._fe_config.target_type == TASK_TYPE_REGRESSION
            else TASK_TYPE_CLASSIFICATION
        )

        # Feature selector oluştur
        self.feature_selector = FeatureSelector(config=self.config)

        # Feature selection yap
        (
            self.X_train_scaled,
            self.X_test_scaled,
            selected_features,
        ) = self.feature_selector.select_features(
            self.X_train_scaled,
            self.y_train,
            self.X_test_scaled,
            task_type=task_type,
        )

        # Feature columns'ı güncelle
        self.feature_columns = selected_features

        logger.info(LOG_SEPARATOR)
        logger.info("✅ FEATURE SELECTION TAMAMLANDI!")
        logger.info(LOG_SEPARATOR + "\n")

    def train_model(self, **kwargs) -> None:
        """
        Model'i eğit.

        Args:
            **kwargs: Model fit için ek argümanlar
        """
        logger.info("🚀 Model eğitimi başlatılıyor...")

        if self.X_train_scaled is None:
            raise ValueError(
                "Veri henüz hazırlanmadı. Önce prepare_data() ve split_data() çağrılmalı."
            )

        self.model = self._create_model()

        # Validation set hazırla (opsiyonel)
        eval_set = None

        if self._model_config.validation_size > 0:
            val_size = self._model_config.validation_size
            val_split = int(len(self.X_train_scaled) * (1 - val_size))
            eval_set = [
                (
                    self.X_train_scaled.iloc[val_split:],
                    self.y_train.iloc[val_split:],
                )
            ]
            kwargs["eval_set"] = eval_set
            kwargs["eval_names"] = ["validation"]

            # Early stopping (validation set varsa)
            if self._model_config.early_stopping_rounds is not None:
                # LightGBM 4.0+ için callbacks kullan
                from lightgbm import early_stopping, log_evaluation

                callbacks = kwargs.get("callbacks", [])
                callbacks.append(
                    early_stopping(
                        stopping_rounds=self._model_config.early_stopping_rounds,
                    )
                )

                # Verbose için log_evaluation callback'i kullan
                if self._model_config.early_stopping_verbose:
                    verbose_level = self._model_config.early_stopping_verbose_level
                    callbacks.append(log_evaluation(period=verbose_level))

                kwargs["callbacks"] = callbacks
                # verbose parametresini kwargs'tan çıkar (artık callback kullanıyoruz)
                kwargs.pop("verbose", None)

                logger.info(
                    f"⏹️  Early stopping aktif: {self._model_config.early_stopping_rounds} round"
                )
        elif self._model_config.early_stopping_rounds is not None:
            # Early stopping istendi ama validation set yok
            logger.warning(
                "⚠️  Early stopping için validation set gerekli ama validation_size=0. "
                "Early stopping devre dışı bırakıldı. "
                "Early stopping kullanmak için validation_size > 0 ayarlayın."
            )

        self.model.fit(self.X_train_scaled, self.y_train, **kwargs)

        logger.info("✅ Model eğitimi tamamlandı")

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = "target",
        exclude_columns: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Full training pipeline.

        Args:
            df: DataFrame (feature engineering sonrası)
            target_column: Target column adı
            exclude_columns: Exclude edilecek kolonlar
            **kwargs: Model fit için ek argümanlar
        """
        logger.info(LOG_SEPARATOR)
        logger.info("🚀 MODEL TRAINING PIPELINE")
        logger.info(LOG_SEPARATOR)

        # 1. Prepare data
        X, y = self.prepare_data(df, target_column, exclude_columns)

        # 2. Train/test split
        self.split_data(X, y)

        # 3. Preprocessing
        self.apply_preprocessing()

        # 4. Feature selection (opsiyonel)
        if self._feature_selection_config.enable_selection:
            self.apply_feature_selection()

        # 5. Hyperparameter tuning (opsiyonel)
        if self._tuning_config.enable_tuning:
            self.perform_tuning()

        # 6. Model training
        self.train_model(**kwargs)

        logger.info(LOG_SEPARATOR)
        logger.info("✅ MODEL TRAINING TAMAMLANDI!")
        logger.info(LOG_SEPARATOR)

        # 7. Walk-forward validation (opsiyonel)
        if self._model_config.use_walk_forward:
            self.walk_forward_validation()

        # 8. Evaluation (opsiyonel)
        if self._model_config.save_model:
            self.evaluate()

        # 9. Model analysis (opsiyonel)
        self.perform_analysis()

    def perform_tuning(self) -> None:
        """
        Hyperparameter tuning yap.
        """
        logger.info("\n" + LOG_SEPARATOR)
        logger.info("🔍 HYPERPARAMETER TUNING")
        logger.info(LOG_SEPARATOR)

        if self.X_train_scaled is None or self.y_train is None:
            raise ValueError(
                "Tuning için önce preprocessing yapılmalı. "
                "apply_preprocessing() çağrılmalı."
            )

        # Model oluştur
        if self.model is None:
            self.model = self._create_model()

        # Tuner oluştur
        self.tuner = HyperparameterTuner(config=self.config)

        # Tuning yap
        if self._tuning_config.use_incremental:
            # İki aşamalı tuning
            # Coarse space oluştur (geniş aralıklar)
            coarse_space = self._tuning_config.param_grids.copy()

            # Fine space None - otomatik oluşturulacak
            self.tuner.incremental_tune(
                self.model,
                self.X_train_scaled,
                np.asarray(self.y_train.values),  # Convert Series to ndarray
                coarse_space,
                fine_space=None,
            )
        else:
            # Normal tuning
            self.tuner.tune(
                self.model,
                self.X_train_scaled,
                np.asarray(self.y_train.values),  # Convert Series to ndarray
                param_space=self._tuning_config.param_grids,
            )

        # Best parametreleri modele uygula
        self.tuner.apply_best_params(self.model)

        logger.info(LOG_SEPARATOR)
        logger.info("✅ HYPERPARAMETER TUNING TAMAMLANDI!")
        logger.info(LOG_SEPARATOR + "\n")

    def perform_analysis(self) -> None:
        """
        Model analysis yap (feature importance ve SHAP).
        """
        if (
            not self._analysis_config.enable_feature_importance
            and not self._analysis_config.enable_shap
        ):
            return

        logger.info("\n" + LOG_SEPARATOR)
        logger.info("📊 MODEL ANALYSIS")
        logger.info(LOG_SEPARATOR)

        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")

        if self.X_train_scaled is None:
            raise ValueError("Preprocessing yapılmamış.")

        # Feature importance
        if self._analysis_config.enable_feature_importance:
            self._model_analyzer.analyze_feature_importance(
                self.model, self.X_train_scaled
            )
            self.feature_importance_analyzer = (
                self._model_analyzer.feature_importance_analyzer
            )

        # SHAP analysis
        if self._analysis_config.enable_shap:
            self._model_analyzer.analyze_shap(self.model, self.X_train_scaled)
            self.shap_analyzer = self._model_analyzer.shap_analyzer

        logger.info(LOG_SEPARATOR)
        logger.info("✅ MODEL ANALYSIS TAMAMLANDI!")
        logger.info(LOG_SEPARATOR + "\n")

    def analyze_feature_importance(self) -> None:
        """
        Feature importance analizi yap.
        (Deprecated: perform_analysis() kullanın)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        if self.X_train_scaled is None:
            raise ValueError("Preprocessing yapılmamış.")
        self._model_analyzer.analyze_feature_importance(self.model, self.X_train_scaled)
        self.feature_importance_analyzer = (
            self._model_analyzer.feature_importance_analyzer
        )

    def analyze_shap(self) -> None:
        """
        SHAP analizi yap.
        (Deprecated: perform_analysis() kullanın)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        if self.X_train_scaled is None:
            raise ValueError("Preprocessing yapılmamış.")
        self._model_analyzer.analyze_shap(self.model, self.X_train_scaled)
        self.shap_analyzer = self._model_analyzer.shap_analyzer

    def evaluate(self) -> Dict[str, float]:
        """
        Model'i değerlendir.

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")

        logger.info("\n" + LOG_SEPARATOR)
        logger.info("📊 MODEL EVALUATION")
        logger.info(LOG_SEPARATOR)

        # Test predictions (zaten preprocessing ve feature selection yapılmış veriyi kullan)
        if self.X_test_scaled is None:
            raise ValueError("Test verisi henüz hazırlanmadı.")

        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = (
            self.model.predict_proba(self.X_test_scaled)
            if hasattr(self.model, "predict_proba")
            else None
        )

        # Evaluation
        self.evaluator = ModelEvaluator(config=self.config)
        # Convert to numpy array for type safety
        y_test_array = np.asarray(self.y_test.values)
        metrics = self.evaluator.evaluate(
            y_test_array,
            y_pred,
            y_proba,
        )

        # Classification report (classification için)
        if self._fe_config.target_type != TASK_TYPE_REGRESSION:
            self.evaluator.print_classification_report(y_test_array, y_pred)

        logger.info(LOG_SEPARATOR)
        logger.info("✅ EVALUATION TAMAMLANDI!")
        logger.info(LOG_SEPARATOR)

        return metrics

    def walk_forward_validation(
        self,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: Optional[int] = None,
        window_type: Optional[str] = None,
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Walk-forward validation (Time series için).

        Args:
            n_splits: Fold sayısı (None ise config'den alınır)
            test_size: Her fold'ta test boyutu (None ise otomatik)
            gap: Train-test arası gap (purged walk-forward için)
            window_type: 'expanding' veya 'rolling' (None ise config'den alınır)
            window_size: Rolling window için pencere boyutu (None ise otomatik)

        Returns:
            Dictionary of validation results
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Veri henüz hazırlanmadı. Önce prepare_data() ve split_data() çağrılmalı."
            )

        return self._walk_forward_validator.validate(  # type: ignore[no-any-return]
            self.X_train,
            self.y_train,
            model_factory=self._create_model,
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            window_type=window_type,
            window_size=window_size,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediction yap.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")

        # Preprocessing uygula
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        # Feature selection uygula (eğer aktifse)
        if (
            self.feature_selector is not None
            and self._feature_selection_config.enable_selection
        ):
            X = self.feature_selector.selector.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Probability prediction yap (classification için).

        Args:
            X: Feature matrix

        Returns:
            Probability array or None
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")

        # Preprocessing uygula
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        # Feature selection uygula (eğer aktifse)
        if (
            self.feature_selector is not None
            and self._feature_selection_config.enable_selection
        ):
            X = self.feature_selector.selector.transform(X)

        return self.model.predict_proba(X)

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Model ve preprocessor'ı kaydet.

        Args:
            filepath: Dosya yolu (None ise otomatik oluşturulur)

        Returns:
            str: Kaydedilen dosya yolu
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")

        return self._model_persistence.save_model(  # type: ignore[no-any-return]
            self.model,
            self.preprocessor,
            self.feature_columns,
            self.config,
            filepath,
        )

    def load_model(self, filepath: str) -> "ModelTrainer":
        """
        Model ve preprocessor'ı yükle.

        Args:
            filepath: Dosya yolu

        Returns:
            self
        """
        data = self._model_persistence.load_model(filepath, self.config)
        self.model = data["model"]
        self.preprocessor = data["preprocessor"]
        self.feature_columns = data["feature_columns"]
        return self
