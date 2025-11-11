"""
Model analysis helper sınıfı - Feature importance ve SHAP analizi
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from ..logger import get_logger
from .base import BaseModel
from .analysis import FeatureImportanceAnalyzer, SHAPAnalyzer
from .constants import (
    ORIENTATION_HORIZONTAL,
    FILE_FEATURE_IMPORTANCE_PLOT,
    FILE_FEATURE_IMPORTANCE_CSV,
    DIR_SHAP_PLOTS,
    FILE_SHAP_SUMMARY,
    FILE_SHAP_WATERFALL,
    FILE_SHAP_DEPENDENCE_PREFIX,
    FILE_SHAP_VALUES,
    SHAP_PLOT_SUMMARY,
    SHAP_PLOT_WATERFALL,
    SHAP_PLOT_DEPENDENCE,
)

logger = get_logger("MLProject.ModelAnalyzer")


class ModelAnalyzer:
    """Model analysis için helper sınıf"""

    def __init__(self, analysis_config, model_config, paths_config):
        """
        Args:
            analysis_config: Analysis config
            model_config: Model config
            paths_config: Paths config
        """
        self._analysis_config = analysis_config
        self._model_config = model_config
        self._paths_config = paths_config
        self.feature_importance_analyzer: Optional[FeatureImportanceAnalyzer] = None
        self.shap_analyzer: Optional[SHAPAnalyzer] = None

    def analyze_feature_importance(
        self, model: BaseModel, X_train_scaled: pd.DataFrame
    ) -> None:
        """
        Feature importance analizi yap.

        Args:
            model: Eğitilmiş model
            X_train_scaled: Scaled training features
        """
        logger.info("\n📊 Feature Importance Analizi...")

        # Analyzer oluştur
        self.feature_importance_analyzer = FeatureImportanceAnalyzer(
            model=model,
            feature_names=list(X_train_scaled.columns),
        )

        # Importance al
        top_features = self.feature_importance_analyzer.get_top_features(
            top_n=self._analysis_config.feature_importance_top_n
        )

        logger.info(f"✅ Top {len(top_features)} feature importance:")
        for idx, row in top_features.iterrows():
            logger.info(f"   {row['rank']}. {row['feature']}: {row['importance']:.4f}")

        # Summary
        summary = self.feature_importance_analyzer.get_summary()
        logger.info("\n📈 Feature Importance Özeti:")
        logger.info(f"   Toplam feature: {summary['total_features']}")
        logger.info(f"   Ortalama importance: {summary['mean_importance']:.4f}")
        logger.info(f"   Max importance: {summary['max_importance']:.4f}")

        # Plot kaydet
        if self._analysis_config.feature_importance_save_plot:
            plot_path = (
                Path(self._paths_config.models_dir) / FILE_FEATURE_IMPORTANCE_PLOT
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            if (
                self._analysis_config.feature_importance_plot_type
                == ORIENTATION_HORIZONTAL
            ):
                self.feature_importance_analyzer.plot_horizontal(
                    top_n=self._analysis_config.feature_importance_top_n,
                    save_path=str(plot_path),
                    show=False,
                )
            else:
                self.feature_importance_analyzer.plot_importance(
                    top_n=self._analysis_config.feature_importance_top_n,
                    save_path=str(plot_path),
                    show=False,
                )

        # CSV kaydet
        if self._analysis_config.feature_importance_save_csv:
            csv_path = Path(self._paths_config.models_dir) / FILE_FEATURE_IMPORTANCE_CSV
            self.feature_importance_analyzer.save_to_csv(
                file_path=str(csv_path),
                top_n=None,  # Tüm feature'ları kaydet
            )

    def analyze_shap(self, model: BaseModel, X_train_scaled: pd.DataFrame) -> None:
        """
        SHAP analizi yap.

        Args:
            model: Eğitilmiş model
            X_train_scaled: Scaled training features
        """
        logger.info("\n🔍 SHAP Analizi...")

        # Background data (SHAP için)
        background_data = X_train_scaled
        if (
            self._analysis_config.shap_sample_size
            and len(background_data) > self._analysis_config.shap_sample_size
        ):
            background_data = background_data.sample(
                n=self._analysis_config.shap_sample_size,
                random_state=self._model_config.random_seed,
            )
            logger.info(f"   Background data örnekleme: {len(background_data)} satır")

        # Analyzer oluştur
        try:
            self.shap_analyzer = SHAPAnalyzer(
                model=model,
                X=background_data,
                feature_names=list(background_data.columns),
            )
        except ImportError:
            logger.error("❌ SHAP paketi yüklü değil. Yüklemek için: pip install shap")
            return

        # SHAP değerlerini hesapla
        self.shap_analyzer.calculate_shap_values(
            explainer_type=self._analysis_config.shap_explainer_type
        )

        # Feature importance from SHAP
        shap_importance = self.shap_analyzer.get_feature_importance_from_shap()
        sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[
            : self._analysis_config.shap_top_n
        ]

        logger.info(f"✅ Top {len(sorted_shap)} SHAP feature importance:")
        for rank, (feature, importance) in enumerate(sorted_shap, 1):
            logger.info(f"   {rank}. {feature}: {importance:.4f}")

        # Plot'ları kaydet
        if self._analysis_config.shap_save_plots:
            plots_dir = Path(self._paths_config.models_dir) / DIR_SHAP_PLOTS
            plots_dir.mkdir(parents=True, exist_ok=True)

            for plot_type in self._analysis_config.shap_plot_types:
                if plot_type == SHAP_PLOT_SUMMARY:
                    plot_path = plots_dir / FILE_SHAP_SUMMARY
                    self.shap_analyzer.plot_summary(
                        top_n=self._analysis_config.shap_top_n,
                        plot_type="bar",
                        save_path=str(plot_path),
                        show=False,
                    )
                elif plot_type == SHAP_PLOT_WATERFALL:
                    plot_path = plots_dir / FILE_SHAP_WATERFALL
                    self.shap_analyzer.plot_waterfall(
                        instance_idx=0,
                        save_path=str(plot_path),
                        show=False,
                    )
                elif plot_type == SHAP_PLOT_DEPENDENCE:
                    # İlk birkaç önemli feature için
                    top_features = [f[0] for f in sorted_shap[:3]]
                    for feature in top_features:
                        plot_path = (
                            plots_dir / f"{FILE_SHAP_DEPENDENCE_PREFIX}{feature}.png"
                        )
                        self.shap_analyzer.plot_dependence(
                            feature=feature,
                            save_path=str(plot_path),
                            show=False,
                        )

        # SHAP değerlerini kaydet
        if self._analysis_config.shap_save_values:
            shap_values_path = Path(self._paths_config.models_dir) / FILE_SHAP_VALUES
            self.shap_analyzer.save_shap_values(file_path=str(shap_values_path))
