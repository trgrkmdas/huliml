"""
SHAP Analyzer - Model interpretability için SHAP değerleri
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ...logger import get_logger
from ..base import BaseModel

logger = get_logger("MLProject.SHAPAnalysis")


class SHAPAnalyzer:
    """SHAP değerleri ile model interpretability analizi"""

    def __init__(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Eğitilmiş model (BaseModel)
            X: Feature matrix (background data için)
            feature_names: Feature isimleri (None ise X.columns kullanılır)
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP paketi yüklü değil. Yüklemek için: pip install shap"
            )

        if not model.is_fitted:
            raise ValueError("Model henüz fit edilmedi. Önce fit() çağrılmalı.")

        self.model = model
        self.X = X
        self.feature_names = feature_names or list(X.columns)
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None

        # Model'den sklearn estimator'ı al
        if hasattr(model, "model"):
            self.sklearn_model = model.model
        else:
            self.sklearn_model = model

    def _create_explainer(self, explainer_type: str = "tree") -> None:
        """
        SHAP explainer oluştur.

        Args:
            explainer_type: Explainer tipi ('tree', 'kernel', 'linear')
        """
        logger.info(f"🔍 SHAP {explainer_type} explainer oluşturuluyor...")

        # Background data (örnekleme ile hızlandırma)
        background_size = min(100, len(self.X))
        background_data = self.X.sample(n=background_size, random_state=42).values

        if explainer_type == "tree":
            # TreeExplainer (LightGBM, XGBoost, etc. için)
            self.explainer = shap.TreeExplainer(self.sklearn_model)
        elif explainer_type == "kernel":
            # KernelExplainer (herhangi bir model için, yavaş)
            self.explainer = shap.KernelExplainer(
                self.sklearn_model.predict, background_data
            )
        elif explainer_type == "linear":
            # LinearExplainer (linear modeller için)
            self.explainer = shap.LinearExplainer(self.sklearn_model, background_data)
        else:
            raise ValueError(
                f"Geçersiz explainer_type: {explainer_type}. "
                "Must be 'tree', 'kernel', or 'linear'"
            )

        logger.info("✅ SHAP explainer oluşturuldu")

    def calculate_shap_values(
        self,
        X_explain: Optional[pd.DataFrame] = None,
        explainer_type: str = "tree",
        max_evals: Optional[int] = None,
    ) -> np.ndarray:
        """
        SHAP değerlerini hesapla.

        Args:
            X_explain: Açıklanacak veri (None ise self.X kullanılır)
            explainer_type: Explainer tipi
            max_evals: Maksimum evaluation sayısı (kernel için)

        Returns:
            SHAP değerleri array
        """
        if self.explainer is None:
            self._create_explainer(explainer_type=explainer_type)

        if X_explain is None:
            X_explain = self.X

        # Sample if too large (SHAP yavaş olabilir)
        if len(X_explain) > 1000:
            logger.warning(
                f"⚠️  Veri çok büyük ({len(X_explain)} satır). "
                "İlk 1000 satır kullanılacak."
            )
            X_explain = X_explain.head(1000)

        X_explain_values = X_explain.values

        logger.info(f"📊 SHAP değerleri hesaplanıyor ({len(X_explain)} örnek)...")

        if explainer_type == "kernel" and max_evals is not None:
            self.shap_values = self.explainer.shap_values(
                X_explain_values, nsamples=max_evals
            )
        else:
            self.shap_values = self.explainer.shap_values(X_explain_values)

        # Classification için (shap_values bir liste olabilir)
        if isinstance(self.shap_values, list):
            # Binary classification için genelde tek class'ın değerleri
            self.shap_values = (
                self.shap_values[1]
                if len(self.shap_values) > 1
                else self.shap_values[0]
            )

        logger.info("✅ SHAP değerleri hesaplandı")

        return self.shap_values

    def plot_summary(
        self,
        X_explain: Optional[pd.DataFrame] = None,
        top_n: int = 20,
        plot_type: str = "bar",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        SHAP summary plot.

        Args:
            X_explain: Açıklanacak veri
            top_n: Kaç feature gösterilecek
            plot_type: Plot tipi ('bar', 'dot', 'violin')
            show: Görselleştirmeyi göster
            save_path: Kaydetme yolu
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_explain=X_explain)

        if X_explain is None:
            X_explain = self.X

        # Sample if too large
        if len(X_explain) > 1000:
            X_explain = X_explain.head(1000)
            self.shap_values = self.shap_values[:1000]

        logger.info(f"📊 SHAP summary plot oluşturuluyor ({plot_type})...")

        shap.summary_plot(
            self.shap_values,
            X_explain,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=top_n,
            show=show,
        )

        if save_path:
            import matplotlib.pyplot as plt

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"💾 SHAP summary plot kaydedildi: {save_path}")
            if not show:
                plt.close()

    def plot_waterfall(
        self,
        instance_idx: int = 0,
        X_explain: Optional[pd.DataFrame] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        SHAP waterfall plot (tek bir örnek için).

        Args:
            instance_idx: Hangi örnek gösterilecek
            X_explain: Açıklanacak veri
            show: Görselleştirmeyi göster
            save_path: Kaydetme yolu
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_explain=X_explain)

        if X_explain is None:
            X_explain = self.X

        logger.info(f"📊 SHAP waterfall plot oluşturuluyor (örnek {instance_idx})...")

        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=(
                    self.explainer.expected_value
                    if hasattr(self.explainer, "expected_value")
                    else 0
                ),
                data=X_explain.iloc[instance_idx].values,
                feature_names=self.feature_names,
            ),
            show=show,
        )

        if save_path:
            import matplotlib.pyplot as plt

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"💾 SHAP waterfall plot kaydedildi: {save_path}")
            if not show:
                plt.close()

    def plot_dependence(
        self,
        feature: str,
        X_explain: Optional[pd.DataFrame] = None,
        interaction_feature: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        SHAP dependence plot (feature etkileşimi).

        Args:
            feature: Analiz edilecek feature
            X_explain: Açıklanacak veri
            interaction_feature: Etkileşim feature'ı (opsiyonel)
            show: Görselleştirmeyi göster
            save_path: Kaydetme yolu
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_explain=X_explain)

        if X_explain is None:
            X_explain = self.X

        if feature not in self.feature_names:
            raise ValueError(f"Feature bulunamadı: {feature}")

        feature_idx = self.feature_names.index(feature)
        interaction_idx = (
            self.feature_names.index(interaction_feature)
            if interaction_feature and interaction_feature in self.feature_names
            else None
        )

        logger.info(f"📊 SHAP dependence plot oluşturuluyor ({feature})...")

        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X_explain.values,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=show,
        )

        if save_path:
            import matplotlib.pyplot as plt

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"💾 SHAP dependence plot kaydedildi: {save_path}")
            if not show:
                plt.close()

    def get_feature_importance_from_shap(self) -> Dict[str, float]:
        """
        SHAP değerlerinden feature importance hesapla.

        Returns:
            Dictionary of feature importance (mean absolute SHAP values)
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        # Mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        return dict(zip(self.feature_names, mean_abs_shap))

    def save_shap_values(self, file_path: str) -> None:
        """
        SHAP değerlerini kaydet.

        Args:
            file_path: Kaydetme yolu (CSV veya NPY)
        """
        if self.shap_values is None:
            raise ValueError("SHAP değerleri henüz hesaplanmadı.")

        file_path_obj = Path(file_path)

        if file_path_obj.suffix == ".csv":
            df = pd.DataFrame(
                self.shap_values,
                columns=self.feature_names,
                index=self.X.index[: len(self.shap_values)],
            )
            df.to_csv(file_path)
            logger.info(f"💾 SHAP değerleri CSV'ye kaydedildi: {file_path}")
        elif file_path_obj.suffix == ".npy":
            np.save(file_path, self.shap_values)
            logger.info(f"💾 SHAP değerleri NPY'ye kaydedildi: {file_path}")
        else:
            raise ValueError("Desteklenmeyen dosya formatı. CSV veya NPY kullanın.")
