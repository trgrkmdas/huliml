"""
Feature Importance Analyzer - Görselleştirme ve analiz
"""

from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ...logger import get_logger
from ..base import BaseModel

logger = get_logger("MLProject.FeatureImportance")


class FeatureImportanceAnalyzer:
    """Feature importance analizi ve görselleştirme"""

    def __init__(self, model: BaseModel, feature_names: Optional[List[str]] = None):
        """
        Args:
            model: Eğitilmiş model (BaseModel)
            feature_names: Feature isimleri (None ise model'den alınır)
        """
        if not model.is_fitted:
            raise ValueError("Model henüz fit edilmedi. Önce fit() çağrılmalı.")

        self.model = model
        self.feature_names = feature_names
        self.importance_dict: Optional[Dict[str, float]] = None

    def get_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Feature importance'ı al.

        Args:
            importance_type: Importance tipi ('gain', 'split', etc.)

        Returns:
            Dictionary of feature importance
        """
        self.importance_dict = self.model.get_feature_importance(
            importance_type=importance_type
        )
        return self.importance_dict.copy()

    def get_top_features(self, top_n: int = 20) -> pd.DataFrame:
        """
        En önemli feature'ları al.

        Args:
            top_n: Kaç feature döndürülecek

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.importance_dict is None:
            self.get_importance()

        # Sort by importance
        sorted_features = sorted(
            self.importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        df = pd.DataFrame(sorted_features, columns=["feature", "importance"])
        df["rank"] = range(1, len(df) + 1)

        return df

    def plot_importance(
        self,
        top_n: int = 20,
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None,
        show: bool = True,
        orientation: str = "horizontal",
    ) -> None:
        """
        Feature importance görselleştir.

        Args:
            top_n: Kaç feature gösterilecek
            figsize: Figure boyutu
            save_path: Kaydetme yolu (None ise kaydetme)
            show: Görselleştirmeyi göster
            orientation: 'horizontal' (yatay) veya 'vertical' (dikey)
        """
        if self.importance_dict is None:
            self.get_importance()

        # Top features
        top_features_df = self.get_top_features(top_n=top_n)

        # Plot orientation'a göre ayarla
        if orientation == "horizontal":
            x_col, y_col = "importance", "feature"
        elif orientation == "vertical":
            x_col, y_col = "feature", "importance"
        else:
            raise ValueError(
                f"Geçersiz orientation: {orientation}. 'horizontal' veya 'vertical' olmalı."
            )

        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(
            data=top_features_df,
            x=x_col,
            y=y_col,
            palette="viridis",
        )
        plt.title(f"Top {top_n} Feature Importance", fontsize=16, fontweight="bold")
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 Feature importance plot kaydedildi: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_horizontal(
        self,
        top_n: int = 20,
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Yatay bar chart ile feature importance görselleştir.
        (Deprecated: plot_importance(orientation='horizontal') kullanın)

        Args:
            top_n: Kaç feature gösterilecek
            figsize: Figure boyutu
            save_path: Kaydetme yolu
            show: Görselleştirmeyi göster
        """
        self.plot_importance(
            top_n=top_n,
            figsize=figsize,
            save_path=save_path,
            show=show,
            orientation="horizontal",
        )

    def save_to_csv(self, file_path: str, top_n: Optional[int] = None) -> None:
        """
        Feature importance'ı CSV'ye kaydet.

        Args:
            file_path: Kaydetme yolu
            top_n: Kaç feature kaydedilecek (None ise hepsi)
        """
        if self.importance_dict is None:
            self.get_importance()

        if top_n is None:
            df = pd.DataFrame(
                list(self.importance_dict.items()), columns=["feature", "importance"]
            )
        else:
            df = self.get_top_features(top_n=top_n)

        df = df.sort_values("importance", ascending=False)
        df.to_csv(file_path, index=False)
        logger.info(f"💾 Feature importance CSV'ye kaydedildi: {file_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Feature importance özeti.

        Returns:
            Dictionary with summary statistics
        """
        if self.importance_dict is None:
            self.get_importance()

        importances = list(self.importance_dict.values())

        return {
            "total_features": len(self.importance_dict),
            "mean_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "max_importance": np.max(importances),
            "min_importance": np.min(importances),
            "top_10_percent": len(
                [x for x in importances if x > np.percentile(importances, 90)]
            ),
        }
