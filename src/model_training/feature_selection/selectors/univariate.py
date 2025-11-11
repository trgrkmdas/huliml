"""
Univariate feature selection - SelectKBest, SelectPercentile
"""

from typing import List, Optional
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
)

from ..base import BaseFeatureSelector


class UnivariateFeatureSelector(BaseFeatureSelector):
    """Univariate feature selection - SelectKBest or SelectPercentile"""

    def __init__(
        self,
        method: str = "k_best",
        k: int = 20,
        percentile: int = 10,
        score_func: str = "f_classif",
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            method: 'k_best' or 'percentile'
            k: Number of features to select (for k_best)
            percentile: Percentile of features to select (for percentile)
            score_func: Scoring function ('f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression', 'chi2')
            exclude_columns: List of column names to exclude from selection
        """
        super().__init__(exclude_columns)
        self.method = method
        self.k = k
        self.percentile = percentile
        self.score_func_name = score_func

    def _get_score_func(self):
        """Get sklearn score function"""
        score_funcs = {
            "f_classif": f_classif,
            "f_regression": f_regression,
            "mutual_info_classif": mutual_info_classif,
            "mutual_info_regression": mutual_info_regression,
            "chi2": chi2,
        }

        if self.score_func_name not in score_funcs:
            raise ValueError(
                f"Geçersiz score_func: {self.score_func_name}. "
                f"Seçenekler: {list(score_funcs.keys())}"
            )

        return score_funcs[self.score_func_name]

    def _create_selector(self):
        """Create the underlying sklearn selector"""
        score_func = self._get_score_func()

        if self.method == "k_best":
            return SelectKBest(score_func=score_func, k=self.k)
        elif self.method == "percentile":
            return SelectPercentile(score_func=score_func, percentile=self.percentile)
        else:
            raise ValueError(
                f"Geçersiz method: {self.method}. 'k_best' veya 'percentile' olmalı."
            )
