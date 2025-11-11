"""
Feature engineering için yardımcı fonksiyonlar.
"""

from typing import List, Optional
import pandas as pd


def find_column_by_pattern(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """
    DataFrame'de belirli pattern'lere göre kolon bulur.

    Args:
        df: DataFrame
        patterns: Aranacak pattern'ler (örn: ["MACD_", "STOCHk"])

    Returns:
        str: Bulunan ilk kolon adı, yoksa None
    """
    for col in df.columns:
        for pattern in patterns:
            if pattern in col:
                return col
    return None


def find_columns_by_patterns(
    df: pd.DataFrame, patterns_list: List[List[str]]
) -> List[Optional[str]]:
    """
    Birden fazla pattern için kolon bulur.

    Args:
        df: DataFrame
        patterns_list: Her biri pattern listesi olan liste

    Returns:
        List[Optional[str]]: Her pattern için bulunan kolonlar
    """
    return [find_column_by_pattern(df, patterns) for patterns in patterns_list]
