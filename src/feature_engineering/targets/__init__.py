"""
Target variable modülleri.
"""

from .binary import BinaryTarget
from .multiclass import MulticlassTarget
from .regression import RegressionTarget

__all__ = ["BinaryTarget", "MulticlassTarget", "RegressionTarget"]
