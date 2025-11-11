"""
Model training modülü için sabitler
"""

# Logger separator
LOG_SEPARATOR = "=" * 60

# Model types
MODEL_TYPE_LIGHTGBM = "lightgbm"

# Task types
TASK_TYPE_REGRESSION = "regression"
TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_BINARY = "binary"
TASK_TYPE_MULTI_CLASS = "multi_class"

# Column names
COLUMN_DATETIME = "datetime"
COLUMN_TARGET = "target"
COLUMN_FUTURE_RETURN = "future_return"

# Window types
WINDOW_TYPE_EXPANDING = "expanding"
WINDOW_TYPE_ROLLING = "rolling"

# Plot orientations
ORIENTATION_HORIZONTAL = "horizontal"
ORIENTATION_VERTICAL = "vertical"

# SHAP plot types
SHAP_PLOT_SUMMARY = "summary"
SHAP_PLOT_WATERFALL = "waterfall"
SHAP_PLOT_DEPENDENCE = "dependence"

# File names
FILE_MODEL_PREFIX = "model_"
FILE_MODEL_EXTENSION = ".pkl"
FILE_SCALER_NAME = "scaler.pkl"
FILE_FEATURE_IMPORTANCE_PLOT = "feature_importance.png"
FILE_FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
FILE_SHAP_VALUES = "shap_values.npy"
DIR_SHAP_PLOTS = "shap_plots"
FILE_SHAP_SUMMARY = "shap_summary.png"
FILE_SHAP_WATERFALL = "shap_waterfall.png"
FILE_SHAP_DEPENDENCE_PREFIX = "shap_dependence_"
