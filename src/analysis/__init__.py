"""Paquete de análisis: EDA, estadística, evaluación y asociación."""

from .eda import ExploratoryDataAnalysis, perform_eda
from .statistics import StatisticalAnalyzer, analyze
from .evaluation import (
    bootstrap_stability,
    elbow_point,
    evaluate_k,
    final_metrics,
    select_k,
)
from .association import MarketBasketAnalyzer

__all__ = [
    "ExploratoryDataAnalysis",
    "perform_eda",
    "StatisticalAnalyzer",
    "analyze",
    "bootstrap_stability",
    "elbow_point",
    "evaluate_k",
    "final_metrics",
    "select_k",
    "MarketBasketAnalyzer",
]
