"""Paquete de reportes."""

from .summary import ReportGenerator, generate_summary
from .kdd_report import KDDReport

__all__ = [
    "ReportGenerator",
    "generate_summary",
    "KDDReport",
]
