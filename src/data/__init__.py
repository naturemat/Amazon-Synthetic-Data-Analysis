"""Paquete de datos: carga, validación, limpieza y transformación."""

from .data_loader import DataLoader, load_data
from .data_cleaner import DataCleaner, clean_data
from .transformation import CustomerTransformer, build_customer_features
from .schema import DATA_DICTIONARY, to_markdown

__all__ = [
    "DataLoader",
    "load_data",
    "DataCleaner",
    "clean_data",
    "CustomerTransformer",
    "build_customer_features",
    "DATA_DICTIONARY",
    "to_markdown",
]
