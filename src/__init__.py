"""
Proyecto de aplicación práctica del proceso KDD.

Incluye carga y limpieza de datos, transformación, minería, evaluación,
experimento de degradación y traducción del conocimiento a acciones.
"""

from . import config
from . import core
from . import data
from . import analysis
from . import visualization
from . import clustering
from . import reports
from . import kdd
from . import experiments

__version__ = "2.0.0"
__author__ = "Proyecto KDD"

__all__ = [
    "config",
    "core",
    "data",
    "analysis",
    "visualization",
    "clustering",
    "reports",
    "kdd",
    "experiments",
]
