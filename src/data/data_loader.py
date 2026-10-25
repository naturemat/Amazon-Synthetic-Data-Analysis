"""
Carga, validación y perfilado del dataset (fases 2 y 3 de KDD).

El perfilado no es decorativo: produce la evidencia que permite justificar la
selección de datos y detectar señales de que el dataset es simulado.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config.settings import (
    DATASET,
    DTYPES,
    REQUIRED_COLUMNS,
    RESULTS_DIR,
)


class DataLoader:
    """Carga el CSV y construye un perfil reproducible del dataset."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath or DATASET)
        self._df: Optional[pd.DataFrame] = None
        self._profile: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ load
    def load(self, parse_dates: bool = True) -> pd.DataFrame:
        """Carga el dataset aplicando tipos explícitos y validando el esquema."""
        self._df = pd.read_csv(
            self.filepath,
            parse_dates=["OrderDate"] if parse_dates else False,
            dtype=DTYPES,
        )
        self.validate_schema()
        return self._df

    def validate_schema(self) -> None:
        """Comprueba que estén todas las columnas necesarias y la fecha sea válida."""
        if self._df is None:
            raise ValueError("No hay datos cargados. Ejecute load() primero.")

        missing = [c for c in REQUIRED_COLUMNS if c not in self._df.columns]
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")

        if "OrderDate" in self._df.columns:
            invalid_dates = int(self._df["OrderDate"].isna().sum())
            if invalid_dates:
                raise ValueError(
                    f"{invalid_dates} fechas no se pudieron interpretar. "
                    "Revise el formato de OrderDate."
                )

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No hay datos cargados. Ejecute load() primero.")
        return self._df

    # --------------------------------------------------------------- profile
    def profile(self) -> Dict[str, Any]:
        """Calcula el perfil del dataset y guarda la evidencia de simulación."""
        df = self.data
        duplicate_rows = int(df.duplicated().sum())
        nulls = df.isna().sum()
        years = df["OrderDate"].dt.year.value_counts().sort_index()
        rows_per_year = {int(year): int(count) for year, count in years.items()}
        year_cv = float(years.std(ddof=0) / years.mean()) if len(years) > 1 else 0.0

        items_per_order = df.groupby("OrderID").size()
        max_items_per_order = int(items_per_order.max())
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

        profile: Dict[str, Any] = {
            "archivo": self.filepath.name,
            "filas": int(len(df)),
            "columnas": int(df.shape[1]),
            "valores_nulos": int(nulls.sum()),
            "columnas_con_nulos": {
                str(col): int(value) for col, value in nulls[nulls > 0].items()
            },
            "filas_duplicadas": duplicate_rows,
            "clientes_unicos": int(df["CustomerID"].nunique()),
            "pedidos_unicos": int(df["OrderID"].nunique()),
            "productos_unicos": int(df["ProductID"].nunique()),
            "vendedores_unicos": int(df["SellerID"].nunique()),
            "fecha_min": df["OrderDate"].min().date().isoformat(),
            "fecha_max": df["OrderDate"].max().date().isoformat(),
            "dias_cubiertos": int((df["OrderDate"].max() - df["OrderDate"].min()).days),
            "filas_por_anio": rows_per_year,
            "coeficiente_variacion_anual": round(year_cv, 5),
            "maximo_items_por_pedido": max_items_per_order,
            "columnas_constantes": constant_cols,
            "estados_pedido": {
                str(k): int(v) for k, v in df["OrderStatus"].value_counts().items()
            },
        }

        reasons = []
        if profile["filas"] == 100_000:
            reasons.append("El número de filas es exactamente 100 000.")
        if profile["valores_nulos"] == 0:
            reasons.append("No hay valores nulos en ninguna columna.")
        if duplicate_rows == 0:
            reasons.append("No hay filas duplicadas exactas.")
        if year_cv < 0.02:
            reasons.append(
                f"Las filas por año son casi uniformes (CV = {year_cv:.4f})."
            )
        if max_items_per_order == 1:
            reasons.append(
                "Cada pedido contiene exactamente un producto (no hay canastas)."
            )
        if profile["fecha_min"].startswith("2020-01-01"):
            reasons.append("El periodo inicia exactamente el 2020-01-01.")

        profile["evidencia_dataset_simulado"] = reasons
        profile["es_probablemente_simulado"] = len(reasons) >= 3
        profile["conclusion_perfilado"] = (
            "El dataset presenta un patrón compatible con datos simulados. "
            "Se declara así en la metodología; la actividad permite datos reales "
            "o simulados, por lo que esto no invalida el ejercicio."
            if profile["es_probablemente_simulado"]
            else "No hay evidencia suficiente para afirmar que el dataset sea simulado."
        )

        self._profile = profile
        return profile

    def get_info(self, profile: Optional[Dict[str, Any]] = None) -> None:
        """Imprime un resumen legible del dataset y de su perfil."""
        profile = profile or self.profile()
        print("\n" + "=" * 80)
        print("FASE 2. PERFIL DEL DATASET")
        print("=" * 80)
        print(f"Archivo: {profile['archivo']}")
        print(f"Dimensiones: {profile['filas']:,} filas x {profile['columnas']} columnas")
        print(f"Periodo: {profile['fecha_min']} a {profile['fecha_max']}")
        print(f"Clientes / pedidos / productos: "
              f"{profile['clientes_unicos']:,} / {profile['pedidos_unicos']:,} / "
              f"{profile['productos_unicos']:,}")
        print(f"Nulos: {profile['valores_nulos']} | Duplicados: {profile['filas_duplicadas']}")
        print(f"Máximo de ítems por pedido: {profile['maximo_items_por_pedido']}")
        print(f"¿Probablemente simulado?: {profile['es_probablemente_simulado']}")
        for reason in profile["evidencia_dataset_simulado"]:
            print(f"  - {reason}")

    def save_profile(self, path: Optional[str] = None) -> str:
        """Guarda el perfil en JSON para anexarlo al documento."""
        profile = self._profile or self.profile()
        target = Path(path or RESULTS_DIR / "perfil_dataset.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(target)


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Función de conveniencia para cargar el dataset."""
    return DataLoader(filepath).load()
