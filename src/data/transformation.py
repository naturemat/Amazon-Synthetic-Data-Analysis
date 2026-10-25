"""
Transformación y construcción del perfil de cliente (fase 4 de KDD).

El dataset está a nivel de pedido; la minería se hace a nivel de cliente. Por
eso la transformación central es una agregación por ``CustomerID`` que resume
el comportamiento en cinco variables interpretables:

- ``Ingreso_Neto``: ingreso acumulado descontando promociones.
- ``Ticket_Promedio``: ingreso neto medio por pedido.
- ``Frecuencia``: número de pedidos realizados.
- ``Descuento_Promedio``: sensibilidad al precio.
- ``Dias_Ultima_Compra``: recencia medida contra la fecha de corte del dataset.

Decisiones metodológicas
------------------------
- La recencia usa ``OrderDate.max()`` como fecha de corte (nunca ``now()``),
  de modo que el resultado es reproducible y no depende del día de ejecución.
- Las colas largas se recortan al percentil 1-99 solo para el modelo; el
  reporte conserva los valores originales.
- La estandarización es obligatoria porque K-Means usa distancias euclidianas
  y las variables tienen escalas muy distintas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config.settings import (
    CLUSTERING_VARS,
    PCA_VARIANCE_TARGET,
    RANDOM_STATE,
    RESULTS_DIR,
    WINSOR_QUANTILES,
    WINSOR_VARS,
)


class CustomerTransformer:
    """Agrega pedidos por cliente y prepara la matriz de modelado."""

    def __init__(self, df: pd.DataFrame, cutoff_date: Optional[pd.Timestamp] = None,
                 clustering_vars: Optional[list] = None):
        if df.empty:
            raise ValueError("El dataframe de ventas está vacío.")
        self._df = df.copy()
        self._clustering_vars = clustering_vars or CLUSTERING_VARS
        self._cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else pd.Timestamp(df["OrderDate"].max())
        self._features: Optional[pd.DataFrame] = None
        self._model_features: Optional[pd.DataFrame] = None
        self._model_values: Optional[pd.DataFrame] = None
        self._scaler: Optional[StandardScaler] = None
        self.report: Dict[str, Any] = {
            "fecha_corte": self._cutoff.date().isoformat(),
            "filas_pedido": int(len(df)),
            "variables_modelo": self._clustering_vars,
        }

    # ------------------------------------------------------------- properties
    @property
    def cutoff_date(self) -> pd.Timestamp:
        return self._cutoff

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self.aggregate()
        return self._features

    @property
    def model_features(self) -> pd.DataFrame:
        if self._model_features is None:
            self.prepare_model_features()
        return self._model_features

    @property
    def scaler(self) -> StandardScaler:
        if self._scaler is None:
            self.prepare_model_features()
        return self._scaler

    # -------------------------------------------------------------- aggregate
    def aggregate(self) -> pd.DataFrame:
        """Construye una fila por cliente con variables de comportamiento."""
        df = self._df.copy()

        grouped = df.groupby("CustomerID", observed=True)
        features = grouped.agg(
            Ingreso_Neto=("Ingreso_Neto", "sum"),
            Ingreso_Facturado=("Ingreso_Facturado", "sum"),
            Ticket_Promedio=("Ingreso_Neto", "mean"),
            Frecuencia=("OrderID", "count"),
            Descuento_Promedio=("Discount", "mean"),
            Ultima_Compra=("OrderDate", "max"),
            Primera_Compra=("OrderDate", "min"),
        )

        features["Dias_Ultima_Compra"] = (
            self._cutoff - features["Ultima_Compra"]
        ).dt.days.clip(lower=0).astype(int)

        # Atributos descriptivos del cliente (no entran al modelo).
        # Se calculan con groupby + conteo, que es mucho más rápido que aplicar
        # una función de moda a 29 000 grupos.
        def _top_by_customer(column: str) -> pd.Series:
            counts = (
                df.groupby(["CustomerID", column], observed=True)
                .size()
                .reset_index(name="_n")
            )
            idx = counts.groupby("CustomerID")["_n"].idxmax()
            top = counts.loc[idx].set_index("CustomerID")[column]
            top.name = f"_{column}"
            return top

        descriptive = pd.DataFrame(index=features.index)
        descriptive["Categoria_Favorita"] = _top_by_customer("Category").reindex(features.index)
        descriptive["Marca_Favorita"] = _top_by_customer("Brand").reindex(features.index)
        descriptive["Ciudad"] = _top_by_customer("City").reindex(features.index)
        descriptive["Estado"] = _top_by_customer("State").reindex(features.index)
        descriptive["Metodo_Pago_Favorito"] = _top_by_customer("PaymentMethod").reindex(features.index)
        features = features.join(descriptive)

        # Antigüedad y valor anualizado: contexto para interpretar el ingreso.
        features["Dias_Como_Cliente"] = (
            self._cutoff - features["Primera_Compra"]
        ).dt.days.clip(lower=0).astype(int) + 1
        features["Ingreso_Anualizado"] = (
            features["Ingreso_Neto"] / features["Dias_Como_Cliente"] * 365.25
        )

        # Puntajes RFM por quintiles (1 = peor, 5 = mejor).
        features["R_Score"] = pd.qcut(
            features["Dias_Ultima_Compra"].rank(method="first"), 5,
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        features["F_Score"] = pd.qcut(
            features["Frecuencia"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        features["M_Score"] = pd.qcut(
            features["Ingreso_Neto"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        features["RFM_Score"] = (
            features["R_Score"] + features["F_Score"] + features["M_Score"]
        )

        self._features = features
        self.report["clientes"] = int(len(features))
        self.report["recencia"] = {
            "min_dias": int(features["Dias_Ultima_Compra"].min()),
            "mediana_dias": float(features["Dias_Ultima_Compra"].median()),
            "max_dias": int(features["Dias_Ultima_Compra"].max()),
            "fecha_corte": self._cutoff.date().isoformat(),
            "formula": "(fecha_corte - max(OrderDate del cliente)).days",
        }
        return features

    # --------------------------------------------------------------- modeling
    def prepare_model_features(self, winsorize: bool = True) -> pd.DataFrame:
        """Recorta colas, valida nulos y estandariza las variables del modelo."""
        features = self.features
        missing_cols = [c for c in self._clustering_vars if c not in features.columns]
        if missing_cols:
            raise ValueError(f"Faltan variables de modelado: {missing_cols}")
        if features[self._clustering_vars].isna().any().any():
            raise ValueError("Hay valores nulos en las variables de modelado.")

        model = features[self._clustering_vars].astype(float).copy()
        report: Dict[str, Any] = {"winsorizacion": {}, "antes": {}, "despues": {}}

        for col in self._clustering_vars:
            report["antes"][col] = {
                "min": float(model[col].min()),
                "max": float(model[col].max()),
                "media": float(model[col].mean()),
                "desv_est": float(model[col].std()),
            }
            if winsorize and col in WINSOR_VARS:
                low = float(model[col].quantile(WINSOR_QUANTILES[0]))
                high = float(model[col].quantile(WINSOR_QUANTILES[1]))
                clipped = int(((model[col] < low) | (model[col] > high)).sum())
                model[col] = model[col].clip(low, high)
                report["winsorizacion"][col] = {
                    "percentil_inferior": WINSOR_QUANTILES[0],
                    "percentil_superior": WINSOR_QUANTILES[1],
                    "limite_inferior": round(low, 4),
                    "limite_superior": round(high, 4),
                    "valores_recortados": clipped,
                    "porcentaje_recortado": round(100 * clipped / len(model), 4),
                }
            report["despues"][col] = {
                "min": float(model[col].min()),
                "max": float(model[col].max()),
                "media": float(model[col].mean()),
                "desv_est": float(model[col].std()),
            }

        scaler = StandardScaler()
        scaled = scaler.fit_transform(model)
        self._model_values = model.copy()
        self._model_features = pd.DataFrame(
            scaled, index=model.index, columns=self._clustering_vars
        )
        self._scaler = scaler
        self.report["estandarizacion"] = report
        self.report["medias_escalador"] = {
            col: round(float(mean), 4)
            for col, mean in zip(self._clustering_vars, scaler.mean_)
        }
        self.report["desviaciones_escalador"] = {
            col: round(float(std), 4)
            for col, std in zip(self._clustering_vars, scaler.scale_)
        }
        return self._model_features

    def pca_diagnostics(self, n_components: Optional[int] = None) -> Dict[str, Any]:
        """Calcula varianza explicada y carga de componentes (para justificar ACP)."""
        X = self.model_features.values
        n_components = n_components or len(self._clustering_vars)
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        pca.fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_for_target = int(np.searchsorted(cumulative, PCA_VARIANCE_TARGET) + 1)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=self._clustering_vars,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )

        result = {
            "varianza_explicada": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "varianza_acumulada": [round(float(v), 4) for v in cumulative],
            "componentes_para_85pct": int(min(n_for_target, n_components)),
            "varianza_2_componentes": round(float(cumulative[min(1, len(cumulative) - 1)]), 4),
            "cargas": loadings.round(4).to_dict(),
        }
        self.report["acp"] = result
        return result

    def get_model_matrix(self) -> pd.DataFrame:
        return self.model_features.copy()

    @property
    def model_values(self) -> pd.DataFrame:
        """Variables de modelado después del recorte, antes de estandarizar."""
        if self._model_values is None:
            self.prepare_model_features()
        return self._model_values.copy()

    def get_features_with_index(self) -> pd.DataFrame:
        return self.features.copy()

    def save_features(self, path: Optional[str] = None) -> str:
        """Guarda el perfil de cliente en CSV (sin datos personales)."""
        target = Path(path or RESULTS_DIR / "clientes_perfil.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(target, index=True, encoding="utf-8-sig")
        return str(target)

    def save_report(self, path: Optional[str] = None) -> str:
        target = Path(path or RESULTS_DIR / "reporte_transformacion.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)


def build_customer_features(df: pd.DataFrame,
                            cutoff_date: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, CustomerTransformer]:
    """Función de conveniencia para construir el perfil de cliente."""
    transformer = CustomerTransformer(df, cutoff_date=cutoff_date)
    features = transformer.aggregate()
    return features, transformer
