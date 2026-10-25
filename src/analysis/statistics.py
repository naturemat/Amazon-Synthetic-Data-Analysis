"""
Statistical analysis module for Amazon Sales Data Analysis.

Implements the transformation stage at customer level (phase 4 of KDD):
aggregation of transactional records into behavioral features, RFM
discretization and descriptive statistics.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config.settings import CLUSTERING_VARS, RFM_QUANTILES


class StatisticalAnalyzer:
    """Handles statistical analysis and customer-level transformations."""

    def __init__(self, df: pd.DataFrame,
                 reference_date: Optional[pd.Timestamp] = None):
        """
        Initialize the StatisticalAnalyzer.

        Parameters
        ----------
        df : pd.DataFrame
            Transactional dataframe (one row per order line).
        reference_date : pandas.Timestamp, optional
            Fixed cutoff used to compute recency. Defaults to the last order
            date, which keeps the analysis reproducible over time.
        """
        self._df = df
        self.reference_date = (
            pd.Timestamp(reference_date)
            if reference_date is not None
            else pd.Timestamp(df["OrderDate"].max())
        )

    # ------------------------------------------------------------- descriptive
    def calculate_descriptive_stats(self, columns: Optional[list] = None) -> pd.DataFrame:
        """Calculate descriptive statistics."""
        if columns:
            return self._df[columns].describe()
        return self._df.describe()

    def calculate_percentiles(self, columns: list,
                              percentiles: list = None) -> pd.DataFrame:
        """Calculate percentiles for the specified columns."""
        percentiles = percentiles or [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        return self._df[columns].describe(percentiles=percentiles)

    # ------------------------------------------------------------- aggregation
    def aggregate_by_customer(self) -> pd.DataFrame:
        """
        Aggregate the transactional data into customer-level behavioral features.

        Returns
        -------
        pd.DataFrame
            One row per customer with income, ticket, frequency, average
            discount, average price, last purchase and recency (days).
        """
        clientes = (
            self._df.groupby("CustomerID")
            .agg(
                Ingreso_Total=("TotalAmount", "sum"),
                Ticket_Promedio=("TotalAmount", "mean"),
                Frecuencia=("TotalAmount", "count"),
                Cantidad_Total=("Quantity", "sum"),
                Descuento_Promedio=("Discount", "mean"),
                Precio_Promedio=("UnitPrice", "mean"),
                Ultima_Compra=("OrderDate", "max"),
            )
            .round(2)
        )

        # Recency measured against a fixed cutoff, never against "today"
        clientes["Dias_Ultima_Compra"] = (
            self.reference_date - clientes["Ultima_Compra"]
        ).dt.days

        return clientes

    # --------------------------------------------------------------------- RFM
    @staticmethod
    def _score(series: pd.Series, higher_is_better: bool = True,
               quantiles: int = RFM_QUANTILES) -> pd.Series:
        """Discretize a variable into 1..q scores using quantiles."""
        ranks = series.rank(method="first", ascending=not higher_is_better)
        bins = pd.qcut(ranks, q=quantiles, labels=False, duplicates="drop")
        return (bins + 1).astype(int)

    def add_rfm_scores(self, clientes: pd.DataFrame) -> pd.DataFrame:
        """
        Add R, F and M scores (1 = worst, 5 = best) and an RFM segment.

        This is the discretization step of the transformation phase: it turns
        continuous behavioral variables into an interpretable scale that can be
        cross-checked against the K-Means solution.
        """
        result = clientes.copy()

        result["R_Score"] = self._score(result["Dias_Ultima_Compra"],
                                        higher_is_better=False)
        result["F_Score"] = self._score(result["Frecuencia"], higher_is_better=True)
        result["M_Score"] = self._score(result["Ingreso_Total"], higher_is_better=True)
        result["RFM_Score"] = (
            result["R_Score"] + result["F_Score"] + result["M_Score"]
        )
        result["RFM_Segmento"] = [
            self.rfm_segment(r, f, m)
            for r, f, m in zip(result["R_Score"], result["F_Score"], result["M_Score"])
        ]
        return result

    @staticmethod
    def rfm_segment(r: int, f: int, m: int) -> str:
        """Map an RFM triple to a business-readable segment name."""
        if r >= 4 and f >= 4 and m >= 4:
            return "Campeones"
        if r >= 3 and f >= 3:
            return "Leales"
        if r >= 4 and f <= 2:
            return "Prometedores"
        if r <= 2 and (f >= 3 or m >= 3):
            return "En riesgo (valioso)"
        if r <= 2 and f <= 2:
            return "Perdidos"
        return "Necesitan atención"

    # ---------------------------------------------------------------- profiles
    def calculate_cluster_profile(self, clientes_agg: pd.DataFrame,
                                  clustering_vars: List[str] = None) -> pd.DataFrame:
        """Calculate the mean profile of each cluster."""
        clustering_vars = clustering_vars or CLUSTERING_VARS
        return clientes_agg.groupby("Segmento")[clustering_vars].mean()

    def calculate_relative_profile(self, perfil_segmentos: pd.DataFrame,
                                   clientes_agg: pd.DataFrame,
                                   clustering_vars: List[str] = None) -> pd.DataFrame:
        """Profile relative to the global average (1.0 = average)."""
        clustering_vars = clustering_vars or CLUSTERING_VARS
        global_mean = clientes_agg[clustering_vars].mean().replace(0, np.nan)
        return perfil_segmentos / global_mean

    def calculate_inertia_analysis(self, X: np.ndarray, kmeans_model) -> dict:
        """Decompose total inertia into intra- and inter-cluster inertia."""
        inertia_intra = float(kmeans_model.inertia_)
        global_center = X.mean(axis=0)
        inertia_total = float(np.sum((X - global_center) ** 2))
        inertia_inter = inertia_total - inertia_intra

        return {
            "inertia_total": inertia_total,
            "inertia_intra": inertia_intra,
            "inertia_inter": inertia_inter,
            "pct_intra": 100 * inertia_intra / inertia_total,
            "pct_inter": 100 * inertia_inter / inertia_total,
        }


def analyze(df: pd.DataFrame,
            reference_date: Optional[pd.Timestamp] = None) -> StatisticalAnalyzer:
    """Convenience function to create a StatisticalAnalyzer."""
    return StatisticalAnalyzer(df, reference_date)
