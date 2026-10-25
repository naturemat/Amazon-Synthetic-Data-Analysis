"""
Análisis de componentes principales (fase 4 de KDD).

El ACP se usa con dos propósitos:

1. Diagnóstico de redundancia entre las variables de comportamiento.
2. Visualización bidimensional de los clusters.

La minería se ejecuta sobre las variables estandarizadas para no perder la
información que queda fuera de los dos primeros componentes (≈34 %). El
pipeline compara empíricamente ambas alternativas y deja constancia en el log.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config.settings import PCA_CONFIG, PCA_VARIANCE_TARGET, RANDOM_STATE
from ..core.plotting import save_figure


class PCAModel:
    """PCA con escalado integrado y utilidades de interpretación."""

    def __init__(self, n_components: Optional[int] = None,
                 random_state: Optional[int] = None):
        self.n_components = n_components or PCA_CONFIG["n_components"]
        self.random_state = random_state or RANDOM_STATE
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.scaler = StandardScaler()
        self._X_scaled: Optional[np.ndarray] = None
        self._X_pca: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None

    def fit_transform(self, data) -> np.ndarray:
        """Estandariza y proyecta. No imputa silenciosamente con cero."""
        if isinstance(data, pd.DataFrame):
            if data.isna().any().any():
                raise ValueError(
                    "El ACP recibió valores nulos. Impute o elimine antes de "
                    "transformar; usar fillna(0) sesga la media y la varianza."
                )
            self.feature_names = list(data.columns)
            values = data.values
        else:
            values = np.asarray(data)
            self.feature_names = [f"x{i}" for i in range(values.shape[1])]

        self._X_scaled = self.scaler.fit_transform(values)
        self._X_pca = self.pca.fit_transform(self._X_scaled)
        self._explained_variance = self.pca.explained_variance_ratio_
        return self._X_pca

    def transform(self, data) -> np.ndarray:
        values = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
        return self.pca.transform(self.scaler.transform(values))

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._explained_variance

    @property
    def total_explained_variance(self) -> float:
        return float(np.sum(self._explained_variance))

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self._explained_variance)

    def components_for_variance(self, target: float = PCA_VARIANCE_TARGET) -> int:
        """Número mínimo de componentes que alcanza la varianza objetivo."""
        return int(np.searchsorted(self.cumulative_variance, target) + 1)

    def get_loadings(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        names = feature_names or self.feature_names
        columns = [f"PC{i + 1}" for i in range(self.pca.n_components_)]
        return pd.DataFrame(self.pca.components_.T, columns=columns, index=names)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_componentes": int(self.pca.n_components_),
            "varianza_explicada": [round(float(v), 4) for v in self._explained_variance],
            "varianza_acumulada": [round(float(v), 4) for v in self.cumulative_variance],
            "varianza_2_componentes": round(float(self.cumulative_variance[1]), 4)
            if len(self.cumulative_variance) > 1 else None,
            "componentes_para_85pct": self.components_for_variance(),
        }

    def plot_variance(self, name: str = "acp_varianza") -> str:
        """Scree plot con varianza individual y acumulada."""
        if self._explained_variance is None:
            raise ValueError("Ejecute fit_transform() primero.")
        comps = np.arange(1, len(self._explained_variance) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(comps, self._explained_variance * 100, color="#4c72b0", alpha=0.8,
               label="Varianza por componente")
        ax.plot(comps, self.cumulative_variance * 100, marker="o", color="#c44e52",
                label="Varianza acumulada")
        ax.axhline(PCA_VARIANCE_TARGET * 100, color="gray", linestyle="--",
                   label=f"Objetivo {PCA_VARIANCE_TARGET:.0%}")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Varianza explicada (%)")
        ax.set_title("ACP: ¿cuánta información retienen los componentes?")
        ax.set_xticks(comps)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return save_figure(fig, name)

    def plot_projection(self, labels: Optional[np.ndarray] = None,
                        title: str = "Proyección de clientes en el plano PCA",
                        name: str = "acp_clientes") -> str:
        if self._X_pca is None:
            raise ValueError("Ejecute fit_transform() primero.")
        fig, ax = plt.subplots(figsize=(9, 7))
        if labels is None:
            ax.scatter(self._X_pca[:, 0], self._X_pca[:, 1], s=8, alpha=0.35,
                       color="gray")
        else:
            labels = np.asarray(labels)
            cmap = plt.get_cmap("tab10", int(labels.max()) + 1)
            for cluster in np.unique(labels):
                mask = labels == cluster
                ax.scatter(self._X_pca[mask, 0], self._X_pca[mask, 1], s=10,
                           alpha=0.55, color=cmap(int(cluster)),
                           label=f"Cluster {cluster}")
            ax.legend(fontsize=8, markerscale=2)
        ax.set_xlabel(f"PC1 ({self._explained_variance[0] * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({self._explained_variance[1] * 100:.2f}%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return save_figure(fig, name)

    def print_summary(self) -> None:
        print("\nACP")
        print(f"  Varianza PC1 + PC2: {self.total_explained_variance:.2%}")
        print(f"  Componentes para 85 %: {self.components_for_variance()}")


def perform_pca(data, n_components: Optional[int] = None) -> Tuple[PCAModel, np.ndarray]:
    model = PCAModel(n_components=n_components)
    return model, model.fit_transform(data)
