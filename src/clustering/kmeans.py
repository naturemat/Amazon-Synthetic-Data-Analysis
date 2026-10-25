"""
K-Means y selección del número de clusters (fase 5 de KDD).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from ..analysis.evaluation import elbow_point, evaluate_k, select_k
from ..config.settings import KMEANS_CONFIG, K_RANGE, RANDOM_STATE
from ..core.plotting import save_figure


class KMeansModel:
    """Envuelve K-Means y conserva la evidencia de la selección de k."""

    def __init__(self, n_clusters: Optional[int] = None, **kwargs):
        config = {**KMEANS_CONFIG, **kwargs}
        self.n_clusters = int(n_clusters or config["n_clusters"])
        self.config = config
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=config.get("init", "k-means++"),
            n_init=config.get("n_init", 20),
            max_iter=config.get("max_iter", 300),
            algorithm=config.get("algorithm", "lloyd"),
            random_state=config.get("random_state", RANDOM_STATE),
        )
        self._labels: Optional[np.ndarray] = None
        self._inertia: Optional[float] = None
        self._centroids: Optional[np.ndarray] = None
        self.evaluation_table = None
        self.selection: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> "KMeansModel":
        self.kmeans.fit(X)
        self._labels = self.kmeans.labels_
        self._inertia = float(self.kmeans.inertia_)
        self._centroids = self.kmeans.cluster_centers_
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self._labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(X)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def inertia(self) -> float:
        return self._inertia

    @property
    def centroids(self) -> np.ndarray:
        return self._centroids

    # -------------------------------------------------------------- selection
    def select_n_clusters(self, X: np.ndarray,
                          k_range: Optional[List[int]] = None) -> Dict[str, Any]:
        """Evalúa k con varias métricas y devuelve la decisión justificada."""
        k_range = k_range or K_RANGE
        table = evaluate_k(X, k_range)
        elbow = elbow_point(table["k"], table["inercia"])
        selection = select_k(table)
        selection["codo"] = elbow
        self.evaluation_table = table
        self.selection = selection
        plot_k_selection(table, elbow, selection["k_seleccionado"])
        return selection

    # ---------------------------------------------------------------- metrics
    def calculate_inertia_analysis(self, X: np.ndarray) -> Dict[str, float]:
        inertia_intra = self._inertia
        global_center = np.asarray(X).mean(axis=0)
        inertia_total = float(np.sum((np.asarray(X) - global_center) ** 2))
        inertia_inter = inertia_total - inertia_intra
        return {
            "inertia_total": inertia_total,
            "inertia_intra": inertia_intra,
            "inertia_inter": inertia_inter,
            "pct_intra": 100 * inertia_intra / inertia_total if inertia_total else 0.0,
            "pct_inter": 100 * inertia_inter / inertia_total if inertia_total else 0.0,
        }

    def print_inertia_analysis(self, X: np.ndarray) -> None:
        analysis = self.calculate_inertia_analysis(X)
        print("\nAnálisis de inercias")
        print(f"  Inercia total:      {analysis['inertia_total']:.4f}")
        print(f"  Inercia intraclase: {analysis['inertia_intra']:.4f} "
              f"({analysis['pct_intra']:.2f}%)")
        print(f"  Inercia interclase: {analysis['inertia_inter']:.4f} "
              f"({analysis['pct_inter']:.2f}%)")

    def save_evaluation(self, path: str) -> str:
        import json

        if self.evaluation_table is None or self.selection is None:
            raise ValueError("Ejecute select_n_clusters() primero.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.selection, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)


def plot_k_selection(table, elbow: Dict[str, Any], selected_k: int) -> str:
    """Grafica inercia, silhouette y Davies-Bouldin para justificar k."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(table["k"], table["inercia"], marker="o", color="#1f77b4")
    axes[0].axvline(selected_k, color="#d62728", linestyle="--", label=f"k elegido = {selected_k}")
    axes[0].axvline(elbow.get("k_codo", selected_k), color="gray", linestyle=":",
                    label=f"codo = {elbow.get('k_codo')}")
    axes[0].set_title("Método del codo")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inercia intraclase")
    axes[0].legend(fontsize=8)

    axes[1].plot(table["k"], table["silhouette"], marker="o", color="#2ca02c")
    axes[1].axvline(selected_k, color="#d62728", linestyle="--")
    axes[1].set_title("Coeficiente de silueta (mayor = mejor)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    axes[2].plot(table["k"], table["davies_bouldin"], marker="o", color="#ff7f0e")
    axes[2].axvline(selected_k, color="#d62728", linestyle="--")
    axes[2].set_title("Davies-Bouldin (menor = mejor)")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Davies-Bouldin")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Selección de k con evidencia empírica", fontsize=13)
    fig.tight_layout()
    return save_figure(fig, "seleccion_k")


def perform_kmeans(X: np.ndarray, n_clusters: Optional[int] = None) -> Tuple[KMeansModel, np.ndarray]:
    model = KMeansModel(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels
