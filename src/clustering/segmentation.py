"""
Segmentación de clientes e interpretación de clusters (fases 5 y 6 de KDD).

Combina la matriz estandarizada, K-Means y el perfil de cliente para producir
segmentos con etiquetas únicas, tabla de perfiles, métricas de validación y
figuras para el documento.

La etiqueta no se asigna con umbrales fijos sobre una sola variable: se parte
del cuadrante valor/recencia y se desempata con frecuencia, ticket o descuento.
Así se evita el error de dos clusters con la misma etiqueta.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from ..analysis.evaluation import bootstrap_stability, final_metrics
from ..config.settings import CLUSTERING_VARS, RESULTS_DIR
from ..core.plotting import save_figure
from .kmeans import KMeansModel


class CustomerSegmenter:
    """Ejecuta K-Means, valida la solución y traduce clusters a segmentos."""

    def __init__(self, features: pd.DataFrame, model_matrix: pd.DataFrame,
                 clustering_vars: Optional[List[str]] = None):
        if len(features) != len(model_matrix):
            raise ValueError("features y model_matrix deben tener el mismo número de filas.")
        self.features = features.copy()
        self.X = model_matrix.copy()
        self.clustering_vars = clustering_vars or CLUSTERING_VARS
        self.kmeans_model: Optional[KMeansModel] = None
        self.labels: Optional[np.ndarray] = None
        self.label_map: Dict[int, str] = {}
        self.evaluation: Dict[str, Any] = {}
        self.selection: Dict[str, Any] = {}

    # ------------------------------------------------------------------- fit
    def fit(self, k: Optional[int] = None, evaluate: bool = True) -> Dict[str, Any]:
        """Ajusta K-Means; si k es None lo selecciona con métricas."""
        X = self.X.values
        if evaluate:
            probe = KMeansModel()
            selection = probe.select_n_clusters(X)
            if k is None:
                k = selection["k_seleccionado"]
            else:
                selection["k_usuario"] = int(k)
                selection["k_seleccionado_por_metricas"] = selection["k_seleccionado"]
                selection["k_seleccionado"] = int(k)
            self.selection = selection
            self.evaluation["seleccion_k"] = selection
        else:
            self.selection = {"k_seleccionado": int(k or 4),
                              "criterio": "k fijado por el usuario"}
            self.evaluation["seleccion_k"] = self.selection

        k = int(k or self.selection["k_seleccionado"])
        self.kmeans_model = KMeansModel(n_clusters=k)
        self.labels = self.kmeans_model.fit_predict(X)
        self.features["Cluster"] = self.labels
        self.kmeans_model.print_inertia_analysis(X)
        self.evaluation["metricas_finales"] = final_metrics(X, self.labels)
        return self.selection

    # --------------------------------------------------------------- labeling
    def assign_segment_labels(self) -> Dict[int, str]:
        """Asigna etiquetas únicas usando valor, recencia y desempates."""
        if self.labels is None:
            raise ValueError("Ejecute fit() primero.")

        profile = self.features.groupby("Cluster")[self.clustering_vars].mean()
        overall = self.features[self.clustering_vars].mean()
        relative = (profile / overall.replace(0, np.nan)).fillna(0)

        base_labels: Dict[int, str] = {}
        for cluster, row in relative.iterrows():
            high_value = row["Ingreso_Neto"] >= 1.0
            recent = row["Dias_Ultima_Compra"] <= 1.0
            if high_value and recent:
                base = "VIP Leales"
            elif high_value and not recent:
                base = "VIP en Riesgo"
            elif not high_value and recent:
                base = "Potenciales de Crecimiento"
            else:
                base = "Ocasionales Inactivos"
            base_labels[int(cluster)] = base

        final: Dict[int, str] = {}
        used: Counter = Counter()

        # Agrupar clusters que comparten cuadrante para diferenciarlos por rasgo.
        groups: Dict[str, List[int]] = {}
        for cluster, base in base_labels.items():
            groups.setdefault(base, []).append(cluster)

        def _unique(label: str, cluster: int) -> str:
            if used[label] == 0:
                used[label] += 1
                return label
            candidate = f"{label} - C{cluster}"
            used[candidate] += 1
            return candidate

        for base, clusters in groups.items():
            if len(clusters) == 1:
                final[clusters[0]] = _unique(base, clusters[0])
                continue

            if base == "VIP Leales":
                ordered = sorted(
                    clusters,
                    key=lambda c: relative.loc[c, "Frecuencia"],
                    reverse=True,
                )
                if relative.loc[ordered[0], "Frecuencia"] >= 1.5:
                    final[ordered[0]] = _unique("VIP Elite", ordered[0])
                    remaining_label = "VIP Leales"
                else:
                    final[ordered[0]] = _unique("VIP Leales", ordered[0])
                    remaining_label = "VIP Leales (Ticket Alto)"
                for cluster in ordered[1:]:
                    final[cluster] = _unique(remaining_label, cluster)

            elif base == "Ocasionales Inactivos":
                ordered = sorted(
                    clusters,
                    key=lambda c: relative.loc[c, "Descuento_Promedio"],
                    reverse=True,
                )
                final[ordered[0]] = _unique("Cazadores de Ofertas", ordered[0])
                for cluster in ordered[1:]:
                    final[cluster] = _unique("Ocasionales Inactivos", cluster)

            else:
                for cluster in clusters:
                    final[cluster] = _unique(base, cluster)

        if len(set(final.values())) != len(final):
            raise RuntimeError(f"Etiquetas duplicadas: {final}")

        self.label_map = final
        self.features["Etiqueta"] = self.features["Cluster"].map(final)
        return final

    # --------------------------------------------------------------- profiles
    def profile_table(self) -> pd.DataFrame:
        """Tabla de perfil por segmento, ordenada por ingreso."""
        if "Etiqueta" not in self.features.columns:
            self.assign_segment_labels()

        grouped = self.features.groupby(["Cluster", "Etiqueta"], observed=True)
        rows = []
        total_revenue = float(self.features["Ingreso_Neto"].sum())
        total_customers = len(self.features)

        for (cluster, label), group in grouped:
            row = {
                "Cluster": int(cluster),
                "Etiqueta": label,
                "Clientes": int(len(group)),
                "Pct_Clientes": round(100 * len(group) / total_customers, 2),
                "Ingreso_Neto": round(float(group["Ingreso_Neto"].sum()), 2),
                "Participacion_Ingresos": round(
                    100 * float(group["Ingreso_Neto"].sum()) / total_revenue, 2
                ) if total_revenue else 0.0,
                "Ticket_Promedio": round(float(group["Ticket_Promedio"].mean()), 2),
                "Frecuencia": round(float(group["Frecuencia"].mean()), 2),
                "Descuento_Promedio": round(float(group["Descuento_Promedio"].mean()), 4),
                "Dias_Ultima_Compra": round(float(group["Dias_Ultima_Compra"].mean()), 1),
                "Ingreso_Anualizado": round(float(group["Ingreso_Anualizado"].mean()), 2),
                "R_Score": round(float(group["R_Score"].mean()), 2),
                "F_Score": round(float(group["F_Score"].mean()), 2),
                "M_Score": round(float(group["M_Score"].mean()), 2),
                "Categoria_Favorita": group["Categoria_Favorita"].mode().iloc[0],
                "Marca_Favorita": group["Marca_Favorita"].mode().iloc[0],
                "Ciudad_Principal": group["Ciudad"].mode().iloc[0],
                "Metodo_Pago_Principal": group["Metodo_Pago_Favorito"].mode().iloc[0],
            }
            rows.append(row)

        table = pd.DataFrame(rows).sort_values("Ingreso_Neto", ascending=False)
        return table.reset_index(drop=True)

    def segment_summary(self) -> List[Dict[str, Any]]:
        return self.profile_table().to_dict(orient="records")

    def validate(self, n_bootstrap: int = 10) -> Dict[str, Any]:
        """Añade estabilidad bootstrap al informe de evaluación."""
        X = self.X.values
        k = int(self.kmeans_model.n_clusters)
        stability = bootstrap_stability(X, k, n_iter=n_bootstrap)
        self.evaluation["estabilidad"] = stability
        self.evaluation["etiquetas"] = {str(k): v for k, v in self.label_map.items()}
        self.evaluation["perfil"] = self.segment_summary()
        return self.evaluation

    # ------------------------------------------------------------------ plots
    def plot_pca_clusters(self, pca_model, name: str = "clusters_pca") -> str:
        """Dispersión PCA con centroides, etiquetas y círculo de correlaciones."""
        if self.labels is None:
            raise ValueError("Ejecute fit() primero.")
        X_pca = pca_model._X_pca
        var = pca_model.explained_variance_ratio
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("tab10", len(self.label_map))

        for cluster in sorted(self.label_map):
            mask = self.labels == cluster
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.45,
                       color=cmap(cluster), label=self.label_map[cluster])

        centroids_pca = pca_model.pca.transform(self.kmeans_model.centroids)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker="X", s=220,
                   color="black", edgecolor="white", linewidth=1.2,
                   label="Centroides", zorder=5)

        loadings = pca_model.get_loadings(self.clustering_vars)
        limit = max(np.abs(X_pca[:, :2]).max(), 1e-6) * 0.95
        ax.add_patch(Circle((0, 0), radius=limit * 0.55, fill=False, color="gray",
                            linestyle="--", linewidth=1, alpha=0.7))
        for var_name in loadings.index:
            x = loadings.loc[var_name, "PC1"] * limit * 0.55
            y = loadings.loc[var_name, "PC2"] * limit * 0.55
            ax.arrow(0, 0, x, y, color="#333333", width=0.0,
                     head_width=limit * 0.035, length_includes_head=True, alpha=0.8)
            ax.text(x * 1.08, y * 1.08, var_name, fontsize=8, ha="center", va="center")

        ax.axhline(0, color="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.2f}%)")
        ax.set_title("Segmentos de clientes en el plano PCA")
        ax.legend(fontsize=8, markerscale=2, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return save_figure(fig, name)

    def plot_radar(self, name: str = "perfil_radar") -> str:
        """Radar normalizado de los perfiles de segmento."""
        if "Etiqueta" not in self.features.columns:
            self.assign_segment_labels()
        means = self.features.groupby("Etiqueta")[self.clustering_vars].mean()
        normalized = (means - means.min()) / (means.max() - means.min()).replace(0, 1)

        angles = np.linspace(0, 2 * np.pi, len(self.clustering_vars), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        colors = plt.get_cmap("tab10", len(normalized))

        for i, (label, row) in enumerate(normalized.iterrows()):
            values = row.tolist() + [row.tolist()[0]]
            ax.plot(angles, values, linewidth=2, label=label, color=colors(i))
            ax.fill(angles, values, alpha=0.12, color=colors(i))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.clustering_vars, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title("Perfil relativo de cada segmento (0 = mínimo, 1 = máximo)", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        fig.tight_layout()
        return save_figure(fig, name)

    def plot_segment_value(self, name: str = "segmentos_valor") -> str:
        """Barras comparadas: % de clientes vs % de ingreso por segmento."""
        table = self.profile_table().sort_values("Participacion_Ingresos", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(table))
        height = 0.38
        ax.barh(y - height / 2, table["Pct_Clientes"], height=height,
                label="% de clientes", color="#4c72b0")
        ax.barh(y + height / 2, table["Participacion_Ingresos"], height=height,
                label="% de ingreso", color="#dd8452")
        for i, (clients, revenue) in enumerate(zip(table["Pct_Clientes"], table["Participacion_Ingresos"])):
            ax.text(clients + 0.5, i - height / 2, f"{clients:.1f}%", va="center", fontsize=8)
            ax.text(revenue + 0.5, i + height / 2, f"{revenue:.1f}%", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(table["Etiqueta"], fontsize=9)
        ax.set_xlabel("Porcentaje (%)")
        ax.set_title("Concentración de clientes e ingreso por segmento")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return save_figure(fig, name)

    def plot_profile_heatmap(self, name: str = "perfil_relativo") -> str:
        """Mapa de calor con el índice relativo de cada variable por segmento."""
        table = self.profile_table().set_index("Etiqueta")
        global_means = self.features[self.clustering_vars].mean()
        relative = (table[self.clustering_vars] / global_means.replace(0, np.nan)).astype(float)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(relative.values, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.6)
        ax.set_xticks(range(len(relative.columns)))
        ax.set_xticklabels(relative.columns, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(relative.index)))
        ax.set_yticklabels(relative.index, fontsize=9)
        for i in range(relative.shape[0]):
            for j in range(relative.shape[1]):
                ax.text(j, i, f"{relative.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=8)
        ax.set_title("Índice relativo al promedio global (1.00 = promedio)")
        fig.colorbar(im, ax=ax, label="Índice")
        fig.tight_layout()
        return save_figure(fig, name)

    # ----------------------------------------------------------------- export
    def save_segments(self, path: Optional[str] = None) -> str:
        target = Path(path or RESULTS_DIR / "segmentos.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k": int(self.kmeans_model.n_clusters),
            "etiquetas": {str(k): v for k, v in self.label_map.items()},
            "segmentos": self.segment_summary(),
            "evaluacion": self.evaluation,
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)

    def save_customers(self, path: Optional[str] = None) -> str:
        target = Path(path or RESULTS_DIR / "clientes_segmentados.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(target, index=True, encoding="utf-8-sig")
        return str(target)


def segment_customers(features: pd.DataFrame, model_matrix: pd.DataFrame,
                      k: Optional[int] = None) -> CustomerSegmenter:
    segmenter = CustomerSegmenter(features, model_matrix)
    segmenter.fit(k=k)
    segmenter.assign_segment_labels()
    segmenter.validate()
    return segmenter
