"""
Experimento controlado de degradación (evidencia para la fase 6 y la discusión).

Pregunta: ¿qué pasa con los segmentos si se omite o se ejecuta mal una fase?

Se construye una solución de referencia con el pipeline completo (limpieza,
filtro de ventas realizadas, recencia con fecha de corte, recorte de colas,
estandarización y k elegido con métricas). Después se introducen errores
controlados, se repite el agrupamiento con el mismo k y se mide el Adjusted Rand
Index (ARI) contra la referencia:

- ARI = 1.0 → mismos segmentos.
- ARI ≈ 0 → segmentos sin relación.

Esto convierte la afirmación "omitir fases degrada el resultado" en evidencia
cuantificada y no en una opinión.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from ..config.settings import (
    DEGRADATION_DUPLICATE_RATE,
    DEGRADATION_NOISE_RATE,
    DEGRADATION_NULL_RATE,
    DEGRADATION_SEEDS,
    RANDOM_STATE,
    RESULTS_DIR,
    STABILITY_ARI_THRESHOLD,
)
from ..core.plotting import save_figure
from ..data.transformation import CustomerTransformer


def _ensure_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza las columnas de ingreso usadas por la agregación."""
    out = df.copy()
    if "Ingreso_Neto" not in out.columns:
        out["Ingreso_Bruto"] = out["Quantity"] * out["UnitPrice"]
        out["Ingreso_Neto"] = out["Ingreso_Bruto"] * (1 - out["Discount"])
        out["Ingreso_Facturado"] = (
            out["Ingreso_Neto"] + out["Tax"] + out["ShippingCost"]
        )
    return out


def _features_for(df: pd.DataFrame, cutoff: pd.Timestamp,
                  winsorize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transformer = CustomerTransformer(_ensure_revenue(df), cutoff_date=cutoff)
    features = transformer.aggregate()
    scaled = transformer.prepare_model_features(winsorize=winsorize)
    values = transformer.model_values
    return features, scaled, values


class DegradationExperiment:
    """Ejecuta escenarios de error y reporta su impacto en los segmentos."""

    def __init__(self, sales_df: pd.DataFrame, full_df: pd.DataFrame,
                 reference_features: pd.DataFrame,
                 reference_model: pd.DataFrame,
                 reference_labels: np.ndarray,
                 k: int,
                 cutoff: pd.Timestamp,
                 seeds: Optional[List[int]] = None):
        self.sales = _ensure_revenue(sales_df)
        self.full = _ensure_revenue(full_df)
        self.reference_features = reference_features
        self.reference_model = reference_model
        reference_labels = np.asarray(reference_labels)
        if reference_labels.ndim != 1 or len(reference_labels) != len(reference_features):
            raise ValueError(
                "reference_labels debe ser un vector con una etiqueta por cliente "
                f"(recibido: forma {reference_labels.shape}, clientes "
                f"{len(reference_features)})."
            )
        self.reference_labels = reference_labels
        self.k = int(k)
        self.cutoff = pd.Timestamp(cutoff)
        self.seeds = seeds or DEGRADATION_SEEDS
        self.rows: List[Dict[str, Any]] = []
        self.summary: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------- scenarios
    def _duplicates(self, seed: int) -> pd.DataFrame:
        sample = self.sales.sample(
            frac=DEGRADATION_DUPLICATE_RATE, random_state=seed
        )
        return pd.concat([self.sales, sample], ignore_index=True)

    def _nulls_zero(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = self.sales.copy()
        columns = ["Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost"]
        for col in columns:
            out[col] = out[col].astype(float)
            mask = rng.random(len(out)) < DEGRADATION_NULL_RATE
            out.loc[mask, col] = np.nan
        out[columns] = out[columns].fillna(0)
        out["Ingreso_Neto"] = out["Quantity"] * out["UnitPrice"] * (1 - out["Discount"])
        out["Ingreso_Facturado"] = out["Ingreso_Neto"] + out["Tax"] + out["ShippingCost"]
        return out

    def _all_statuses(self, seed: int) -> pd.DataFrame:
        # No se filtra por estado: se incluyen cancelados, devueltos y pendientes.
        return self.full.copy()

    def _noise(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = self.sales.copy()
        out["UnitPrice"] = out["UnitPrice"].astype(float)
        mask = rng.random(len(out)) < DEGRADATION_NOISE_RATE
        factor = rng.uniform(1 - DEGRADATION_NOISE_RATE, 1 + DEGRADATION_NOISE_RATE, mask.sum())
        out.loc[mask, "UnitPrice"] = out.loc[mask, "UnitPrice"] * factor
        out["Ingreso_Neto"] = out["Quantity"] * out["UnitPrice"] * (1 - out["Discount"])
        out["Ingreso_Facturado"] = out["Ingreso_Neto"] + out["Tax"] + out["ShippingCost"]
        return out

    # ---------------------------------------------------------------- runner
    def _ari(self, features: pd.DataFrame, labels: np.ndarray) -> Tuple[float, int]:
        ref = pd.Series(self.reference_labels, index=self.reference_features.index)
        common = ref.index.intersection(features.index)
        if len(common) < self.k:
            return float("nan"), len(common)
        labels_series = pd.Series(labels, index=features.index)
        return float(adjusted_rand_score(
            ref.loc[common], labels_series.loc[common]
        )), len(common)

    def _run_raw_scenario(self, name: str, description: str,
                          builder) -> None:
        for seed in self.seeds:
            df = builder(seed)
            try:
                features, scaled, _ = _features_for(df, self.cutoff)
                labels = KMeans(
                    n_clusters=self.k, init="k-means++", n_init=10,
                    max_iter=300, random_state=seed,
                ).fit_predict(scaled.values)
                ari, n_common = self._ari(features, labels)
                self.rows.append({
                    "escenario": name,
                    "descripcion": description,
                    "semilla": seed,
                    "ari": round(ari, 4) if not np.isnan(ari) else None,
                    "clientes_evaluados": int(n_common),
                    "fase_afectada": self._phase_of(name),
                })
            except Exception as exc:  # pragma: no cover - defensivo
                self.rows.append({
                    "escenario": name,
                    "descripcion": description,
                    "semilla": seed,
                    "ari": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fase_afectada": self._phase_of(name),
                })

    def _phase_of(self, name: str) -> str:
        mapping = {
            "Duplicados 5% (limpieza omitida)": "Fase 3. Preprocesamiento",
            "Nulos 10% imputados con 0": "Fase 3. Preprocesamiento",
            "Sin filtrar estados no realizados": "Fase 3. Preprocesamiento",
            "Ruido ±20% en precio (20% filas)": "Fase 3-4. Calidad y transformación",
            "Sin estandarización": "Fase 4. Transformación",
        }
        return mapping.get(name, "Fase 5. Minería")

    def run(self) -> pd.DataFrame:
        self._run_raw_scenario(
            "Duplicados 5% (limpieza omitida)",
            "Se duplican aleatoriamente 5% de las filas y se agregan sin depurar.",
            self._duplicates,
        )
        self._run_raw_scenario(
            "Nulos 10% imputados con 0",
            "Se reemplazan 10% de valores numéricos por nulos y luego por cero.",
            self._nulls_zero,
        )
        self._run_raw_scenario(
            "Sin filtrar estados no realizados",
            "Se incluyen pedidos cancelados, devueltos y pendientes en ingreso y recencia.",
            self._all_statuses,
        )
        self._run_raw_scenario(
            "Ruido ±20% en precio (20% filas)",
            "Se perturba el precio unitario en 20% de las filas antes de agregar.",
            self._noise,
        )

        # Escenario de transformación: mismos datos, sin estandarizar.
        for seed in self.seeds:
            features, _, values = _features_for(self.sales, self.cutoff)
            labels = KMeans(
                n_clusters=self.k, init="k-means++", n_init=10,
                max_iter=300, random_state=seed,
            ).fit_predict(values.values)
            ari, n_common = self._ari(features, labels)
            self.rows.append({
                "escenario": "Sin estandarización",
                "descripcion": "K-Means sobre variables en su escala original; el ingreso domina la distancia.",
                "semilla": seed,
                "ari": round(ari, 4),
                "clientes_evaluados": int(n_common),
                "fase_afectada": "Fase 4. Transformación",
            })

        # Escenario de minería: k fijado sin evaluación (k=2).
        for seed in self.seeds:
            labels = KMeans(
                n_clusters=2, init="k-means++", n_init=10,
                max_iter=300, random_state=seed,
            ).fit_predict(self.reference_model.values)
            ari, n_common = self._ari(self.reference_features, labels)
            self.rows.append({
                "escenario": "k=2 fijado sin evaluación",
                "descripcion": "Se elige k=2 a priori en lugar del k seleccionado con métricas.",
                "semilla": seed,
                "ari": round(ari, 4),
                "clientes_evaluados": int(n_common),
                "fase_afectada": "Fase 5. Minería",
            })

        results = pd.DataFrame(self.rows)
        valid = results.dropna(subset=["ari"])
        self.summary = (
            valid.groupby(["escenario", "fase_afectada"])["ari"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
            .rename(columns={"mean": "ari_media", "std": "ari_desv",
                             "min": "ari_min", "max": "ari_max",
                             "count": "repeticiones"})
            .sort_values("ari_media", ascending=False)
        )
        self.summary["ari_media"] = self.summary["ari_media"].round(4)
        self.summary["ari_desv"] = self.summary["ari_desv"].round(4)
        return results

    # ------------------------------------------------------------------ plots
    def plot(self, name: str = "experimento_degradacion") -> str:
        if self.summary is None:
            raise ValueError("Ejecute run() primero.")
        data = self.summary.sort_values("ari_media", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        colors = ["#c44e52" if v < STABILITY_ARI_THRESHOLD else "#55a868"
                  for v in data["ari_media"]]
        ax.barh(data["escenario"], data["ari_media"], color=colors,
                xerr=data["ari_desv"].fillna(0), capsize=4)
        ax.axvline(STABILITY_ARI_THRESHOLD, color="gray", linestyle="--",
                   label=f"Umbral de estabilidad ({STABILITY_ARI_THRESHOLD:.0%})")
        ax.axvline(1.0, color="black", linestyle=":", label="Referencia (ARI = 1)")
        for i, value in enumerate(data["ari_media"]):
            ax.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("ARI medio frente a la solución de referencia")
        ax.set_title("Impacto de omitir o ejecutar mal una fase de KDD")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return save_figure(fig, name)

    def save_results(self, path: Optional[str] = None,
                     plot_path: Optional[str] = None) -> str:
        if self.summary is None:
            raise ValueError("Ejecute run() primero.")
        target = Path(path or RESULTS_DIR / "experimento_degradacion.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pregunta": "¿Cuánto se degradan los segmentos si se omite o ejecuta mal una fase?",
            "metrica": "Adjusted Rand Index (ARI) contra la solución de referencia",
            "k_referencia": self.k,
            "fecha_corte": self.cutoff.date().isoformat(),
            "umbral_estabilidad": STABILITY_ARI_THRESHOLD,
            "referencia_ari": 1.0,
            "resumen": self.summary.to_dict(orient="records"),
            "detalle": self.rows,
            "figura": plot_path,
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)
