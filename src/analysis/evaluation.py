"""
Evaluación del agrupamiento (fases 5 y 6 de KDD).

La inercia siempre disminuye al aumentar *k*, por lo que no sirve por sí sola
para elegir el número de clusters. Este módulo combina:

- Método del codo (distancia a la cuerda sobre la curva de inercia).
- Coeficiente de silueta (mayor es mejor).
- Índice Davies-Bouldin (menor es mejor).
- Índice Calinski-Harabasz (mayor es mejor).
- Tamaño mínimo de cluster (criterio de accionabilidad).
- Estabilidad por remuestreo bootstrap medida con ARI.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ..config.settings import (
    BOOTSTRAP_FRACTION,
    K_MAX_ACTIONABLE,
    K_MIN_ACTIONABLE,
    MIN_CLUSTER_SHARE,
    MIN_CLUSTER_SHARE_HARD,
    N_BOOTSTRAP_STABILITY,
    RANDOM_STATE,
    SILHOUETTE_SAMPLE_SIZE,
)


def evaluate_k(X: np.ndarray, k_range: Iterable[int],
               random_state: int = RANDOM_STATE,
               n_init: int = 10,
               sample_size: int = SILHOUETTE_SAMPLE_SIZE) -> pd.DataFrame:
    """Evalúa varios valores de k con métricas internas y de accionabilidad."""
    X = np.asarray(X)
    n = len(X)
    sample_size = min(sample_size, n)

    rows: List[Dict[str, Any]] = []
    for k in k_range:
        model = KMeans(
            n_clusters=k, init="k-means++", n_init=n_init,
            max_iter=300, random_state=random_state,
        )
        labels = model.fit_predict(X)
        sizes = np.bincount(labels, minlength=k)
        shares = sizes / n

        silhouette = silhouette_score(
            X, labels, sample_size=sample_size, random_state=random_state
        ) if 1 < k < n else float("nan")

        rows.append({
            "k": int(k),
            "inercia": round(float(model.inertia_), 4),
            "silhouette": round(float(silhouette), 6),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 6),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 4),
            "cluster_min_pct": round(float(shares.min()) * 100, 3),
            "cluster_max_pct": round(float(shares.max()) * 100, 3),
            "clusters_menores_5pct": int((shares < MIN_CLUSTER_SHARE).sum()),
            "tamanos": [int(s) for s in sizes],
        })

    table = pd.DataFrame(rows)
    table["delta_inercia"] = table["inercia"].diff().round(4)
    table["delta_inercia_pct"] = (
        table["inercia"].pct_change() * 100
    ).round(3)
    return table


def elbow_point(k_values: Iterable[int], inertia: Iterable[float]) -> Dict[str, Any]:
    """Estima el codo como el punto de mayor distancia a la cuerda normalizada."""
    k = np.asarray(list(k_values), dtype=float)
    y = np.asarray(list(inertia), dtype=float)
    if len(k) < 3:
        return {"k_codo": int(k[0]) if len(k) else None, "metodo": "insuficientes puntos"}

    kx = (k - k.min()) / (k.max() - k.min())
    yx = (y - y.min()) / (y.max() - y.min())
    # Distancia vertical a la recta que une el primer y el último punto.
    distances = np.abs(yx - (1 - kx))
    idx = int(np.argmax(distances))
    return {
        "k_codo": int(k[idx]),
        "distancia_maxima": round(float(distances[idx]), 6),
        "metodo": "máxima distancia a la cuerda (curva de inercia normalizada)",
    }


def select_k(table: pd.DataFrame,
             k_min: int = K_MIN_ACTIONABLE,
             k_max: int = K_MAX_ACTIONABLE,
             min_share: float = MIN_CLUSTER_SHARE,
             min_share_hard: float = MIN_CLUSTER_SHARE_HARD) -> Dict[str, Any]:
    """
    Selecciona k con una regla explícita y auditable.

    Regla: entre los k accionables (3-6) que cumplen tamaño mínimo de cluster,
    se elige el de mayor silueta. Si hay empate, se prefiere el menor
    Davies-Bouldin. Si ningún k cumple el 5 %, se relaja a un mínimo técnico del
    1 % y se deja constancia de la relajación en ``criterio``.
    """
    in_range = table[table["k"].between(k_min, k_max)].copy()
    eligible = in_range[
        (in_range["cluster_min_pct"] >= min_share * 100)
        & (in_range["silhouette"] > 0)
    ].copy()

    relaxed = False
    applied_share = min_share
    fallback_all = False
    if eligible.empty:
        eligible = in_range[
            (in_range["cluster_min_pct"] >= min_share_hard * 100)
            & (in_range["silhouette"] > 0)
        ].copy()
        relaxed = True
        applied_share = min_share_hard

    if eligible.empty:
        eligible = table[table["silhouette"] > 0].copy()
        relaxed = True
        applied_share = 0.0
        fallback_all = True

    if eligible.empty:
        raise ValueError("No hay valores de k con silhouette positivo para seleccionar.")

    eligible = eligible.sort_values(
        ["silhouette", "davies_bouldin"], ascending=[False, True]
    )
    best = eligible.iloc[0]
    runner_up = eligible.iloc[1] if len(eligible) > 1 else None

    if fallback_all:
        criterio = (
            "Mayor silhouette entre todos los k evaluados: no hubo ninguna "
            f"solución en el rango {k_min}..{k_max}. Desempate por menor "
            "Davies-Bouldin."
        )
    elif relaxed:
        criterio = (
            f"Mayor silhouette entre k={k_min}..{k_max}. Ningún k alcanzó el "
            f"{min_share:.0%} mínimo por cluster; la restricción se relajó al "
            f"{applied_share:.0%} técnico y se documenta. Desempate por menor "
            "Davies-Bouldin."
        )
    else:
        criterio = (
            f"Mayor silhouette entre k={k_min}..{k_max} con todos los clusters "
            f">= {min_share:.0%} de la base; desempate por menor Davies-Bouldin."
        )

    return {
        "k_seleccionado": int(best["k"]),
        "criterio": criterio,
        "silhouette": float(best["silhouette"]),
        "davies_bouldin": float(best["davies_bouldin"]),
        "calinski_harabasz": float(best["calinski_harabasz"]),
        "cluster_min_pct": float(best["cluster_min_pct"]),
        "restriccion_relajada": relaxed,
        "umbral_minimo_aplicado_pct": applied_share * 100,
        "umbral_preferido_pct": min_share * 100,
        "runner_up": None if runner_up is None else {
            "k": int(runner_up["k"]),
            "silhouette": float(runner_up["silhouette"]),
            "davies_bouldin": float(runner_up["davies_bouldin"]),
        },
        "k_evaluados": [int(v) for v in table["k"].tolist()],
        "tabla": table.to_dict(orient="records"),
    }


def final_metrics(X: np.ndarray, labels: np.ndarray,
                  sample_size: int = SILHOUETTE_SAMPLE_SIZE) -> Dict[str, Any]:
    """Calcula las métricas internas para una solución final."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    sample_size = min(sample_size, len(X))
    return {
        "n_clusters": int(len(sizes)),
        "silhouette": round(float(silhouette_score(
            X, labels, sample_size=sample_size, random_state=RANDOM_STATE
        )), 6),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 4),
        "tamanos": [int(s) for s in sizes],
        "participacion": [round(float(s / len(X)) * 100, 3) for s in sizes],
    }


def bootstrap_stability(X: np.ndarray, k: int,
                        n_iter: int = N_BOOTSTRAP_STABILITY,
                        fraction: float = BOOTSTRAP_FRACTION,
                        random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Mide estabilidad: entrena con una submuestra y evalúa ARI en las filas no vistas.

    Un ARI cercano a 1 indica que la asignación de segmentos es reproducible;
    cercano a 0 indica que los segmentos cambian con la muestra.
    """
    X = np.asarray(X)
    n = len(X)
    reference_model = KMeans(
        n_clusters=k, init="k-means++", n_init=20,
        max_iter=300, random_state=random_state,
    )
    reference = reference_model.fit_predict(X)

    rng = np.random.default_rng(random_state)
    scores: List[float] = []
    sample_n = max(int(n * fraction), k + 1)

    for _ in range(n_iter):
        idx = rng.choice(n, size=sample_n, replace=False)
        held = np.setdiff1d(np.arange(n), idx)
        model = KMeans(
            n_clusters=k, init="k-means++", n_init=10,
            max_iter=300, random_state=random_state,
        )
        model.fit(X[idx])
        predicted = model.predict(X[held])
        scores.append(float(adjusted_rand_score(reference[held], predicted)))

    scores_array = np.asarray(scores)
    return {
        "n_iteraciones": n_iter,
        "fraccion_entrenamiento": fraction,
        "ari_media": round(float(scores_array.mean()), 4),
        "ari_desv": round(float(scores_array.std(ddof=1)) if len(scores) > 1 else 0.0, 4),
        "ari_min": round(float(scores_array.min()), 4),
        "ari_max": round(float(scores_array.max()), 4),
        "valores": [round(float(s), 4) for s in scores],
        "interpretacion": (
            "Estable" if scores_array.mean() >= 0.70 else
            "Moderadamente estable" if scores_array.mean() >= 0.50 else
            "Inestable: los segmentos dependen de la muestra"
        ),
    }
