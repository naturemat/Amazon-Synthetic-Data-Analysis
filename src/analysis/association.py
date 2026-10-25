"""
Análisis de asociación como complemento exploratorio (fase 5 de KDD).

El caso de la actividad sugiere reglas de asociación ("productos que se compran
juntos"). Antes de aplicarlas se comprueba la estructura del dataset: cada
pedido contiene exactamente un producto, por lo que **no existen canastas** y
Apriori clásico no es aplicable. Ocultarlo sería un error metodológico.

Como alternativa honesta se calculan reglas de co-compra a nivel de cliente:
si un cliente compró A, ¿qué probabilidad hay de que también haya comprado B?
Se reportan soporte, confianza y lift, y se aclara que la unidad de análisis es
el cliente histórico, no el pedido.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.settings import (
    ASSOCIATION_MIN_LIFT,
    ASSOCIATION_MIN_SUPPORT,
    ASSOCIATION_TOP_N,
    RESULTS_DIR,
)
from ..core.plotting import save_figure


class MarketBasketAnalyzer:
    """Diagnostica canastas y, si no existen, calcula co-compra por cliente."""

    def __init__(self, sales_df: pd.DataFrame):
        self.df = sales_df.copy()
        self.diagnostic: Dict[str, Any] = {}
        self.rules: Optional[pd.DataFrame] = None

    def diagnose_baskets(self) -> Dict[str, Any]:
        """Verifica si el dataset permite un análisis de canasta de mercado."""
        items = self.df.groupby("OrderID").size()
        self.diagnostic = {
            "pedidos": int(len(items)),
            "items_por_pedido_media": round(float(items.mean()), 4),
            "items_por_pedido_max": int(items.max()),
            "pedidos_con_mas_de_un_producto": int((items > 1).sum()),
            "apriori_aplicable": bool(items.max() > 1),
            "explicacion": (
                "Cada pedido contiene un único producto; no hay canastas que "
                "permitan reglas de asociación intra-pedido. Se justifica usar "
                "clustering de clientes y se complementa con co-compra histórica."
                if items.max() <= 1 else
                "Existen pedidos con varios productos; Apriori clásico es aplicable."
            ),
        }
        return self.diagnostic

    def customer_cooccurrence(self,
                              min_support: float = ASSOCIATION_MIN_SUPPORT,
                              min_lift: float = ASSOCIATION_MIN_LIFT,
                              top_n: int = ASSOCIATION_TOP_N) -> pd.DataFrame:
        """
        Calcula reglas A -> B a nivel de cliente.

        support(A,B) = clientes que compraron A y B / total de clientes
        confidence(A->B) = clientes que compraron A y B / clientes que compraron A
        lift(A->B) = confidence(A->B) / support(B)
        """
        self.diagnose_baskets()
        customer_product = (
            self.df.assign(value=1)
            .pivot_table(index="CustomerID", columns="ProductID", values="value",
                         aggfunc="max", fill_value=0)
            .astype(np.int8)
        )
        n_customers = len(customer_product)
        counts = customer_product.sum(axis=0).to_numpy()
        co = customer_product.T.to_numpy().astype(np.int32) @ customer_product.to_numpy().astype(np.int32)
        product_ids = customer_product.columns.to_numpy()
        names = (
            self.df.drop_duplicates("ProductID")
            .set_index("ProductID")["ProductName"]
            .to_dict()
        )

        rows: List[Dict[str, Any]] = []
        n_products = len(product_ids)
        for i, j in combinations(range(n_products), 2):
            pair_count = int(co[i, j])
            if pair_count == 0:
                continue
            support = pair_count / n_customers
            if support < min_support:
                continue
            for a, b in ((i, j), (j, i)):
                confidence = pair_count / counts[a] if counts[a] else 0.0
                support_b = counts[b] / n_customers if n_customers else 0.0
                lift = confidence / support_b if support_b else 0.0
                if lift >= min_lift:
                    rows.append({
                        "antecedente_id": product_ids[a],
                        "consecuente_id": product_ids[b],
                        "antecedente": names.get(product_ids[a], product_ids[a]),
                        "consecuente": names.get(product_ids[b], product_ids[b]),
                        "soporte": round(float(support), 5),
                        "confianza": round(float(confidence), 4),
                        "lift": round(float(lift), 4),
                        "clientes_con_ambos": pair_count,
                        "unidad": "cliente histórico (no pedido)",
                    })

        rules = pd.DataFrame(rows)
        if not rules.empty:
            rules = rules.sort_values(["lift", "soporte"], ascending=False).head(top_n)
        self.rules = rules
        return rules

    def plot_top_rules(self, name: str = "reglas_asociacion") -> Optional[str]:
        if self.rules is None or self.rules.empty:
            return None
        data = self.rules.head(10).iloc[::-1]
        labels = [f"{a[:22]} -> {b[:22]}" for a, b in zip(data["antecedente"], data["consecuente"])]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels, data["lift"], color="#55a868")
        for i, value in enumerate(data["lift"]):
            ax.text(value + 0.01, i, f"{value:.2f}", va="center", fontsize=8)
        ax.set_xlabel("Lift")
        ax.set_title("Principales reglas de co-compra por cliente")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return save_figure(fig, name)

    def save_results(self, path: Optional[str] = None) -> str:
        target = Path(path or RESULTS_DIR / "reglas_asociacion.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "diagnostico_canastas": self.diagnostic,
            "reglas": [] if self.rules is None else self.rules.to_dict(orient="records"),
            "advertencia": (
                "Las reglas se calculan sobre el historial del cliente, no sobre "
                "pedidos individuales, porque no existen canastas multi-producto."
            ),
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)
