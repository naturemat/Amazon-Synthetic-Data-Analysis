"""Pruebas de minería, evaluación y etiquetado (fases 5-7)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.association import MarketBasketAnalyzer
from src.analysis.evaluation import (
    bootstrap_stability,
    elbow_point,
    evaluate_k,
    final_metrics,
    select_k,
)
from src.clustering.kmeans import KMeansModel
from src.clustering.segmentation import CustomerSegmenter
from src.experiments.degradation import DegradationExperiment
from src.kdd.knowledge import build_knowledge_payload, build_playbook_markdown
from src.kdd.phase_logger import KDDLogger


def test_evaluate_k_returns_all_metrics(blobs):
    table = evaluate_k(blobs, range(2, 5), sample_size=100)
    assert list(table["k"]) == [2, 3, 4]
    for col in ["inercia", "silhouette", "davies_bouldin",
                "calinski_harabasz", "cluster_min_pct"]:
        assert col in table.columns
    assert table.loc[table["k"] == 2, "silhouette"].iloc[0] > 0.5


def test_select_k_respects_actionable_range(blobs):
    table = evaluate_k(blobs, range(2, 7), sample_size=100)
    selection = select_k(table)
    assert 3 <= selection["k_seleccionado"] <= 6
    assert selection["silhouette"] > 0
    assert "tabla" in selection


def test_select_k_documents_relaxation_when_share_constraint_unmet():
    table = pd.DataFrame({
        "k": [3, 4, 5],
        "inercia": [300.0, 200.0, 120.0],
        "silhouette": [0.20, 0.30, 0.40],
        "davies_bouldin": [1.2, 1.1, 0.9],
        "calinski_harabasz": [10.0, 12.0, 14.0],
        "cluster_min_pct": [1.0, 1.0, 1.0],
    })
    selection = select_k(table)
    assert selection["k_seleccionado"] == 5
    assert selection["restriccion_relajada"] is True
    assert selection["umbral_minimo_aplicado_pct"] == 1.0
    assert "relajó" in selection["criterio"]

    table_ok = table.copy()
    table_ok["cluster_min_pct"] = [6.0, 7.0, 8.0]
    selection_ok = select_k(table_ok)
    assert selection_ok["restriccion_relajada"] is False
    assert "relajó" not in selection_ok["criterio"]


def test_elbow_point_and_final_metrics(blobs):
    table = evaluate_k(blobs, range(2, 6), sample_size=100)
    elbow = elbow_point(table["k"], table["inercia"])
    assert elbow["k_codo"] in set(table["k"])

    model = KMeansModel(n_clusters=2).fit(blobs)
    metrics = final_metrics(blobs, model.labels, sample_size=100)
    assert metrics["n_clusters"] == 2
    assert metrics["silhouette"] > 0.5
    assert sum(metrics["tamanos"]) == len(blobs)


def test_bootstrap_stability_is_high_for_separated_blobs(blobs):
    result = bootstrap_stability(blobs, k=2, n_iter=3, fraction=0.8)
    assert 0.0 <= result["ari_media"] <= 1.0
    assert result["ari_media"] > 0.5


def test_segment_labels_are_unique_and_complete(cleaned_orders):
    from src.data.transformation import CustomerTransformer

    sales = cleaned_orders.analytical_data
    transformer = CustomerTransformer(sales)
    features = transformer.aggregate()
    matrix = transformer.prepare_model_features()

    segmenter = CustomerSegmenter(features, matrix)
    segmenter.fit(k=2, evaluate=False)
    labels = segmenter.assign_segment_labels()

    assert len(labels) == 2
    assert len(set(labels.values())) == 2
    assert set(segmenter.features["Etiqueta"]) == set(labels.values())
    table = segmenter.profile_table()
    assert table["Clientes"].sum() == len(features)


def test_degradation_requires_per_customer_labels(cleaned_orders):
    from src.data.transformation import CustomerTransformer

    sales = cleaned_orders.analytical_data
    transformer = CustomerTransformer(sales)
    features = transformer.aggregate()
    matrix = transformer.prepare_model_features()

    segmenter = CustomerSegmenter(features, matrix)
    segmenter.fit(k=2, evaluate=False)

    with pytest.raises(ValueError, match="una etiqueta por cliente"):
        DegradationExperiment(
            sales_df=sales,
            full_df=cleaned_orders.data,
            reference_features=features,
            reference_model=matrix,
            reference_labels=segmenter.assign_segment_labels(),
            k=2,
            cutoff=cleaned_orders.data["OrderDate"].max(),
        )


def test_association_diagnoses_single_item_orders(cleaned_orders):
    analyzer = MarketBasketAnalyzer(cleaned_orders.analytical_data)
    diagnostic = analyzer.diagnose_baskets()
    assert diagnostic["items_por_pedido_max"] == 1
    assert diagnostic["apriori_aplicable"] is False
    rules = analyzer.customer_cooccurrence(min_support=0.0)
    assert isinstance(rules, pd.DataFrame)


def test_kdd_logger_exports_json_and_markdown(tmp_path):
    logger = KDDLogger()
    with logger.phase(1) as phase:
        phase.metric("pregunta", "¿Cómo se aplican las fases de KDD?")
        phase.decision("Se define el objetivo de negocio.")
    with logger.phase(2) as phase:
        phase.metric("filas", 100)
        phase.warn("Dataset simulado.")

    paths = logger.export(
        json_path=str(tmp_path / "kdd_log.json"),
        markdown_path=str(tmp_path / "kdd.md"),
    )
    payload = json.loads((tmp_path / "kdd_log.json").read_text(encoding="utf-8"))
    assert len(payload["phases"]) == 2
    assert payload["phases"][0]["metrics"]["pregunta"]
    assert "Fase 2" in (tmp_path / "kdd.md").read_text(encoding="utf-8")
    assert paths["json"].endswith("kdd_log.json")


def test_knowledge_layer_produces_actions_and_impact():
    segments = [
        {"Etiqueta": "VIP Leales", "Clientes": 10, "Pct_Clientes": 10.0,
         "Ingreso_Neto": 1000.0, "Participacion_Ingresos": 50.0,
         "Ticket_Promedio": 100.0, "Frecuencia": 5.0,
         "Dias_Ultima_Compra": 10.0},
        {"Etiqueta": "Cazadores de Ofertas", "Clientes": 90, "Pct_Clientes": 90.0,
         "Ingreso_Neto": 1000.0, "Participacion_Ingresos": 50.0,
         "Ticket_Promedio": 20.0, "Frecuencia": 1.0,
         "Dias_Ultima_Compra": 300.0},
    ]
    payload = build_knowledge_payload(segments, "2024-12-29", "Segmentar clientes.")
    assert payload["impacto_estimado_total_usd"] > 0
    assert len(payload["segmentos"]) == 2
    markdown = build_playbook_markdown(payload)
    assert "VIP Leales" in markdown
    assert "KPI" in markdown
