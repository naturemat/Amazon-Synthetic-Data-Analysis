"""
Punto de entrada del proyecto: aplicación práctica de las fases de KDD.

Ejecuta las siete fases de Fayyad et al. (1996) sobre el dataset de ventas,
registra métricas y decisiones por fase, genera figuras y deja la evidencia en
``docs/`` para el ensayo académico.

Uso
---
    python main.py                     # pipeline completo + experimento
    python main.py --no-experiment     # omite el experimento de degradación
    python main.py --k 4               # fija k (solo para pruebas)
    python main.py --quick             # evaluación reducida (desarrollo)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pipeline batch: no abrir ventanas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.association import MarketBasketAnalyzer
from src.clustering.pca import PCAModel
from src.clustering.segmentation import CustomerSegmenter
from src.config.settings import (
    BUSINESS_OBJECTIVE,
    BUSINESS_QUESTION,
    CLUSTERING_VARS,
    DATASET,
    FIGURES_DIR,
    KDD_OUTPUT_DIR,
    N_BOOTSTRAP_STABILITY,
    RESULTS_DIR,
    STABILITY_ARI_THRESHOLD,
    SUCCESS_CRITERIA,
)
from src.core.formatters import configure_matplotlib_style, configure_pandas_display
from src.core.plotting import save_figure
from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataLoader
from src.data.schema import to_markdown
from src.data.transformation import CustomerTransformer
from src.experiments.degradation import DegradationExperiment
from src.kdd.knowledge import build_knowledge_payload, build_playbook_markdown
from src.kdd.phase_logger import KDDLogger
from src.reports.kdd_report import KDDReport


def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _ensure_dirs() -> None:
    Path(KDD_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def _descriptive_eda(full: pd.DataFrame, sales: pd.DataFrame) -> dict:
    """Estadística descriptiva y dos figuras de contexto."""
    _section("ANÁLISIS DESCRIPTIVO (apoyo a fases 2 y 3)")
    numeric = sales[["Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost",
                     "Ingreso_Neto", "Ingreso_Facturado"]].describe().T
    print(numeric.round(3))

    stats = {
        "filas_ventas_realizadas": int(len(sales)),
        "clientes": int(sales["CustomerID"].nunique()),
        "ingreso_neto_total": round(float(sales["Ingreso_Neto"].sum()), 2),
        "ticket_neto_medio": round(float(sales["Ingreso_Neto"].mean()), 2),
        "descuento_medio": round(float(sales["Discount"].mean()), 4),
        "estados": {str(k): int(v) for k, v in full["OrderStatus"].value_counts().items()},
        "estadisticas_numericas": {
            col: {stat: round(float(value), 4) for stat, value in numeric.loc[col].items()}
            for col in numeric.index
        },
    }
    (Path(RESULTS_DIR) / "estadisticas_descriptivas.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Serie mensual de ingreso neto.
    monthly = sales.set_index("OrderDate")["Ingreso_Neto"].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly.index, monthly.values, color="#1f77b4", linewidth=1.8)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#1f77b4")
    ax.set_title("Ingreso neto mensual (ventas realizadas)")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "serie_ingreso_mensual")

    # Top categorías por ingreso neto.
    cat = sales.groupby("Category", observed=True)["Ingreso_Neto"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cat.index.astype(str), cat.values, color="#4c72b0")
    ax.set_title("Ingreso neto por categoría")
    ax.set_xlabel("USD")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "ingreso_por_categoria")
    return stats


def run_pipeline(k: int | None = None, run_experiment: bool = True,
                 quick: bool = False) -> dict:
    """Ejecuta el pipeline KDD completo y devuelve las rutas de evidencia."""
    configure_pandas_display()
    configure_matplotlib_style()
    _ensure_dirs()

    logger = KDDLogger()
    artifacts: dict = {}

    # =====================================================================
    # FASE 1. Comprensión del dominio y definición del objetivo
    # =====================================================================
    with logger.phase(1) as phase:
        _section("FASE 1. COMPRENSIÓN DEL DOMINIO Y OBJETIVO")
        phase.metric("pregunta_investigacion", BUSINESS_QUESTION)
        phase.metric("objetivo_general", BUSINESS_OBJECTIVE)
        phase.metric("criterios_exito", SUCCESS_CRITERIA)
        phase.metric("unidad_de_analisis", "cliente (agregación de pedidos)")
        phase.metric("tarea_de_mineria", "clustering no supervisado (K-Means)")
        phase.decision(
            "Se elige clustering y no reglas de asociación porque el objetivo es "
            "segmentar clientes para retención, no descubrir canastas. Además, cada "
            "pedido contiene un solo producto, por lo que Apriori intra-pedido no aplica."
        )
        phase.decision(
            "Se declara que el dataset es simulado; la actividad permite datos reales "
            "o simulados y ocultarlo invalidaría la interpretación."
        )
        phase.decision(
            "La métrica de éxito no es solo estadística: los segmentos deben ser "
            "accionables, estables y traducibles a KPI de negocio."
        )
        print(f"Pregunta: {BUSINESS_QUESTION}")
        print(f"Objetivo: {BUSINESS_OBJECTIVE}")

    # =====================================================================
    # FASE 2. Selección de datos
    # =====================================================================
    with logger.phase(2) as phase:
        _section("FASE 2. SELECCIÓN DE DATOS")
        loader = DataLoader(DATASET)
        raw = loader.load()
        profile = loader.profile()
        loader.get_info(profile)
        artifacts["perfil_dataset"] = loader.save_profile()

        phase.metrics_update({
            "filas": profile["filas"],
            "columnas": profile["columnas"],
            "periodo": f"{profile['fecha_min']} a {profile['fecha_max']}",
            "clientes": profile["clientes_unicos"],
            "productos": profile["productos_unicos"],
            "nulos": profile["valores_nulos"],
            "duplicados": profile["filas_duplicadas"],
            "max_items_por_pedido": profile["maximo_items_por_pedido"],
            "es_probablemente_simulado": profile["es_probablemente_simulado"],
            "evidencia_simulacion": profile["evidencia_dataset_simulado"],
        })
        phase.decision(
            "Se conserva el periodo completo 2020-2024: permite analizar recencia y "
            "estacionalidad sin un sesgo de ventana arbitrario."
        )
        phase.decision(
            "Se descarta el análisis de canasta porque el 100 % de los pedidos tiene "
            "un solo producto; se documenta como limitación del dataset."
        )
        if profile["es_probablemente_simulado"]:
            phase.warn(
                "El dataset presenta evidencia fuerte de simulación; los hallazgos "
                "son ilustrativos y no deben presentarse como resultados de Amazon."
            )
        phase.metric("diccionario_datos", "docs/results/diccionario_datos.md")
        (Path(RESULTS_DIR) / "diccionario_datos.md").write_text(
            to_markdown(), encoding="utf-8"
        )

    # =====================================================================
    # FASE 3. Preprocesamiento y limpieza
    # =====================================================================
    with logger.phase(3) as phase:
        _section("FASE 3. PREPROCESAMIENTO Y LIMPIEZA")
        cleaner = DataCleaner(raw)
        cleaner.clean()
        full = cleaner.data
        sales = cleaner.analytical_data
        cutoff = pd.Timestamp(raw["OrderDate"].max())
        artifacts["reporte_limpieza"] = cleaner.save_report()

        revenue = cleaner.report["ingresos"]
        phase.metrics_update({
            "filas_entrada": cleaner.report["filas_entrada"],
            "filas_limpias": len(full),
            "filas_ventas_realizadas": len(sales),
            "duplicados_eliminados": cleaner.report.get("duplicados_eliminados", 0),
            "nulos_antes": cleaner.report.get("total_nulos_antes", 0),
            "nulos_despues": cleaner.report.get("total_nulos_despues", 0),
            "estados_incluidos": cleaner.report["ventas_realizadas"]["estados_incluidos"],
            "estados_excluidos": cleaner.report["ventas_realizadas"]["filas_excluidas"],
            "ingreso_reportado_total_amount": revenue["ingreso_total_reportado"],
            "ingreso_neto_con_descuento": revenue["ingreso_neto_con_descuento"],
            "sobreestimacion_por_descuento": revenue["diferencia_total_amount_vs_neto"],
            "filas_inconsistentes_total_amount": revenue["filas_inconsistentes_total_amount"],
            "fecha_corte": cutoff.date().isoformat(),
        })
        for decision in cleaner.report["decisiones"]:
            phase.decision(decision)
        if cleaner.report.get("total_nulos_antes", 0) == 0:
            phase.warn(
                "No había nulos ni duplicados: la limpieza no cambia el resultado en "
                "este dataset. Su impacto se demuestra en el experimento de degradación."
            )
        phase.metric("reporte_limpieza", artifacts["reporte_limpieza"])

    # =====================================================================
    # FASE 4. Transformación y reducción
    # =====================================================================
    with logger.phase(4) as phase:
        _section("FASE 4. TRANSFORMACIÓN Y REDUCCIÓN")
        transformer = CustomerTransformer(sales, cutoff_date=cutoff)
        features = transformer.aggregate()
        model_matrix = transformer.prepare_model_features(winsorize=True)
        acp_diag = transformer.pca_diagnostics()

        pca = PCAModel(n_components=2)
        pca.fit_transform(transformer.model_values)
        artifacts["acp_varianza"] = pca.plot_variance()
        artifacts["acp_sin_clusters"] = pca.plot_projection(
            title="Clientes en el plano PCA (sin clusters)",
            name="acp_clientes_sin_clusters",
        )
        artifacts["reporte_transformacion"] = transformer.save_report()
        artifacts["clientes_perfil"] = transformer.save_features()

        phase.metrics_update({
            "clientes_agregados": len(features),
            "fecha_corte_recencia": cutoff.date().isoformat(),
            "recencia_min_dias": int(features["Dias_Ultima_Compra"].min()),
            "recencia_mediana_dias": float(features["Dias_Ultima_Compra"].median()),
            "recencia_max_dias": int(features["Dias_Ultima_Compra"].max()),
            "variables_modelo": CLUSTERING_VARS,
            "winsorizacion": transformer.report["estandarizacion"]["winsorizacion"],
            "acp_varianza_2_componentes": acp_diag["varianza_2_componentes"],
            "acp_componentes_para_85pct": acp_diag["componentes_para_85pct"],
            "acp_varianza_acumulada": acp_diag["varianza_acumulada"],
        })
        phase.decision(
            "La recencia se calcula contra la fecha máxima del dataset "
            f"({cutoff.date()}), no contra la fecha del sistema; así el resultado es "
            "reproducible en cualquier día de ejecución."
        )
        phase.decision(
            "Se recortan Ingreso_Neto, Ticket_Promedio y Frecuencia al percentil 1-99 "
            "para evitar que valores extremos dominen la distancia euclidiana."
        )
        phase.decision(
            "Se estandariza antes de K-Means porque las variables tienen escalas "
            "incompatibles (USD vs días vs número de pedidos)."
        )
        phase.decision(
            f"El ACP con 2 componentes retiene {acp_diag['varianza_2_componentes']:.1%}. "
            "Por eso se usa para visualización y diagnóstico, no como entrada única del "
            "agrupamiento; K-Means trabaja sobre las cinco variables estandarizadas."
        )
        if acp_diag["varianza_2_componentes"] < 0.75:
            phase.warn(
                "PC1+PC2 retienen menos del 75 % de la varianza; reducir a 2 componentes "
                "antes de agrupar habría descartado información relevante."
            )

    # =====================================================================
    # FASE 5. Minería de datos
    # =====================================================================
    with logger.phase(5) as phase:
        _section("FASE 5. MINERÍA DE DATOS")
        segmenter = CustomerSegmenter(features, model_matrix)
        if quick:
            # Evaluación reducida para desarrollo.
            segmenter.selection = {
                "k_seleccionado": k or 4, "criterio": "modo rápido",
                "tabla": [], "codo": {},
            }
            segmenter.fit(k=k or 4, evaluate=False)
        else:
            segmenter.fit(k=k, evaluate=True)
        labels = segmenter.assign_segment_labels()
        selected_k = int(segmenter.kmeans_model.n_clusters)

        # Complemento de asociación: diagnóstico + co-compra por cliente.
        association = MarketBasketAnalyzer(sales)
        rules = association.customer_cooccurrence()
        artifacts["reglas_asociacion"] = association.save_results()
        artifacts["reglas_figura"] = association.plot_top_rules()

        phase.metrics_update({
            "k_seleccionado": selected_k,
            "criterio_k": segmenter.selection.get("criterio"),
            "metricas_k": {
                key: value for key, value in segmenter.selection.items()
                if key in {"silhouette", "davies_bouldin", "calinski_harabasz",
                           "cluster_min_pct", "runner_up", "codo"}
            },
            "etiquetas": {str(k): v for k, v in labels.items()},
            "apriori_aplicable": association.diagnostic.get("apriori_aplicable"),
            "max_items_por_pedido": association.diagnostic.get("items_por_pedido_max"),
            "reglas_co_compra_encontradas": int(len(rules)),
        })
        silhouette_txt = (
            f"{segmenter.selection['silhouette']:.3f}"
            if segmenter.selection.get("silhouette") is not None else "n/d"
        )
        phase.decision(
            f"k = {selected_k} se elige con evidencia: mayor silhouette entre k=3..6 "
            f"({silhouette_txt}), desempate por "
            "Davies-Bouldin y umbral mínimo de tamaño documentado. El codo se "
            "reporta como diagnóstico complementario."
        )
        if segmenter.selection.get("restriccion_relajada"):
            min_pct = segmenter.selection.get("cluster_min_pct")
            min_txt = f"{min_pct:.2f} %" if min_pct is not None else "n/d"
            phase.warn(
                "Ningún k de 3 a 6 alcanzó el 5 % mínimo por cluster "
                f"(el menor: {min_txt}). La estructura natural contiene un segmento "
                "élite pequeño que concentra una parte crítica del ingreso. Se relajó "
                "el umbral al 1 % técnico en lugar de fusionarlo o descartarlo, y la "
                "relajación queda registrada en la evaluación."
            )
        phase.decision(
            "No se usa Apriori clásico: no hay canastas multi-producto. Se calculan "
            "reglas de co-compra a nivel de cliente como complemento exploratorio."
        )

    # =====================================================================
    # FASE 6. Interpretación y evaluación
    # =====================================================================
    with logger.phase(6) as phase:
        _section("FASE 6. INTERPRETACIÓN Y EVALUACIÓN")
        n_bootstrap = 3 if quick else N_BOOTSTRAP_STABILITY
        evaluation = segmenter.validate(n_bootstrap=n_bootstrap)
        segments_df = segmenter.profile_table()
        print("\nPerfil de segmentos:")
        print(segments_df[[
            "Etiqueta", "Clientes", "Pct_Clientes", "Ingreso_Neto",
            "Participacion_Ingresos", "Ticket_Promedio", "Frecuencia",
            "Descuento_Promedio", "Dias_Ultima_Compra",
        ]].to_string(index=False))

        artifacts["clusters_pca"] = segmenter.plot_pca_clusters(pca)
        artifacts["perfil_radar"] = segmenter.plot_radar()
        artifacts["segmentos_valor"] = segmenter.plot_segment_value()
        artifacts["perfil_relativo"] = segmenter.plot_profile_heatmap()
        artifacts["segmentos_json"] = segmenter.save_segments()
        artifacts["clientes_segmentados"] = segmenter.save_customers()

        evaluation_path = Path(RESULTS_DIR) / "evaluacion_clustering.json"
        evaluation_path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        artifacts["evaluacion"] = str(evaluation_path)

        degradation_payload = None
        if run_experiment:
            _section("EXPERIMENTO CONTROLADO DE DEGRADACIÓN")
            experiment = DegradationExperiment(
                sales_df=sales,
                full_df=full,
                reference_features=features,
                reference_model=model_matrix,
                reference_labels=segmenter.labels,
                k=selected_k,
                cutoff=cutoff,
            )
            experiment.run()
            artifacts["experimento_figura"] = experiment.plot()
            artifacts["experimento_json"] = experiment.save_results(
                plot_path=artifacts["experimento_figura"]
            )
            degradation_payload = {
                "resumen": experiment.summary.to_dict(orient="records"),
                "detalle": experiment.rows,
                "fecha_corte": experiment.cutoff.date().isoformat(),
                "fecha_sistema_usada": experiment.system_cutoff.date().isoformat(),
                "umbral_estabilidad": STABILITY_ARI_THRESHOLD,
            }
            print(experiment.summary.to_string(index=False))

        phase.metrics_update({
            "metricas_finales": evaluation["metricas_finales"],
            "estabilidad_bootstrap": evaluation["estabilidad"],
            "etiquetas_unicas": len({s["Etiqueta"] for s in evaluation["perfil"]})
            == len(evaluation["perfil"]),
            "segmentos": evaluation["perfil"],
            "experimento_degradacion": degradation_payload,
        })
        phase.decision(
            "La evaluación no se limita a la inercia: se reportan silhouette, "
            "Davies-Bouldin, Calinski-Harabasz y estabilidad bootstrap con ARI."
        )
        phase.decision(
            "Las etiquetas se validan como únicas y se vinculan a un perfil de negocio; "
            "dos clusters no pueden compartir la misma etiqueta."
        )
        stability = evaluation["estabilidad"]
        if stability["ari_media"] < 0.70:
            phase.warn(
                f"Estabilidad bootstrap por debajo del umbral: ARI medio = "
                f"{stability['ari_media']}. Los segmentos deben interpretarse con cautela."
            )

    # =====================================================================
    # FASE 7. Presentación y uso del conocimiento
    # =====================================================================
    with logger.phase(7) as phase:
        _section("FASE 7. PRESENTACIÓN Y USO DEL CONOCIMIENTO")
        knowledge = build_knowledge_payload(
            segments_df.to_dict(orient="records"),
            reference_date=cutoff.date().isoformat(),
            objective=BUSINESS_OBJECTIVE,
        )
        knowledge_path = Path(RESULTS_DIR) / "conocimiento_acciones.json"
        knowledge_path.write_text(
            json.dumps(knowledge, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        knowledge_md_path = Path(RESULTS_DIR) / "conocimiento_acciones.md"
        knowledge_md_path.write_text(
            build_playbook_markdown(knowledge), encoding="utf-8"
        )
        artifacts["conocimiento_json"] = str(knowledge_path)
        artifacts["conocimiento_md"] = str(knowledge_md_path)

        phase.metrics_update({
            "ingreso_en_riesgo_usd": knowledge["ingreso_en_riesgo_usd"],
            "potencial_crecimiento_usd": knowledge["potencial_crecimiento_usd"],
            "impacto_estimado_total_usd": knowledge["impacto_estimado_total_usd"],
            "criterio_impacto": knowledge["criterio_impacto"],
            "segmentos_priorizados": [
                {"segmento": s["Etiqueta"], "prioridad": s["prioridad"],
                 "impacto_estimado_usd": s["impacto_estimado_usd"]}
                for s in knowledge["segmentos"]
            ],
        })
        phase.decision(
            "Cada segmento tiene prioridad, acciones, KPI, riesgo e impacto estimado "
            "con supuestos explícitos; el impacto debe validarse con un piloto."
        )
        phase.decision(
            "El conocimiento se entrega en JSON y Markdown para integrarse en un "
            "sistema de información o en un tablero de seguimiento."
        )

    # =====================================================================
    # EDA descriptivo y reporte final
    # =====================================================================
    eda_stats = _descriptive_eda(full, sales)
    artifacts["estadisticas_descriptivas"] = str(
        Path(RESULTS_DIR) / "estadisticas_descriptivas.json"
    )
    artifacts["eda"] = eda_stats

    payload = logger.to_dict()
    export_paths = logger.export(
        json_path=str(Path(RESULTS_DIR) / "kdd_log.json"),
        markdown_path=str(Path(RESULTS_DIR) / "kdd_fases.md"),
    )
    artifacts.update({f"log_{k}": v for k, v in export_paths.items()})

    association_payload = {
        "diagnostico_canastas": association.diagnostic,
        "reglas": [] if rules is None else rules.to_dict(orient="records"),
    }
    report = KDDReport(
        logger_payload=payload,
        profile=profile,
        cleaning=cleaner.report,
        transformation=transformer.report,
        selection=segmenter.selection,
        segments=segments_df.to_dict(orient="records"),
        evaluation=evaluation,
        association=association_payload,
        degradation=degradation_payload,
        knowledge=knowledge,
    )
    artifacts["reporte_tecnico"] = report.save()

    _section("PIPELINE COMPLETO")
    print("Evidencia generada:")
    for key, value in artifacts.items():
        if isinstance(value, str) and (value.startswith("docs") or "docs" in value):
            print(f"  - {key}: {value}")
    print(f"\nReporte técnico: {artifacts['reporte_tecnico']}")
    print("Ensayo académico: propuesta_PDF.md")
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline KDD de segmentación de clientes")
    parser.add_argument("--k", type=int, default=None,
                        help="Fija el número de clusters (omite la selección automática).")
    parser.add_argument("--no-experiment", action="store_true",
                        help="Omite el experimento de degradación.")
    parser.add_argument("--quick", action="store_true",
                        help="Evaluación reducida para desarrollo.")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        run_pipeline(k=args.k, run_experiment=not args.no_experiment,
                     quick=args.quick)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
