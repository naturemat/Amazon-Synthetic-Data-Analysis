"""
Reporte automático de la ejecución KDD.

No sustituye al ensayo académico (``propuesta_PDF.md``): es la evidencia
técnica que respalda cada afirmación del documento. Se genera en Markdown para
que pueda revisarse, versionarse y anexarse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class KDDReport:
    """Construye un informe técnico con los resultados de las siete fases."""

    def __init__(self, logger_payload: Dict[str, Any], profile: Dict[str, Any],
                 cleaning: Dict[str, Any], transformation: Dict[str, Any],
                 selection: Dict[str, Any], segments: List[Dict[str, Any]],
                 evaluation: Dict[str, Any], association: Dict[str, Any],
                 degradation: Optional[Dict[str, Any]] = None,
                 knowledge: Optional[Dict[str, Any]] = None):
        self.logger = logger_payload
        self.profile = profile
        self.cleaning = cleaning
        self.transformation = transformation
        self.selection = selection
        self.segments = segments
        self.evaluation = evaluation
        self.association = association
        self.degradation = degradation
        self.knowledge = knowledge

    # ------------------------------------------------------------------ build
    def to_markdown(self) -> str:
        lines: List[str] = [
            "# Reporte técnico del proceso KDD",
            "",
            "**Proyecto:** Segmentación de clientes de una tienda en línea (Amazon Sales).  ",
            "**Método:** Knowledge Discovery in Databases (Fayyad et al., 1996).  ",
            f"**Generado:** {self.logger.get('finished_at', '')}",
            "",
            "Este documento es evidencia automática del pipeline. El ensayo académico "
            "que interpreta los resultados está en `propuesta_PDF.md`.",
            "",
            "## 1. Perfil del dataset (fase 2)",
            "",
            f"- Filas: **{self.profile['filas']:,}**; columnas: **{self.profile['columnas']}**.",
            f"- Periodo: **{self.profile['fecha_min']}** a **{self.profile['fecha_max']}**.",
            f"- Clientes únicos: **{self.profile['clientes_unicos']:,}**; "
            f"pedidos: **{self.profile['pedidos_unicos']:,}**; "
            f"productos: **{self.profile['productos_unicos']:,}**.",
            f"- Nulos: **{self.profile['valores_nulos']}**; duplicados: "
            f"**{self.profile['filas_duplicadas']}**.",
            f"- Máximo de ítems por pedido: **{self.profile['maximo_items_por_pedido']}**.",
            f"- ¿Probablemente simulado?: **{self.profile['es_probablemente_simulado']}**.",
            "",
            "Evidencia de simulación:",
            "",
        ]
        lines += [f"- {reason}" for reason in self.profile["evidencia_dataset_simulado"]]
        lines += ["", "## 2. Limpieza (fase 3)", ""]
        lines += [
            f"- Filas de entrada: **{self.cleaning['filas_entrada']:,}**; "
            f"filas limpias: **{self.cleaning['filas_salida']:,}**; "
            f"filas analíticas: **{self.cleaning['filas_analiticas']:,}**.",
            f"- Duplicados eliminados: **{self.cleaning.get('duplicados_eliminados', 0)}**.",
            f"- Nulos antes: **{self.cleaning.get('total_nulos_antes', 0)}**; "
            f"nulos después: **{self.cleaning.get('total_nulos_despues', 0)}**.",
        ]
        revenue = self.cleaning.get("ingresos", {})
        lines += [
            f"- Ingreso reportado por `TotalAmount`: **USD {revenue.get('ingreso_total_reportado', 0):,.2f}**.",
            f"- Ingreso neto con descuento: **USD {revenue.get('ingreso_neto_con_descuento', 0):,.2f}**.",
            f"- Sobreestimación detectada: **USD {revenue.get('diferencia_total_amount_vs_neto', 0):,.2f}** "
            f"({revenue.get('filas_inconsistentes_total_amount', 0):,} filas).",
            "",
            "Decisiones de limpieza:",
            "",
        ]
        lines += [f"- {d}" for d in self.cleaning.get("decisiones", [])]

        lines += ["", "## 3. Transformación (fase 4)", ""]
        rec = self.transformation.get("recencia", {})
        acp = self.transformation.get("acp", {})
        lines += [
            f"- Fecha de corte: **{self.transformation.get('fecha_corte')}** "
            "(máximo del dataset, no la fecha del sistema).",
            f"- Recencia: mínima **{rec.get('min_dias')}** días, mediana "
            f"**{rec.get('mediana_dias')}** días, máxima **{rec.get('max_dias')}** días.",
            f"- Clientes agregados: **{self.transformation.get('clientes')}**.",
            f"- Varianza ACP con 2 componentes: **{acp.get('varianza_2_componentes', 0):.1%}**; "
            f"componentes para 85 %: **{acp.get('componentes_para_85pct')}**.",
        ]

        lines += ["", "## 4. Selección de k y minería (fase 5)", ""]
        lines += [
            f"- k seleccionado: **{self.selection.get('k_seleccionado')}**.",
            f"- Criterio: {self.selection.get('criterio')}",
            f"- Silhouette: **{self.selection.get('silhouette')}**; "
            f"Davies-Bouldin: **{self.selection.get('davies_bouldin')}**; "
            f"Calinski-Harabasz: **{self.selection.get('calinski_harabasz')}**.",
            f"- Codo estimado: k = **{self.selection.get('codo', {}).get('k_codo')}**.",
            "",
            "| k | Inercia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cluster mínimo (%) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in self.selection.get("tabla", []):
            lines.append(
                f"| {row['k']} | {row['inercia']:,.2f} | {row['silhouette']:.4f} | "
                f"{row['davies_bouldin']:.4f} | {row['calinski_harabasz']:,.1f} | "
                f"{row['cluster_min_pct']:.2f} |"
            )

        lines += ["", "## 5. Interpretación y evaluación (fase 6)", ""]
        lines += [
            "| Segmento | Clientes | % clientes | Ingreso (USD) | % ingreso | Ticket | Frecuencia | Recencia (días) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for seg in self.segments:
            lines.append(
                f"| {seg['Etiqueta']} | {seg['Clientes']:,} | {seg['Pct_Clientes']:.1f}% | "
                f"{seg['Ingreso_Neto']:,.2f} | {seg['Participacion_Ingresos']:.1f}% | "
                f"{seg['Ticket_Promedio']:,.2f} | {seg['Frecuencia']:.2f} | "
                f"{seg['Dias_Ultima_Compra']:.0f} |"
            )
        stability = self.evaluation.get("estabilidad", {})
        metrics = self.evaluation.get("metricas_finales", {})
        lines += [
            "",
            f"- Silhouette final: **{metrics.get('silhouette')}**.",
            f"- Estabilidad bootstrap (ARI medio): **{stability.get('ari_media')}** "
            f"({stability.get('interpretacion')}); mínimo {stability.get('ari_min')}, "
            f"máximo {stability.get('ari_max')}.",
            f"- Etiquetas únicas: **{len({s['Etiqueta'] for s in self.segments}) == len(self.segments)}**.",
        ]

        lines += ["", "## 6. Asociación como complemento", ""]
        diag = self.association.get("diagnostico_canastas", {})
        lines += [
            f"- Máximo de ítems por pedido: **{diag.get('items_por_pedido_max')}**; "
            f"pedidos con más de un producto: **{diag.get('pedidos_con_mas_de_un_producto')}**.",
            f"- Apriori clásico aplicable: **{diag.get('apriori_aplicable')}**.",
            f"- {diag.get('explicacion', '')}",
        ]
        rules = self.association.get("reglas", [])
        if rules:
            lines += ["", "| Antecedente | Consecuente | Soporte | Confianza | Lift |",
                      "|---|---|---:|---:|---:|"]
            for rule in rules[:10]:
                lines.append(
                    f"| {rule['antecedente']} | {rule['consecuente']} | "
                    f"{rule['soporte']:.4f} | {rule['confianza']:.3f} | {rule['lift']:.3f} |"
                )

        if self.degradation:
            lines += ["", "## 7. Experimento de degradación", "",
                      "ARI frente a la solución de referencia (1.0 = idénticos):", "",
                      "| Escenario | Fase afectada | ARI medio | Desv. | Interpretación |",
                      "|---|---|---:|---:|---|"]
            for row in self.degradation.get("resumen", []):
                if row["ari_media"] >= 0.90:
                    interp = "Partición casi idéntica"
                elif row["ari_media"] >= 0.70:
                    interp = "Degradación moderada"
                elif row["ari_media"] >= 0.50:
                    interp = "Degradación alta"
                else:
                    interp = "Degradación severa"
                lines.append(
                    f"| {row['escenario']} | {row['fase_afectada']} | "
                    f"{row['ari_media']:.4f} | {row['ari_desv']:.4f} | {interp} |"
                )
            lines += [
                "",
                "> Un ARI alto indica que la partición se reproduce, no que los "
                "totales sean correctos: duplicados o ruido pueden preservar los "
                "segmentos y, a la vez, inflar pedidos e ingreso. El ARI debe "
                "acompañarse de auditoría de ingresos y conteos.",
            ]
            system_date = self.degradation.get("fecha_sistema_usada")
            if system_date:
                lines += [
                    "",
                    f"> La fecha del sistema se fijó en {system_date} para que el "
                    "escenario sea reproducible; en producción sería la fecha real "
                    "de consulta.",
                ]

        if self.knowledge:
            lines += ["", "## 8. Uso del conocimiento (fase 7)", "",
                      f"- Ingreso en riesgo: **USD {self.knowledge['ingreso_en_riesgo_usd']:,.2f}**.",
                      f"- Potencial de crecimiento: **USD {self.knowledge['potencial_crecimiento_usd']:,.2f}**.",
                      f"- Impacto estimado total: **USD {self.knowledge['impacto_estimado_total_usd']:,.2f}**.",
                      "", f"> {self.knowledge['criterio_impacto']}", ""]

        lines += ["", "## 9. Registro de fases", "",
                  "| Fase | Nombre | Estado | Duración (s) | Alertas |",
                  "|---|---|---|---:|---:|"]
        for phase in self.logger.get("phases", []):
            lines.append(
                f"| {phase['phase']} | {phase['title']} | {phase['status']} | "
                f"{phase['duration_s']:.2f} | {len(phase['warnings'])} |"
            )

        lines += ["", "## 10. Limitaciones declaradas", "",
                  "- El dataset es simulado; los hallazgos ilustran el proceso, no una empresa real.",
                  "- La etiqueta de segmento es una interpretación del perfil, no una verdad absoluta.",
                  "- La estabilidad y el impacto deben validarse con datos nuevos y con expertos de negocio.",
                  "- El análisis es descriptivo y no establece causalidad.", ""]
        return "\n".join(lines)

    def save(self, path: Optional[str] = None) -> str:
        from ..config.settings import RESULTS_DIR

        target = Path(path or RESULTS_DIR / "reporte_kdd.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return str(target)
