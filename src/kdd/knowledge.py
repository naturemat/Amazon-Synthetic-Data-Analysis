"""
Capa de conocimiento (fase 7 de KDD): del segmento a la acción.

Traduce cada segmento en prioridad, objetivo, acciones, KPI, riesgo e impacto
esperado. Las estimaciones de impacto son explícitas y conservadoras: se basan
en supuestos declarados, no en una promesa de resultado.
"""

from __future__ import annotations

from typing import Any, Dict, List

SEGMENT_PLAYBOOK: Dict[str, Dict[str, Any]] = {
    "VIP Elite": {
        "prioridad": "Máxima",
        "objetivo": "Retener a los clientes de máxima frecuencia y valor.",
        "acciones": [
            "Gestión de cuenta dedicada y beneficios exclusivos no replicables.",
            "Contacto proactivo antes de su ciclo habitual de compra.",
            "Programa de lealtad con recompensas por recurrencia y categorías de interés.",
        ],
        "kpi": "Retención, frecuencia, ticket y participación en el ingreso.",
        "riesgo": "Concentración extrema: pocos clientes explican una parte crítica del ingreso.",
        "supuesto_impacto": "Retención adicional del 3 % del ingreso histórico del segmento.",
        "tasa_impacto": 0.03,
    },
    "VIP Leales": {
        "prioridad": "Máxima",
        "objetivo": "Proteger y retener el ingreso concentrado.",
        "acciones": [
            "Programa de fidelización con atención preferente y beneficios exclusivos.",
            "Contacto proactivo antes de su ciclo habitual de compra.",
            "Seguimiento individual de satisfacción y alertas tempranas de fuga.",
        ],
        "kpi": "Retención, ingreso por cliente y días entre compras.",
        "riesgo": "La pérdida de pocos clientes afecta una porción grande del ingreso.",
        "supuesto_impacto": "Retención adicional del 3 % del ingreso histórico del segmento.",
        "tasa_impacto": 0.03,
    },
    "VIP en Riesgo": {
        "prioridad": "Máxima",
        "objetivo": "Reactivar a los clientes de alto valor que dejaron de comprar.",
        "acciones": [
            "Campaña de recuperación personalizada con incentivo acotado.",
            "Oferta basada en su categoría y marca favoritas.",
            "Contacto humano para los casos de mayor ingreso.",
        ],
        "kpi": "Tasa de reactivación, ingreso recuperado y costo por recuperación.",
        "riesgo": "Es el ingreso más expuesto: ya muestra señales de alejamiento.",
        "supuesto_impacto": "Reactivación del 8 % del ingreso del segmento.",
        "tasa_impacto": 0.08,
    },
    "Potenciales de Crecimiento": {
        "prioridad": "Alta",
        "objetivo": "Aumentar frecuencia y ticket antes de que la relación se enfríe.",
        "acciones": [
            "Recomendaciones personalizadas y venta cruzada por categoría.",
            "Programa de puntos con meta de segunda y tercera compra.",
            "Prueba de bundles con productos complementarios.",
        ],
        "kpi": "Frecuencia, ticket promedio y tasa de recompra.",
        "riesgo": "Sin acción, el cliente puede quedarse en una compra única.",
        "supuesto_impacto": "Aumento del 5 % del ingreso del segmento por mayor frecuencia.",
        "tasa_impacto": 0.05,
    },
    "Cazadores de Ofertas": {
        "prioridad": "Media",
        "objetivo": "Convertir la compra promocional en compra recurrente y rentable.",
        "acciones": [
            "Cupones de segunda compra con vigencia limitada, no descuento permanente.",
            "Medir margen por pedido, no solo volumen.",
            "Migrar a un programa de lealtad cuando se alcance una frecuencia mínima.",
        ],
        "kpi": "Tasa de segunda compra, margen por pedido y uso de cupones.",
        "riesgo": "Dependencia de promociones y bajo margen por cliente.",
        "supuesto_impacto": "Reactivación del 2 % del ingreso del segmento con margen protegido.",
        "tasa_impacto": 0.02,
    },
    "Ocasionales Inactivos": {
        "prioridad": "Media",
        "objetivo": "Reactivar de forma selectiva o depurar la base.",
        "acciones": [
            "Campaña de reactivación de bajo costo y alcance limitado.",
            "Medir retorno antes de escalar; depurar a quien no responda.",
            "Excluir del gasto masivo para no subsidiar clientes no rentables.",
        ],
        "kpi": "Tasa de reactivación y costo por cliente recuperado.",
        "riesgo": "Costo de contacto sin retorno; el segmento es grande y de bajo valor.",
        "supuesto_impacto": "Reactivación del 2 % del ingreso del segmento.",
        "tasa_impacto": 0.02,
    },
}

FALLBACK_PLAYBOOK: Dict[str, Any] = {
    "prioridad": "Baja",
    "objetivo": "Observar el segmento antes de asignar presupuesto.",
    "acciones": [
        "Monitorear su evolución durante el siguiente periodo.",
        "Diseñar una prueba piloto de bajo costo.",
    ],
    "kpi": "Tamaño del segmento y variación de ingreso.",
    "riesgo": "Segmento pequeño o poco estable: evidencia insuficiente para decidir.",
    "supuesto_impacto": "Sin estimación por tamaño reducido.",
    "tasa_impacto": 0.0,
}


def playbook_for(label: str) -> Dict[str, Any]:
    """Devuelve el plan de acción del segmento, tolerando sufijos de desempate."""
    if label in SEGMENT_PLAYBOOK:
        return SEGMENT_PLAYBOOK[label]
    for key, value in SEGMENT_PLAYBOOK.items():
        if label.startswith(key):
            return value
    return FALLBACK_PLAYBOOK


def build_knowledge_payload(segments: List[Dict[str, Any]], reference_date: str,
                            objective: str = "") -> Dict[str, Any]:
    """Construye el JSON de acciones e impacto esperado por segmento."""
    total_revenue = sum(float(s["Ingreso_Neto"]) for s in segments)
    total_customers = sum(int(s["Clientes"]) for s in segments)

    enriched: List[Dict[str, Any]] = []
    for seg in segments:
        plan = playbook_for(seg["Etiqueta"])
        impact = float(seg["Ingreso_Neto"]) * float(plan["tasa_impacto"])
        enriched.append({
            **seg,
            "prioridad": plan["prioridad"],
            "objetivo": plan["objetivo"],
            "acciones": plan["acciones"],
            "kpi": plan["kpi"],
            "riesgo": plan["riesgo"],
            "supuesto_impacto": plan["supuesto_impacto"],
            "impacto_estimado_usd": round(impact, 2),
        })

    prioritized = sorted(
        enriched,
        key=lambda s: ({"Máxima": 3, "Alta": 2, "Media": 1, "Baja": 0}[s["prioridad"]],
                       s["Ingreso_Neto"]),
        reverse=True,
    )
    at_risk = [
        s for s in enriched
        if s["Etiqueta"].startswith(("VIP en Riesgo", "Ocasionales", "Cazadores"))
    ]
    growth = [s for s in enriched if s["Etiqueta"].startswith("Potenciales")]

    return {
        "objetivo": objective,
        "fecha_corte": reference_date,
        "ingreso_total_usd": round(total_revenue, 2),
        "clientes_totales": total_customers,
        "ingreso_en_riesgo_usd": round(sum(s["Ingreso_Neto"] for s in at_risk), 2),
        "potencial_crecimiento_usd": round(sum(s["Ingreso_Neto"] for s in growth), 2),
        "impacto_estimado_total_usd": round(
            sum(s["impacto_estimado_usd"] for s in enriched), 2
        ),
        "criterio_impacto": (
            "Estimaciones conservadoras basadas en tasas de reactivación/crecimiento "
            "declaradas por segmento. No son una proyección financiera garantizada; "
            "deben validarse con un piloto y con el área comercial."
        ),
        "segmentos": prioritized,
    }


def build_playbook_markdown(knowledge: Dict[str, Any]) -> str:
    """Renderiza el informe de conocimiento y acciones en Markdown."""
    segments = knowledge["segmentos"]
    lines = [
        "# Fase 7. Del conocimiento a la acción",
        "",
        f"**Objetivo:** {knowledge.get('objetivo', '')}",
        f"**Fecha de corte:** {knowledge['fecha_corte']}",
        f"**Ingreso analizado:** USD {knowledge['ingreso_total_usd']:,.2f}",
        f"**Ingreso en riesgo:** USD {knowledge['ingreso_en_riesgo_usd']:,.2f}",
        f"**Impacto estimado total:** USD {knowledge['impacto_estimado_total_usd']:,.2f}",
        "",
        f"> {knowledge['criterio_impacto']}",
        "",
        "## Resumen de segmentos",
        "",
        "| Segmento | Prioridad | Clientes | % clientes | Ingreso (USD) | % ingreso | Impacto estimado (USD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for seg in segments:
        lines.append(
            "| {Etiqueta} | {prioridad} | {Clientes:,} | {Pct_Clientes:.1f}% | "
            "{Ingreso_Neto:,.2f} | {Participacion_Ingresos:.1f}% | "
            "{impacto_estimado_usd:,.2f} |".format(**seg)
        )

    lines += ["", "## Plan de acción", ""]
    for seg in segments:
        lines += [
            f"### {seg['Etiqueta']}",
            "",
            f"- **Prioridad:** {seg['prioridad']}",
            f"- **Objetivo:** {seg['objetivo']}",
            f"- **KPI:** {seg['kpi']}",
            f"- **Riesgo:** {seg['riesgo']}",
            f"- **Supuesto de impacto:** {seg['supuesto_impacto']}",
            "- **Acciones:**",
        ]
        lines += [f"  {i}. {a}" for i, a in enumerate(seg["acciones"], 1)]
        lines.append("")

    lines += ["## Lectura ejecutiva", "", _executive_reading(segments), ""]
    return "\n".join(lines)


def _executive_reading(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return "No hay segmentos para interpretar."
    ordered = sorted(segments, key=lambda s: s["Participacion_Ingresos"], reverse=True)
    top = ordered[0]
    top3_share = sum(s["Participacion_Ingresos"] for s in ordered[:3])
    low = [
        s for s in segments
        if s["Etiqueta"].startswith(
            ("VIP en Riesgo", "Ocasionales", "Cazadores")
        )
    ]
    low_share = sum(s["Pct_Clientes"] for s in low)
    return (
        f"El segmento **{top['Etiqueta']}** concentra {top['Participacion_Ingresos']:.1f}% "
        f"del ingreso con {top['Pct_Clientes']:.1f}% de los clientes. Los tres segmentos "
        f"de mayor ingreso explican {top3_share:.1f}% del total. Los segmentos con señal "
        f"de riesgo o baja actividad agrupan {low_share:.1f}% de los clientes; por eso la "
        "reactivación debe ser selectiva y medida, no una campaña masiva."
    )
