"""
Configuración central del proyecto de segmentación de clientes (KDD).

Todas las decisiones metodológicas relevantes están parametrizadas aquí para
que el pipeline sea reproducible y auditable. Los valores se justifican en
``propuesta_PDF.md`` y en el reporte automático de fases (``docs/kdd_report.md``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET = "amazon.csv"

COLUMNS = {
    "id": {
        "order_id": "OrderID",
        "customer_id": "CustomerID",
        "customer_name": "CustomerName",
        "product_id": "ProductID",
        "product_name": "ProductName",
        "seller_id": "SellerID",
    },
    "categorical": {
        "category": "Category",
        "brand": "Brand",
        "payment_method": "PaymentMethod",
        "order_status": "OrderStatus",
        "city": "City",
        "state": "State",
        "country": "Country",
    },
    "numeric": {
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
        "discount": "Discount",
        "tax": "Tax",
        "shipping_cost": "ShippingCost",
        "total_amount": "TotalAmount",
    },
    "date": "OrderDate",
}

DTYPES = {
    "OrderID": "string",
    "CustomerID": "string",
    "CustomerName": "string",
    "ProductID": "string",
    "ProductName": "string",
    "Category": "category",
    "Brand": "category",
    "PaymentMethod": "category",
    "OrderStatus": "category",
    "City": "category",
    "State": "category",
    "Country": "category",
    "SellerID": "string",
}

REQUIRED_COLUMNS = [
    "OrderID", "OrderDate", "CustomerID", "ProductID", "ProductName",
    "Category", "Brand", "Quantity", "UnitPrice", "Discount", "Tax",
    "ShippingCost", "TotalAmount", "PaymentMethod", "OrderStatus",
    "City", "State", "Country", "SellerID",
]

# Columnas que no aportan señal al modelo (identificadores o texto libre).
COLS_TO_DROP = ["OrderID", "CustomerName", "ProductName", "Country"]

NUMERIC_VARS = ["Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost", "TotalAmount"]
CATEGORICAL_VARS = ["Category", "Brand", "PaymentMethod", "OrderStatus", "City", "State"]
ID_VARS = ["CustomerID", "ProductID", "SellerID"]

# ---------------------------------------------------------------------------
# KDD: fase 1 - definición del problema
# ---------------------------------------------------------------------------
BUSINESS_QUESTION = (
    "¿Qué segmentos de clientes existen según su valor, frecuencia, sensibilidad "
    "al descuento y recencia, y cómo debe priorizar la organización sus acciones "
    "de retención y crecimiento?"
)

BUSINESS_OBJECTIVE = (
    "Segmentar la base de clientes de una tienda en línea y traducir los "
    "segmentos en acciones comerciales priorizadas, cuantificando el ingreso "
    "en riesgo y el potencial de crecimiento."
)

SUCCESS_CRITERIA = [
    "Segmentos con tamaño mínimo preferido del 5 % de la base para que sean "
    "accionables; si la estructura natural produce un segmento élite menor, se "
    "admite un mínimo técnico del 1 % y se documenta la relajación.",
    "Coeficiente de silueta (silhouette) positivo y estable; se reporta junto a "
    "Davies-Bouldin y Calinski-Harabasz, no de forma aislada.",
    "Etiquetas de segmento únicas y trazables a un perfil de negocio.",
    "Estabilidad del agrupamiento ≥ 0,70 de ARI promedio en remuestreo bootstrap.",
    "Cada segmento debe tener al menos una acción, un KPI y un riesgo asociado.",
]

# Ventana temporal: se usa la fecha máxima observada, nunca la fecha del sistema.
# Esto evita que la recencia cambie cada día que se ejecuta el pipeline.
CUTOFF_DATE = None  # None -> max(OrderDate) del dataset

# Fase 3: estados que representan una venta realizada. Cancelados y devueltos
# no generan ingreso ni deben contar para la recencia del cliente.
ANALYTICAL_STATUSES = ["Delivered", "Shipped"]
EXCLUDED_STATUSES = ["Cancelled", "Returned", "Pending"]

# Fase 3: reglas de consistencia
MIN_QUANTITY = 1
MIN_UNIT_PRICE = 0.0
VALID_DISCOUNT_RANGE = (0.0, 1.0)
OUTLIER_IQR_FACTOR = 3.0  # solo se reportan; no se eliminan automáticamente

# Fase 4: variables del perfil de cliente usadas en la minería
CLUSTERING_VARS = [
    "Ingreso_Neto",
    "Ticket_Promedio",
    "Frecuencia",
    "Descuento_Promedio",
    "Dias_Ultima_Compra",
]

# Las variables monetarias y de frecuencia tienen colas largas. Se recortan al
# percentil 1 y 99 para que unos pocos clientes extremos no dominen la distancia
# euclidiana. El recorte se reporta y no altera el dataset original.
WINSOR_QUANTILES = (0.01, 0.99)
WINSOR_VARS = ["Ingreso_Neto", "Ticket_Promedio", "Frecuencia"]
RFM_QUANTILES = 5
SCATTER_MATRIX_SAMPLE = 5000

# Fase 4/5: ACP se usa para visualización y para diagnosticar redundancia.
# El agrupamiento se ejecuta sobre las variables estandarizadas (sin pérdida de
# información); esta decisión se compara empíricamente con ACP-2/ACP-3.
PCA_VARIANCE_TARGET = 0.85
CLUSTERING_SPACE = "scaled"  # "scaled" | "pca"

# ---------------------------------------------------------------------------
# Fase 5: K-Means y selección de k
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE = list(range(2, 11))
K_MIN_ACTIONABLE = 3
K_MAX_ACTIONABLE = 6
MIN_CLUSTER_SHARE = 0.05
# Si ningún k cumple el 5 %, se relaja a un mínimo técnico del 1 % (≈290
# clientes) antes de aceptar un cluster como artefacto. La relajación se
# reporta de forma explícita; no se oculta en el criterio de selección.
MIN_CLUSTER_SHARE_HARD = 0.01
SILHOUETTE_SAMPLE_SIZE = 8000
N_BOOTSTRAP_STABILITY = 10
BOOTSTRAP_FRACTION = 0.8

KMEANS_CONFIG = {
    "n_clusters": 4,  # valor de respaldo; el pipeline selecciona k con evidencia
    "init": "k-means++",
    "n_init": 20,
    "max_iter": 300,
    "algorithm": "lloyd",
    "random_state": RANDOM_STATE,
}

PCA_CONFIG = {
    "n_components": 2,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Fase 6: validación
# ---------------------------------------------------------------------------
STABILITY_ARI_THRESHOLD = 0.70
DEGRADATION_SEEDS = [42, 7, 2024]
DEGRADATION_DUPLICATE_RATE = 0.05
DEGRADATION_NULL_RATE = 0.10
DEGRADATION_NOISE_RATE = 0.20

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = str(ROOT_DIR / "docs")
RESEARCH_DIR = Path(OUTPUT_DIR)
FIGURES_DIR = RESEARCH_DIR / "images"
RESULTS_DIR = RESEARCH_DIR / "results"
KDD_OUTPUT_DIR = str(RESEARCH_DIR)

FIGURE_DPI = int(os.getenv("FIGURE_DPI", "140"))
SHOW_PLOTS = os.getenv("SHOW_PLOTS", "0") == "1"
PLOT_STYLE = "seaborn-v0_8-darkgrid"

FIGURE_SIZES = {
    "small": (8, 5),
    "medium": (12, 6),
    "large": (16, 8),
    "extra_large": (20, 12),
}

# ---------------------------------------------------------------------------
# Asociación (complemento exploratorio)
# ---------------------------------------------------------------------------
ASSOCIATION_MIN_SUPPORT = 0.01
ASSOCIATION_MIN_LIFT = 1.10
ASSOCIATION_TOP_N = 15
