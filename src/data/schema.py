"""
Data dictionary for the Amazon sales dataset.

Every column is documented with its type, role inside the KDD process and the
decision taken about it (kept, dropped, used as identifier, used as feature).
The dictionary is rendered to Markdown by :func:`to_markdown`.
"""

from typing import Dict, List

# role values:
#   identifier -> used to group/join, never a feature
#   feature    -> candidate predictor for the mining task
#   target     -> variable to predict (none: unsupervised task)
#   context    -> descriptive/diagnostic attribute
#   dropped    -> removed during selection (no analytical value)
DATA_DICTIONARY: List[Dict[str, str]] = [
    {
        "name": "OrderID",
        "dtype": "string",
        "role": "identifier",
        "description": "Identificador único de la línea de pedido.",
        "decision": "Se elimina del conjunto analítico: no aporta señal y evita fuga de información.",
    },
    {
        "name": "OrderDate",
        "dtype": "datetime",
        "role": "feature",
        "description": "Fecha en que se registró el pedido.",
        "decision": "Se conserva; se usa para la serie temporal y para calcular la recencia.",
    },
    {
        "name": "CustomerID",
        "dtype": "string",
        "role": "identifier",
        "description": "Identificador único del cliente.",
        "decision": "Se conserva como llave de agregación para construir el perfil del cliente.",
    },
    {
        "name": "CustomerName",
        "dtype": "string",
        "role": "context",
        "description": "Nombre del cliente.",
        "decision": "Se elimina del conjunto modelado; solo se usa para reportes legibles.",
    },
    {
        "name": "ProductID",
        "dtype": "string",
        "role": "identifier",
        "description": "Identificador único del producto.",
        "decision": "Se conserva para el análisis de productos y la unión con el nombre.",
    },
    {
        "name": "ProductName",
        "dtype": "string",
        "role": "context",
        "description": "Nombre comercial del producto.",
        "decision": "Se elimina del conjunto modelado; se recupera para gráficos y reportes.",
    },
    {
        "name": "Category",
        "dtype": "category",
        "role": "feature",
        "description": "Categoría de producto.",
        "decision": "Se conserva para el análisis descriptivo y de perfil de segmentos.",
    },
    {
        "name": "Brand",
        "dtype": "category",
        "role": "feature",
        "description": "Marca del producto.",
        "decision": "Se conserva para el análisis descriptivo.",
    },
    {
        "name": "Quantity",
        "dtype": "int64",
        "role": "feature",
        "description": "Unidades compradas en la línea de pedido.",
        "decision": "Se conserva; participa en el ticket y en la validación del monto.",
    },
    {
        "name": "UnitPrice",
        "dtype": "float64",
        "role": "feature",
        "description": "Precio unitario del producto.",
        "decision": "Se conserva; participa en el ticket y en la validación del monto.",
    },
    {
        "name": "Discount",
        "dtype": "float64",
        "role": "feature",
        "description": "Descuento aplicado a la línea, expresado como proporción (0-1).",
        "decision": "Se conserva; se valida su rango y se usa para perfilar sensibilidad al precio.",
    },
    {
        "name": "Tax",
        "dtype": "float64",
        "role": "feature",
        "description": "Impuesto de la línea de pedido.",
        "decision": "Se conserva; forma parte del monto total.",
    },
    {
        "name": "ShippingCost",
        "dtype": "float64",
        "role": "feature",
        "description": "Costo de envío de la línea.",
        "decision": "Se conserva; forma parte del monto total.",
    },
    {
        "name": "TotalAmount",
        "dtype": "float64",
        "role": "target",
        "description": "Monto total facturado de la línea.",
        "decision": (
            "Se conserva como variable objetivo del negocio. Se recalcula cuando no "
            "coincide con Quantity * UnitPrice * (1 - Discount) + Tax + ShippingCost."
        ),
    },
    {
        "name": "PaymentMethod",
        "dtype": "category",
        "role": "feature",
        "description": "Medio de pago utilizado.",
        "decision": "Se conserva para el análisis descriptivo y de perfil.",
    },
    {
        "name": "OrderStatus",
        "dtype": "category",
        "role": "feature",
        "description": "Estado del pedido (entregado, enviado, pendiente, devuelto, cancelado).",
        "decision": "Se conserva; permite diagnosticar devoluciones y cancelaciones.",
    },
    {
        "name": "City",
        "dtype": "category",
        "role": "context",
        "description": "Ciudad de entrega.",
        "decision": "Se conserva para el análisis geográfico descriptivo.",
    },
    {
        "name": "State",
        "dtype": "category",
        "role": "context",
        "description": "Estado o provincia de entrega.",
        "decision": "Se conserva para el análisis geográfico descriptivo.",
    },
    {
        "name": "Country",
        "dtype": "category",
        "role": "dropped",
        "description": "País de entrega (valor constante en el dataset).",
        "decision": "Se elimina: no tiene varianza y no aporta información.",
    },
    {
        "name": "SellerID",
        "dtype": "string",
        "role": "identifier",
        "description": "Identificador del vendedor.",
        "decision": "Se conserva para el ranking de vendedores.",
    },
]


def to_markdown(dictionary: List[Dict[str, str]] = None) -> str:
    """Render the data dictionary as a Markdown table."""
    dictionary = dictionary or DATA_DICTIONARY
    lines = [
        "# Diccionario de datos",
        "",
        "| Variable | Tipo | Rol KDD | Descripción | Decisión |",
        "|---|---|---|---|---|",
    ]
    for row in dictionary:
        lines.append(
            "| {name} | {dtype} | {role} | {description} | {decision} |".format(**row)
        )
    return "\n".join(lines) + "\n"
