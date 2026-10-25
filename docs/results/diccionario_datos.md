# Diccionario de datos

| Variable | Tipo | Rol KDD | Descripción | Decisión |
|---|---|---|---|---|
| OrderID | string | identifier | Identificador único de la línea de pedido. | Se elimina del conjunto analítico: no aporta señal y evita fuga de información. |
| OrderDate | datetime | feature | Fecha en que se registró el pedido. | Se conserva; se usa para la serie temporal y para calcular la recencia. |
| CustomerID | string | identifier | Identificador único del cliente. | Se conserva como llave de agregación para construir el perfil del cliente. |
| CustomerName | string | context | Nombre del cliente. | Se elimina del conjunto modelado; solo se usa para reportes legibles. |
| ProductID | string | identifier | Identificador único del producto. | Se conserva para el análisis de productos y la unión con el nombre. |
| ProductName | string | context | Nombre comercial del producto. | Se elimina del conjunto modelado; se recupera para gráficos y reportes. |
| Category | category | feature | Categoría de producto. | Se conserva para el análisis descriptivo y de perfil de segmentos. |
| Brand | category | feature | Marca del producto. | Se conserva para el análisis descriptivo. |
| Quantity | int64 | feature | Unidades compradas en la línea de pedido. | Se conserva; participa en el ticket y en la validación del monto. |
| UnitPrice | float64 | feature | Precio unitario del producto. | Se conserva; participa en el ticket y en la validación del monto. |
| Discount | float64 | feature | Descuento aplicado a la línea, expresado como proporción (0-1). | Se conserva; se valida su rango y se usa para perfilar sensibilidad al precio. |
| Tax | float64 | feature | Impuesto de la línea de pedido. | Se conserva; forma parte del monto total. |
| ShippingCost | float64 | feature | Costo de envío de la línea. | Se conserva; forma parte del monto total. |
| TotalAmount | float64 | target | Monto total facturado de la línea. | Se conserva como variable objetivo del negocio. Se recalcula cuando no coincide con Quantity * UnitPrice * (1 - Discount) + Tax + ShippingCost. |
| PaymentMethod | category | feature | Medio de pago utilizado. | Se conserva para el análisis descriptivo y de perfil. |
| OrderStatus | category | feature | Estado del pedido (entregado, enviado, pendiente, devuelto, cancelado). | Se conserva; permite diagnosticar devoluciones y cancelaciones. |
| City | category | context | Ciudad de entrega. | Se conserva para el análisis geográfico descriptivo. |
| State | category | context | Estado o provincia de entrega. | Se conserva para el análisis geográfico descriptivo. |
| Country | category | dropped | País de entrega (valor constante en el dataset). | Se elimina: no tiene varianza y no aporta información. |
| SellerID | string | identifier | Identificador del vendedor. | Se conserva para el ranking de vendedores. |
