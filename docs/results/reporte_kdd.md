# Reporte técnico del proceso KDD

**Proyecto:** Segmentación de clientes de una tienda en línea (Amazon Sales).  
**Método:** Knowledge Discovery in Databases (Fayyad et al., 1996).  
**Generado:** 2026-10-25T09:17:39

Este documento es evidencia automática del pipeline. El ensayo académico que interpreta los resultados está en `propuesta_PDF.md`.

## 1. Perfil del dataset (fase 2)

- Filas: **100,000**; columnas: **20**.
- Periodo: **2020-01-01** a **2024-12-29**.
- Clientes únicos: **30,243**; pedidos: **100,000**; productos: **175**.
- Nulos: **0**; duplicados: **0**.
- Máximo de ítems por pedido: **1**.
- ¿Probablemente simulado?: **True**.

Evidencia de simulación:

- El número de filas es exactamente 100 000.
- No hay valores nulos en ninguna columna.
- No hay filas duplicadas exactas.
- Las filas por año son casi uniformes (CV = 0.0049).
- Cada pedido contiene exactamente un producto (no hay canastas).
- El periodo inicia exactamente el 2020-01-01.

## 2. Limpieza (fase 3)

- Filas de entrada: **100,000**; filas limpias: **100,000**; filas analíticas: **89,820**.
- Duplicados eliminados: **0**.
- Nulos antes: **0**; nulos después: **0**.
- Ingreso reportado por `TotalAmount`: **USD 132,424,396.57**.
- Ingreso neto con descuento: **USD 123,182,963.52**.
- Sobreestimación detectada: **USD 9,241,433.05** (59,754 filas).

Decisiones de limpieza:

- Todas las fechas se interpretaron correctamente.
- Duplicados exactos eliminados: 0. Quedan 100,000 filas.
- No se encontraron valores faltantes; no fue necesario imputar. La fase de limpieza se documenta igualmente porque el resultado es una condición del dataset, no una omisión del proceso.
- Reglas de consistencia: se eliminaron 0 filas con cantidad/precio inválido; se recortaron 0 descuentos al rango [0, 1] y se llevaron a cero los impuestos/envíos negativos.
- TotalAmount del dataset no descuenta la promoción (diferencia máxima de 0.30 por redondeo). Se adopta Ingreso_Neto para el análisis y se cuantifica una sobreestimación de 9,241,433.05 USD.
- Los valores atípicos se reportan con IQR (factor 3) y se conservan; en la fase 4 se recortan solo las variables de modelado al percentil 1-99.
- Conjunto analítico: 89,820 ventas realizadas (Delivered, Shipped). Se excluyen 10,180 pedidos cancelados, devueltos o pendientes para no inflar ingreso ni recencia.

## 3. Transformación (fase 4)

- Fecha de corte: **2024-12-29** (máximo del dataset, no la fecha del sistema).
- Recencia: mínima **0** días, mediana **545.0** días, máxima **1824** días.
- Clientes agregados: **28971**.
- Varianza ACP con 2 componentes: **69.1%**; componentes para 85 %: **3**.

## 4. Selección de k y minería (fase 5)

- k seleccionado: **5**.
- Criterio: Mayor silhouette entre k=3..6. Ningún k alcanzó el 5% mínimo por cluster; la restricción se relajó al 1% técnico y se documenta. Desempate por menor Davies-Bouldin.
- Silhouette: **0.318437**; Davies-Bouldin: **0.88439**; Calinski-Harabasz: **18789.4135**.
- Codo estimado: k = **5**.

| k | Inercia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cluster mínimo (%) |
|---:|---:|---:|---:|---:|---:|
| 2 | 97,458.83 | 0.7880 | 0.2407 | 14,088.2 | 1.30 |
| 3 | 71,714.02 | 0.2816 | 0.9904 | 14,772.3 | 1.29 |
| 4 | 53,637.23 | 0.2910 | 1.0012 | 16,421.0 | 1.29 |
| 5 | 40,297.12 | 0.3184 | 0.8844 | 18,789.4 | 1.29 |
| 6 | 35,445.76 | 0.2894 | 0.9739 | 17,881.1 | 1.29 |
| 7 | 31,834.15 | 0.2952 | 1.0234 | 17,138.6 | 1.29 |
| 8 | 29,042.61 | 0.2816 | 1.0148 | 16,499.4 | 1.29 |
| 9 | 26,624.17 | 0.2769 | 1.0152 | 16,076.6 | 1.29 |
| 10 | 24,885.57 | 0.2567 | 1.0611 | 15,513.1 | 1.29 |

## 5. Interpretación y evaluación (fase 6)

| Segmento | Clientes | % clientes | Ingreso (USD) | % ingreso | Ticket | Frecuencia | Recencia (días) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VIP Leales | 4,819 | 16.6% | 40,210,679.25 | 38.7% | 1,714.39 | 5.03 | 347 |
| VIP Elite | 375 | 1.3% | 36,120,356.00 | 34.8% | 1,621.41 | 59.46 | 36 |
| Potenciales de Crecimiento | 11,270 | 38.9% | 16,953,138.70 | 16.3% | 622.44 | 2.26 | 349 |
| Ocasionales Inactivos | 7,408 | 25.6% | 6,145,259.10 | 5.9% | 581.74 | 1.35 | 1266 |
| Cazadores de Ofertas | 5,099 | 17.6% | 4,422,446.30 | 4.3% | 525.88 | 1.53 | 815 |

- Silhouette final: **0.318437**.
- Estabilidad bootstrap (ARI medio): **0.9809** (Estable); mínimo 0.9693, máximo 0.9937.
- Etiquetas únicas: **True**.

## 6. Asociación como complemento

- Máximo de ítems por pedido: **1**; pedidos con más de un producto: **0**.
- Apriori clásico aplicable: **False**.
- Cada pedido contiene un único producto; no hay canastas que permitan reglas de asociación intra-pedido. Se justifica usar clustering de clientes y se complementa con co-compra histórica.

## 7. Experimento de degradación

ARI frente a la solución de referencia (1.0 = idénticos):

| Escenario | Fase afectada | ARI medio | Desv. | Interpretación |
|---|---|---:|---:|---|
| Recencia con la fecha del sistema | Fase 4. Transformación | 0.9945 | 0.0049 | Partición casi idéntica |
| Ruido ±20% en precio (20% filas) | Fase 3-4. Calidad y transformación | 0.9817 | 0.0023 | Partición casi idéntica |
| Duplicados 5% (limpieza omitida) | Fase 3. Preprocesamiento | 0.9666 | 0.0017 | Partición casi idéntica |
| Sin filtrar estados no realizados | Fase 3. Preprocesamiento | 0.8615 | 0.0016 | Degradación moderada |
| Nulos 10% imputados con 0 | Fase 3. Preprocesamiento | 0.7683 | 0.0017 | Degradación moderada |
| Sin estandarización | Fase 4. Transformación | 0.1597 | 0.0001 | Degradación severa |
| k=2 fijado sin evaluación | Fase 5. Minería | 0.0198 | 0.0000 | Degradación severa |

> Un ARI alto indica que la partición se reproduce, no que los totales sean correctos: duplicados o ruido pueden preservar los segmentos y, a la vez, inflar pedidos e ingreso. El ARI debe acompañarse de auditoría de ingresos y conteos.

> La fecha del sistema se fijó en 2026-10-25 para que el escenario sea reproducible; en producción sería la fecha real de consulta.

## 8. Uso del conocimiento (fase 7)

- Ingreso en riesgo: **USD 10,567,705.40**.
- Potencial de crecimiento: **USD 16,953,138.70**.
- Impacto estimado total: **USD 3,348,942.11**.

> Estimaciones conservadoras basadas en tasas de reactivación/crecimiento declaradas por segmento. No son una proyección financiera garantizada; deben validarse con un piloto y con el área comercial.


## 9. Registro de fases

| Fase | Nombre | Estado | Duración (s) | Alertas |
|---|---|---|---:|---:|
| 1 | Comprensión del dominio y definición del objetivo | ok | 0.00 | 0 |
| 2 | Selección de datos | ok | 4.38 | 1 |
| 3 | Preprocesamiento y limpieza | ok | 2.26 | 1 |
| 4 | Transformación y reducción | ok | 9.92 | 1 |
| 5 | Minería de datos | ok | 101.24 | 1 |
| 6 | Interpretación y evaluación | ok | 115.37 | 0 |
| 7 | Presentación y uso del conocimiento | ok | 0.01 | 0 |

## 10. Limitaciones declaradas

- El dataset es simulado; los hallazgos ilustran el proceso, no una empresa real.
- La etiqueta de segmento es una interpretación del perfil, no una verdad absoluta.
- La estabilidad y el impacto deben validarse con datos nuevos y con expertos de negocio.
- El análisis es descriptivo y no establece causalidad.
