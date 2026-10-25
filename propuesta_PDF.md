# Aplicación práctica del proceso KDD en la extracción de conocimiento para la toma de decisiones en sistemas de información

**Nombre:** [Nombre del estudiante]  
**Facultad:** [Facultad]  
**Asignatura:** Minería de Datos  
**Docente:** [Nombre del docente]  
**Fecha:** 25 de octubre de 2026  
**Título breve (encabezado):** KDD EN LA SEGMENTACIÓN DE CLIENTES

---

## Resumen

El presente ensayo analiza la aplicación práctica de las fases del proceso
*Knowledge Discovery in Databases* (KDD) en un caso de segmentación de clientes
de una tienda en línea. A partir de un dataset de 100 000 pedidos registrados
entre 2020 y 2024, se ejecutó un pipeline reproducible en Python que cubre las
siete fases propuestas por Fayyad, Piatetsky-Shapiro y Smyth (1996): comprensión
del dominio, selección de datos, preprocesamiento, transformación, minería,
interpretación y uso del conocimiento. El resultado principal es una
segmentación en cinco grupos con etiquetas únicas (k = 5, silhouette = 0,318,
Davies-Bouldin = 0,884), validada con estabilidad bootstrap (ARI = 0,981) y con
un experimento controlado de degradación. Se concluye que la calidad del
conocimiento no depende del algoritmo sino de la ejecución articulada de las
fases: las decisiones de limpieza, transformación y evaluación son las que
determinan si los segmentos son útiles para decidir.

**Palabras clave:** KDD, minería de datos, segmentación de clientes, K-Means,
sistemas de información, toma de decisiones.

---

## 1. Planteamiento del problema

Las organizaciones actuales generan volúmenes de datos que crecen más rápido que
su capacidad para interpretarlos. Contar con un *data warehouse*, un tablero o
una herramienta de *machine learning* no garantiza que esos datos se conviertan
en conocimiento útil. La minería de datos ofrece técnicas para descubrir
patrones, pero su valor depende de que se inserten en un proceso estructurado
que conecte los datos con una decisión concreta.

El proceso KDD surge precisamente para cubrir esa brecha. Fayyad et al. (1996)
lo definen como el proceso no trivial de identificar patrones válidos, novedosos,
potencialmente útiles y comprensibles en los datos. Sin embargo, en la práctica
es frecuente encontrar tres fallas:

1. **Se omiten fases críticas.** Se entrena un modelo sin limpiar, sin
   transformar o sin evaluar; el resultado es técnicamente ejecutable pero
   difícil de interpretar y de defender.
2. **Se prioriza la herramienta sobre el proceso.** Se elige K-Means, Apriori o
   una red neuronal antes de definir la pregunta de negocio y el criterio de
   éxito.
3. **No se cierra el ciclo.** El análisis termina en una tabla o un gráfico y no
   se traduce en acciones, responsables, indicadores ni mecanismos de
   seguimiento dentro del sistema de información.

El caso de estudio de este ensayo es una tienda en línea con 100 000 pedidos y
30 243 clientes únicos. La organización necesita saber qué segmentos de clientes
existen, cuáles concentran el ingreso y cómo priorizar sus acciones de retención
y crecimiento. Responder esa pregunta exige aplicar cada fase de KDD y, sobre
todo, justificar las decisiones que se toman en cada una.

## 2. Pregunta de investigación

¿Cómo se aplican las fases del proceso KDD en un caso práctico de minería de
datos y cuál es su impacto en la generación de conocimiento útil para la toma de
decisiones?

## 3. Objetivos

### 3.1 Objetivo general

Analizar la aplicación práctica de las fases del proceso KDD en un caso de
minería de datos, evaluando su efectividad en la generación de conocimiento para
la toma de decisiones.

### 3.2 Objetivos específicos

1. Describir las fases del proceso KDD y su relación con los sistemas de
   información.
2. Aplicar cada fase a un conjunto de datos real o simulado.
3. Identificar problemas comunes en cada fase y documentar cómo se trataron.
4. Evaluar los resultados obtenidos y su utilidad para la toma de decisiones.

## 4. Marco teórico

### 4.1 Concepto de minería de datos

La minería de datos es el conjunto de técnicas y algoritmos que permiten
descubrir patrones, asociaciones, cambios, anomalías y estructuras a partir de
grandes volúmenes de datos (Han et al., 2012). No es sinónimo de KDD: la minería
es una de las fases del proceso, la que aplica el algoritmo. Fayyad et al. (1996)
insisten en que el conocimiento se construye antes y después del algoritmo:
antes, cuando se entiende el dominio y se preparan los datos; después, cuando se
interpretan y se validan los patrones.

En el contexto de los sistemas de información, la minería de datos se conecta
con la analítica descriptiva, diagnóstica, predictiva y prescriptiva. Su
propósito no es reemplazar la decisión humana, sino reducir la incertidumbre y
aportar evidencia para decidir. Por eso una técnica puede ser correcta desde el
punto de vista estadístico y, al mismo tiempo, inútil para la organización si no
responde a una pregunta relevante o si no se puede interpretar.

### 4.2 Proceso KDD

Fayyad et al. (1996) organizan el descubrimiento de conocimiento en siete fases
que no son estrictamente lineales: la interpretación puede obligar a volver a la
transformación o a la selección de datos. La Tabla 1 resume cada fase y su
producto.

**Tabla 1.** Fases del proceso KDD y producto esperado.

| Fase | Propósito | Producto |
|---|---|---|
| 1. Comprensión del dominio | Entender el negocio, definir la pregunta y el criterio de éxito | Objetivo, KPI, alcance |
| 2. Selección de datos | Identificar fuentes, periodo, variables y unidad de análisis | Conjunto de datos y diccionario |
| 3. Preprocesamiento | Tratar duplicados, nulos, ruido, inconsistencias y valores atípicos | Dataset limpio y trazable |
| 4. Transformación | Agregar, normalizar, discretizar y reducir dimensionalidad | Matriz de modelado |
| 5. Minería de datos | Aplicar el algoritmo adecuado a la tarea | Modelo y patrones |
| 6. Interpretación y evaluación | Validar, comparar y traducir los patrones a lenguaje de negocio | Segmentos, métricas, hallazgos |
| 7. Uso del conocimiento | Integrar el hallazgo en decisiones y procesos | Acciones, KPI, impacto |

Una lectura ingenua del proceso puede llevar a creer que el esfuerzo principal
está en la fase 5. La experiencia muestra lo contrario: en este caso, las fases
3, 4 y 6 concentraron la mayor parte de las decisiones que afectaron el
resultado. El algoritmo K-Means es relativamente simple; lo difícil es definir
qué se agrupa, cómo se mide y cómo se interpreta.

### 4.3 Importancia del KDD en los sistemas de información

El KDD aporta a los sistemas de información tres capacidades:

- **Apoyo a la toma de decisiones.** Convierte datos operativos en evidencia
  para priorizar recursos, segmentar clientes o detectar riesgos.
- **Optimización de procesos.** Permite identificar cuellos de botella,
  comportamientos atípicos y oportunidades de mejora.
- **Identificación de patrones ocultos.** Revela relaciones que no son evidentes
  en reportes descriptivos tradicionales.

CRISP-DM (Wirth & Hipp, 2000) complementa esta visión con un ciclo de vida
industrial: comprensión del negocio, comprensión de los datos, preparación,
modelado, evaluación y despliegue. Ambos marcos coinciden en que el conocimiento
debe volver al negocio y sostenerse en el tiempo; un modelo sin seguimiento se
degrada.

### 4.4 Clustering y reglas de asociación

La actividad sugiere como ejemplo el análisis de canastas con reglas de
asociación. Sin embargo, la técnica debe elegirse según la pregunta. Las reglas
de asociación (Agrawal & Srikant, 1994) buscan combinaciones de ítems que
aparecen juntos en una transacción; el clustering busca agrupar casos similares.
En este proyecto la pregunta es *qué tipos de clientes existen y cómo
priorizarlos*, no *qué productos se compran juntos*. Por eso la tarea principal
es clustering y la unidad de análisis es el cliente, no el pedido.

Además, el dataset contiene exactamente un producto por pedido, de modo que no
existen canastas multi-producto. Aplicar Apriori clásico sería un error de
diseño. Se optó por K-Means sobre variables de comportamiento y se añadió, como
complemento exploratorio, un análisis de co-compra a nivel de cliente.

K-Means particiona los datos en *k* grupos minimizando la suma de distancias al
cuadrado dentro de cada grupo. Requiere que las variables estén estandarizadas
porque usa distancia euclidiana, y requiere decidir *k* con evidencia. La
literatura propone combinarlo con el método del codo y con índices como
silhouette (Rousseeuw, 1987), Davies-Bouldin (Davies & Bouldin, 1979) y
Calinski-Harabasz. Ningún índice es suficiente por sí solo; la decisión debe
incluir criterios de negocio, como el tamaño mínimo de un segmento para que sea
accionable.

## 5. Metodología

### 5.1 Tipo de investigación

La investigación es **aplicada**, porque resuelve un problema concreto de
segmentación de clientes; **descriptiva y analítica**, porque describe el
comportamiento de los datos y analiza relaciones entre variables; y de **enfoque
mixto**, porque combina el marco conceptual de KDD con la implementación
práctica en Python.

### 5.2 Caso de estudio y datos

El caso es una tienda en línea con ventas entre el 1 de enero de 2020 y el 29 de
diciembre de 2024. El dataset contiene 100 000 filas y 20 columnas: identificadores
de pedido, cliente, producto y vendedor; fecha; categoría; marca; cantidad;
precio unitario; descuento; impuesto; envío; monto total; medio de pago; estado
del pedido; ciudad; estado geográfico y país.

**Declaración de simulación.** El dataset es simulado. La evidencia es la
siguiente: 100 000 filas exactas, cero valores nulos, cero duplicados, ~20 000
filas por año con un coeficiente de variación de 0,005, un solo producto por
pedido y un periodo que inicia exactamente el 2020-01-01. La actividad permite
datos reales o simulados; declararlo es una condición de honestidad académica y
no invalida el ejercicio, porque lo que se evalúa es la aplicación del proceso.

### 5.3 Unidad de análisis y variables

La unidad de análisis es el **cliente**, obtenido al agregar los pedidos por
`CustomerID`. Las variables de modelado son cinco:

**Tabla 2.** Variables de segmentación.

| Variable | Definición | Justificación |
|---|---|---|
| `Ingreso_Neto` | Suma de `Quantity × UnitPrice × (1 − Discount)` | Mide valor sin inflar por descuentos |
| `Ticket_Promedio` | Ingreso neto medio por pedido | Distingue clientes de ticket alto |
| `Frecuencia` | Número de pedidos realizados | Mide lealtad y recurrencia |
| `Descuento_Promedio` | Descuento medio aplicado | Aproxima sensibilidad al precio |
| `Dias_Ultima_Compra` | Días desde la última compra hasta la fecha de corte | Mide recencia y riesgo de fuga |

### 5.4 Herramientas y reproducibilidad

Se utilizó Python 3.13 con `pandas` 2.3.3 para la manipulación de datos,
`scikit-learn` 1.9.0 para estandarización, ACP, K-Means y métricas, y
`matplotlib` 3.9.3 para las figuras. La semilla se fijó en 42. El pipeline se
ejecuta con `python main.py` y genera un registro JSON por fase, figuras y
reportes en `docs/`. Cada decisión metodológica está parametrizada en
`src/config/settings.py` para que el resultado sea auditable.

### 5.5 Consideraciones éticas

No se utilizó el nombre del cliente en el modelado ni en los reportes. El
análisis se realizó sobre identificadores y variables agregadas. Al ser un
dataset simulado, no hay riesgo de reidentificación de personas reales; aun así,
el pipeline excluye nombres y datos de contacto del conjunto analítico.

## 6. Aplicación de las fases de KDD

### 6.1 Fase 1. Comprensión del dominio y definición del objetivo

La pregunta de negocio se formuló antes de elegir el algoritmo: *¿qué segmentos
de clientes existen según su valor, frecuencia, sensibilidad al descuento y
recencia, y cómo debe priorizar la organización sus acciones de retención y
crecimiento?* El objetivo general se descompuso en cinco criterios de éxito:

1. Segmentos con tamaño mínimo preferido del 5 % de la base. Si la estructura
   natural produce un segmento élite menor, se admite un mínimo técnico del 1 %
   y la relajación se documenta; no se oculta ni se fuerza el número de grupos.
2. Métricas de agrupamiento positivas y reportadas en conjunto, no una sola.
3. Etiquetas únicas y trazables a un perfil de negocio.
4. Estabilidad bootstrap con ARI promedio ≥ 0,70.
5. Cada segmento con acción, KPI y riesgo asociado.

Esta fase evitó un error común: empezar por K-Means porque es popular. También
permitió descartar Apriori como técnica principal y justificar el clustering.

### 6.2 Fase 2. Selección de datos

Se conservó el periodo completo 2020-2024 para no imponer una ventana
arbitraria. Se documentó el diccionario de datos y se perfiló el dataset. El
perfilado reveló:

- 100 000 pedidos, 30 243 clientes, 175 productos y 1 999 vendedores.
- Cero nulos y cero duplicados exactos.
- Un solo producto por pedido; por tanto, no hay canastas.
- Fuerte evidencia de simulación.

La selección incluyó una decisión importante: el análisis de asociación
intra-pedido no era viable. En lugar de forzarlo, se documentó la limitación y
se optó por clustering, que sí responde a la pregunta.

### 6.3 Fase 3. Preprocesamiento y limpieza

La Tabla 3 resume los problemas detectados, el tratamiento aplicado y el
resultado. A diferencia de una limpieza que solo imprime cuántos nulos hay, aquí
cada problema se trató y se cuantificó.

**Tabla 3.** Problemas de calidad y tratamiento.

| Problema | Tratamiento | Resultado |
|---|---|---|
| Duplicados exactos | `drop_duplicates()` | 0 en este dataset |
| Valores nulos | Mediana (numéricas) y moda (categóricas) | 0 antes y 0 después |
| Fechas inválidas | `to_datetime(errors="coerce")` + eliminación | 0 |
| Cantidad/precio inválidos | Regla de negocio y eliminación | 0 |
| Descuento fuera de [0, 1] | Recorte al rango | 0 |
| Impuesto/envío negativos | Recorte a cero | 0 |
| `TotalAmount` inconsistente | Cálculo de `Ingreso_Neto` e `Ingreso_Facturado` | 59 754 filas (59,75 %) |
| Pedidos no realizados | Filtro a `Delivered` y `Shipped` | 10 180 excluidos |
| Valores atípicos | Reporte IQR; recorte solo en modelado | 1 865 en `Ingreso_Neto` |

El hallazgo más relevante de esta fase fue que `TotalAmount` **no descuenta la
promoción**: se calcula como cantidad × precio + impuesto + envío. La diferencia
entre el monto reportado y el ingreso neto con descuento es de **USD 9 241 433,05**
en el dataset completo, y afecta a 59 754 filas. Usar `TotalAmount` sin corregir
habría sobreestimado el ingreso y habría hecho que la etiqueta de "sensible al
precio" se apoyara en un descuento que no afectaba el ingreso reportado.

También se excluyeron 10 180 pedidos cancelados, devueltos o pendientes. Un
pedido cancelado no genera ingreso y no debería contar para la recencia; un
pedido pendiente aún no es una venta realizada. El conjunto analítico quedó en
**89 820 ventas realizadas** y **28 971 clientes**.

En este dataset no había nulos ni duplicados, por lo que la limpieza no cambió el
resultado. Esa es una condición del dataset, no una omisión del proceso. Para
demostrar el impacto de la fase 3 se diseñó el experimento de degradación que se
presenta en la sección 6.6.

### 6.4 Fase 4. Transformación y reducción

La transformación central fue la agregación por cliente. Además de las cinco
variables de la Tabla 2, se calcularon puntajes RFM por quintiles para
interpretar los segmentos.

**Recencia con fecha de corte fija.** Un error frecuente es calcular la recencia
con `pd.Timestamp.now()`. Eso hace que el resultado cambie cada día y que, con un
dataset que termina en 2024, todos los clientes parezcan inactivos. En este
proyecto la recencia se calcula contra `OrderDate.max()`, es decir, el
**29 de diciembre de 2024**. La recencia resultante va de 0 a 1 824 días, con
mediana de 545 días. El experimento de la sección 7.5 muestra que usar la fecha
del sistema casi no altera la partición (ARI = 0,994) porque el desplazamiento es
constante, pero corrompe la lectura absoluta; por eso se corrige igualmente.

**Recorte de colas.** `Ingreso_Neto`, `Ticket_Promedio` y `Frecuencia` tienen
colas largas. Se recortaron al percentil 1-99 solo para el modelado: 567
clientes en ingreso (1,96 %), 569 en ticket (1,96 %) y 284 en frecuencia (0,98 %).
El dataset original se conserva sin recortes para los reportes.

**Estandarización.** K-Means usa distancia euclidiana; sin estandarizar, el
ingreso (miles de USD) dominaría sobre la frecuencia (unidades) y la recencia
(días). Se aplicó `StandardScaler`.

**ACP.** El análisis de componentes principales mostró que dos componentes
retienen **69,1 %** de la varianza y tres retienen **86,0 %**. Por eso el ACP se
usa para visualización y diagnóstico, pero **no** como entrada única del
agrupamiento: reducir a dos componentes antes de K-Means habría descartado
alrededor de un tercio de la información. K-Means trabaja sobre las cinco
variables estandarizadas. Esta decisión corrige el riesgo de justificar un ACP
que retiene solo dos tercios de la varianza.

### 6.5 Fase 5. Minería de datos

Se utilizó K-Means con inicialización `k-means++`, 20 inicializaciones en el
modelo final, 10 en la exploración de k, 300 iteraciones y semilla 42. El número
de clusters no se fijó a priori: se evaluaron k = 2 a 10 con cinco criterios:

- Inercia y método del codo.
- Coeficiente de silueta.
- Índice Davies-Bouldin.
- Índice Calinski-Harabasz.
- Tamaño mínimo de cluster (5 % preferido, 1 % técnico).

La regla de selección fue: *entre k = 3 y 6, elegir el mayor silhouette;
desempatar por menor Davies-Bouldin*. El tamaño mínimo se aplicó como filtro de
accionabilidad, no como una cuota. La regla es explícita y reproducible; no
depende de la intuición del analista.

Ningún k de 3 a 6 alcanzó el 5 % mínimo en todos los clusters: en todos los
casos aparece un segmento élite de 375 clientes (1,29 % de la base) que concentra
34,78 % del ingreso. Ante esa estructura, se tomaron dos decisiones explícitas.
Primero, no se fusionó el segmento élite con los VIP Leales, porque habría
ocultado el grupo más valioso y más urgente de retener. Segundo, no se forzó un
k distinto para cumplir una cuota: se relajó el umbral al 1 % técnico, se
documentó la relajación y se eligió k = 5 por evidencia. Con ese umbral, k = 5
obtuvo el mayor silhouette (0,318), por delante de k = 4 (0,291), k = 6 (0,289)
y k = 3 (0,282); además, el codo coincidió en k = 5. k = 2 tuvo un silhouette
mayor (0,788), pero solo separa al grupo élite del resto y no produce una
segmentación accionable, por lo que quedó fuera del rango de decisión.

**Tabla 4.** Evaluación de k en el rango accionable.

| k | Inercia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Cluster menor | Decisión |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 71 714,02 | 0,282 | 0,990 | 14 772,35 | 1,29 % | Descartado |
| 4 | 53 637,23 | 0,291 | 1,001 | 16 420,96 | 1,29 % | Segundo lugar |
| 5 | 40 297,12 | **0,318** | **0,884** | **18 789,41** | 1,29 % | **Elegido** |
| 6 | 35 445,76 | 0,289 | 0,974 | 17 881,14 | 1,29 % | Descartado |

El método del codo se ejecutó y se graficó, pero se usó como diagnóstico
complementario, no como única justificación. La inercia siempre disminuye al
aumentar k, por lo que por sí sola no permite elegir. La inercia intraclase de
la solución final fue 40 297,12 (27,82 % de la inercia total); el 72,18 %
restante corresponde a la separación entre segmentos.

Como complemento se ejecutó un análisis de asociación. El diagnóstico confirmó
que cada pedido tiene un solo producto, por lo que Apriori clásico no aplica. Se
calcularon reglas de co-compra a nivel de cliente (si compró A, probabilidad de
comprar B), reportando soporte, confianza y lift. Es un complemento
exploratorio, no la tarea principal.

### 6.6 Fase 6. Interpretación y evaluación

La evaluación combinó métricas internas y estabilidad:

- **Silhouette** para medir separación y cohesión.
- **Davies-Bouldin** para medir similitud entre clusters.
- **Calinski-Harabasz** para medir dispersión entre grupos.
- **Bootstrap con ARI** para medir si los segmentos se reproducen al cambiar la
  muestra. Un ARI cercano a 1 indica estabilidad; cercano a 0, dependencia de la
  muestra.

Las etiquetas se asignaron por cuadrante valor/recencia y se desempataron con
frecuencia, ticket o descuento. Se verificó explícitamente que no hubiera dos
clusters con la misma etiqueta. Este punto corrige un error frecuente en
segmentaciones automáticas: dos clusters distintos reciben el mismo nombre y el
lector no puede distinguirlos. La solución final produjo cinco etiquetas únicas:
VIP Elite, VIP Leales, Potenciales de Crecimiento, Ocasionales Inactivos y
Cazadores de Ofertas.

La solución de k = 5 obtuvo silhouette = 0,318, Davies-Bouldin = 0,884 y
Calinski-Harabasz = 18 789,41, con un ARI bootstrap de 0,981 (mínimo 0,969) en
10 remuestreos. Los valores se detallan en la sección 7.3. Esa estabilidad es
necesaria, pero no suficiente: mide consistencia del algoritmo, no validez de
negocio.

Además, se ejecutó un **experimento controlado de degradación**. Se construyó
una solución de referencia con el pipeline completo y luego se introdujeron
errores en distintas fases: duplicados (5 %), nulos imputados con cero (10 %),
inclusión de pedidos no realizados, ruido de ±20 % en el precio, recencia
calculada con la fecha del sistema, eliminación de la estandarización y fijación
de k = 2 sin evaluación. Para cada escenario se
repitió K-Means con el k de referencia, salvo el escenario que evalúa
precisamente k = 2, y se midió el ARI contra la referencia. Así, la afirmación
"omitir fases degrada el resultado" se convierte en evidencia cuantificada y no
en una opinión.

### 6.7 Fase 7. Uso del conocimiento

Cada segmento se tradujo en un plan de acción con prioridad, objetivo, acciones,
KPI, riesgo e impacto estimado. Las estimaciones son conservadoras y declaran su
supuesto: retención adicional del 3 % del ingreso en los dos segmentos VIP,
aumento del 5 % en Potenciales de Crecimiento y reactivación del 2 % en
Ocasionales Inactivos y Cazadores de Ofertas. El impacto total estimado es de
USD 3 348 942,11, calculado sobre el ingreso histórico de cada segmento y
distribuido entre acciones de retención, crecimiento y reactivación selectiva. No es una proyección financiera garantizada; es una
hipótesis de trabajo para un piloto. El pipeline también cuantificó un ingreso en
riesgo de **USD 10 567 705,40** (Ocasionales Inactivos + Cazadores de Ofertas) y
tomó el ingreso actual de Potenciales de Crecimiento (USD 16 953 138,70) como base
para estimar un aumento del 5 % por mayor frecuencia. El ingreso en riesgo se
define aquí como el de los segmentos de baja actividad; el ingreso premium se
protege con retención y no se contabiliza como pérdida realizada.

El conocimiento se exportó en JSON y Markdown para que pueda integrarse en un
tablero o en un sistema de información. La fase 7 no termina con una tabla:  
termina con una decisión y un mecanismo de seguimiento.

## 7. Resultados

### 7.1 Perfil del dataset y limpieza

El dataset tiene 100 000 filas y 20 columnas. Después del filtro de ventas
realizadas quedan 89 820 pedidos y 28 971 clientes. El ingreso neto de
mercancía analizado es de **USD 103 851 879,35** (suma de `Ingreso_Neto` de los
segmentos, sin impuestos ni envío). En el dataset completo, `TotalAmount` reporta
USD 132 424 396,57 y el neto con descuento es USD 123 182 963,52; la diferencia
de **USD 9 241 433,05** es la sobreestimación por no aplicar la promoción. La
cifra de USD 123,18 millones incluye pedidos cancelados, devueltos y pendientes;
por eso no coincide con el ingreso realizado de USD 103,85 millones.

### 7.2 Segmentos obtenidos

La Tabla 5 presenta los cinco segmentos. Los nombres son interpretaciones del
perfil, no etiquetas preexistentes.

**Tabla 5.** Segmentos de clientes (k = 5).

| Segmento | Clientes | % clientes | Ingreso neto (USD) | % ingreso | Ticket | Frecuencia | Descuento | Recencia (días) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VIP Leales | 4 819 | 16,63 | 40 210 679,25 | 38,72 | 1 714,39 | 5,03 | 6,88 % | 347,2 |
| VIP Elite | 375 | 1,29 | 36 120 356,00 | 34,78 | 1 621,41 | 59,46 | 7,45 % | 36,2 |
| Potenciales de Crecimiento | 11 270 | 38,90 | 16 953 138,70 | 16,32 | 622,44 | 2,26 | 5,17 % | 349,0 |
| Ocasionales Inactivos | 7 408 | 25,57 | 6 145 259,10 | 5,92 | 581,74 | 1,35 | 3,55 % | 1 266,5 |
| Cazadores de Ofertas | 5 099 | 17,60 | 4 422 446,30 | 4,26 | 525,88 | 1,53 | 18,30 % | 815,0 |

La lectura principal es una concentración fuerte: **17,93 % de los clientes
(VIP Leales + VIP Elite) genera 73,50 % del ingreso**. El segmento VIP Elite es
especialmente relevante: 1,29 % de los clientes explica 34,78 % del ingreso, con
una frecuencia media de 59,46 pedidos y una recencia de 36 días. Los otros tres
segmentos agrupan 82,07 % de los clientes pero solo 26,50 % del ingreso. Si se
miran solo los dos de menor valor, Ocasionales Inactivos y Cazadores de Ofertas
representan 43,17 % de los clientes y 10,18 % del ingreso.

Los Cazadores de Ofertas tienen un descuento medio de 18,30 %, equivalente a
2,5 veces el promedio general (7,38 %), lo que respalda la interpretación de
sensibilidad al precio. Potenciales de Crecimiento no son clientes "recientes"
en sentido estricto: su última compra fue hace 349 días. La etiqueta expresa su
potencial —ticket y frecuencia superiores a los segmentos de bajo valor— y no una
actividad reciente; se eligió ese nombre para no prometer una recencia que los
datos no muestran.

### 7.3 Validación

La solución de k = 5 obtuvo silhouette = 0,318, Davies-Bouldin = 0,884 y
Calinski-Harabasz = 18 789,41. El silhouette es moderado: los segmentos están
separados, pero no son esferas perfectamente aisladas, algo esperable en datos de
comportamiento. La estabilidad bootstrap con 10 remuestreos dio un ARI promedio
de 0,981 (mínimo 0,969), por encima del umbral de 0,70. Las cinco etiquetas son
únicas y cada una tiene un perfil, una acción y un KPI asociados. Los valores
completos están en `docs/results/evaluacion_clustering.json`.

### 7.4 Figuras

Las figuras se generan automáticamente en `docs/images/`:

- `seleccion_k.png`: codo, silhouette y Davies-Bouldin para k = 2..10.
- `clusters_pca.png`: segmentos en el plano PCA con centroides y círculo de
  correlaciones.
- `perfil_radar.png`: perfil relativo de cada segmento.
- `segmentos_valor.png`: porcentaje de clientes vs porcentaje de ingreso.
- `perfil_relativo.png`: mapa de calor de índices relativos.
- `experimento_degradacion.png`: ARI por escenario de error.

### 7.5 Experimento de degradación

La Tabla 6 resume el experimento con tres semillas por escenario. Un ARI de 1,0
significa segmentos idénticos a la referencia; 0,0, ausencia de relación. El
umbral de estabilidad adoptado es 0,70.

**Tabla 6.** Impacto de omitir o ejecutar mal una fase (media de tres semillas).

| Escenario | Fase afectada | ARI medio | Desv. | Lectura |
|---|---|---:|---:|---|
| Recencia con fecha del sistema | Transformación | 0,994 | 0,005 | La partición se mantiene; la lectura absoluta se corrompe |
| Ruido ±20 % en precio | Calidad / transformación | 0,982 | 0,002 | La membresía resiste, pero el ingreso se altera |
| Duplicados 5 % | Preprocesamiento | 0,967 | 0,002 | La membresía resiste, pero conteos e ingreso se inflan |
| Sin filtrar estados | Preprocesamiento | 0,862 | 0,002 | Cambio moderado; incluye ingreso no realizado |
| Nulos imputados con 0 | Preprocesamiento | 0,768 | 0,002 | Cerca del umbral; distorsiona ticket y descuento |
| Sin estandarización | Transformación | 0,160 | 0,000 | Cambio severo; el ingreso domina la distancia |
| k = 2 sin evaluación | Minería | 0,020 | 0,000 | Otra granularidad: élite vs resto, no una segmentación accionable |

El resultado obliga a matizar la afirmación inicial. Los errores de
**transformación y de especificación del modelo** son los que más cambian la
partición: sin estandarización, el ARI cae a 0,160 porque el ingreso en USD domina
la distancia euclidiana; con k = 2, el ARI es 0,020 porque se obtiene otra
granularidad, no porque el algoritmo falle. En cambio, duplicados y ruido
conservan la membresía (ARI > 0,96) porque K-Means es robusto a perturbaciones
pequeñas, pero **sesgan los totales**: más pedidos, más frecuencia y más ingreso
por cliente. Los nulos imputados con cero se acercan al umbral (0,768).

La recencia con la fecha del sistema es un caso especial. Para que el escenario
sea reproducible, la fecha del sistema se fijó en el 25 de octubre de 2026
(fecha de ejecución del ensayo); en producción sería la fecha real de consulta.
El ARI es 0,994 porque sumar una constante a todos los clientes se cancela al
estandarizar: la partición se mantiene, pero la recencia absoluta se corrompe.
Los 36 días del VIP Elite pasarían a más de 700, y cualquier regla de negocio
basada en días ("contactar antes de 60 días") quedaría inválida. Es un error de
interpretación, no de membresía; por eso el pipeline fija la fecha de corte en el
máximo del dataset y no en la fecha del sistema.

La lección no es que "todas las fases valgan lo mismo", sino que el ARI mide una
sola dimensión del daño. Un tablero puede exhibir un ARI alto y, al mismo tiempo,
un ingreso inflado o una frecuencia falsa. Por eso la evaluación de la fase 6
combina métricas de partición con auditoría de ingresos, conteos y perfiles.

## 8. Discusión

### 8.1 ¿Qué fases aportan mayor valor?

En este caso, las fases 3, 4 y 6 aportaron más valor que la fase 5. K-Means es un
algoritmo conocido y su implementación es estable; lo que cambió el resultado fue
la definición del dato y la evaluación. La fase 5 también importó —el experimento
muestra que fijar k = 2 sin evaluación cambia por completo la respuesta—, pero su
calidad depende de que las fases anteriores hayan definido bien qué se agrupa y
cómo se mide. Lo que cambió el resultado fue:

- Detectar que `TotalAmount` no descuenta la promoción.
- Excluir pedidos no realizados.
- Corregir la recencia con una fecha de corte fija.
- Decidir que el ACP no reemplaza a las variables originales.
- Evaluar k con varias métricas y no solo con inercia.

La fase 2 también fue decisiva: descubrir que no hay canastas evitó aplicar una
técnica equivocada y permitió justificar el clustering.

### 8.2 ¿Qué errores afectan más?

Los errores de **definición** (qué es un cliente, qué es una venta, qué es
recencia) afectan más que los errores de **algoritmo**, pero el experimento
permite distinguir dos tipos de daño. El primero es el **cambio de partición**,
que el ARI sí mide: la falta de estandarización (ARI = 0,160) y la fijación de
k = 2 (ARI = 0,020) producen agrupamientos distintos. El segundo es el **sesgo de
magnitudes**, que el ARI no captura: duplicar 5 % de las filas, añadir ruido al
precio o desplazar la recencia con la fecha del sistema deja casi intacta la
membresía (ARI > 0,96), pero infla pedidos, frecuencia e ingreso o corrompe la
lectura temporal. Un tablero podría mostrar segmentos "estables" y, al mismo
tiempo, cifras de negocio equivocadas.

Por eso la evaluación de la fase 6 no puede reducirse a una métrica de partición.
En este proyecto se combinaron silhouette, Davies-Bouldin, Calinski-Harabasz,
estabilidad bootstrap y una auditoría explícita de ingresos y conteos en cada
escenario. La conclusión práctica es que un error de definición se detecta
comparando totales y perfiles, no solo observando el ARI.

### 8.3 Limitaciones del proceso KDD en escenarios reales

1. **Datos simulados.** El dataset es sintético; los hallazgos ilustran el
   proceso, no una empresa real. En un caso real habría que validar la calidad y
   el significado de las variables con el área de negocio.
2. **Análisis descriptivo.** El clustering no establece causalidad. Que un
   segmento tenga mayor ingreso no significa que una acción vaya a aumentarlo.
3. **Validación externa pendiente.** No se contó con un panel de expertos que
   valide las etiquetas. La estabilidad bootstrap mide reproducibilidad, no
   validez de negocio.
4. **Un solo algoritmo.** K-Means asume clusters aproximadamente esféricos y
   requiere elegir k. Otros métodos (jerárquico, DBSCAN, mezclas gaussianas)
   podrían producir segmentos distintos; compararlos sería una extensión.
5. **Impacto estimado con supuestos.** Las tasas de retención y reactivación son
   hipótesis conservadoras; deben validarse con un piloto y con el área
   comercial.
6. **Segmento élite pequeño.** VIP Elite representa 1,29 % de los clientes (375
   personas) y concentra 34,78 % del ingreso. Se mantuvo separado por su valor
   estratégico, aunque su tamaño exige prudencia: un solo cliente extremo puede
   mover sus promedios y cualquier campaña debe evaluarse caso por caso.
7. **Asociación limitada.** Al no haber canastas, las reglas de co-compra son
   un complemento exploratorio y no permiten recomendaciones de *basket*.

### 8.4 Relación con CRISP-DM y con los sistemas de información

KDD y CRISP-DM coinciden en la necesidad de entender el negocio antes de
modelar y de desplegar el conocimiento después de evaluarlo. La diferencia
principal es el énfasis: KDD describe el proceso de descubrimiento; CRISP-DM
describe el ciclo de vida del proyecto. En un sistema de información, la fase 7
de KDD se materializa en un tablero, una regla de negocio o una campaña; sin esa
integración, el conocimiento queda en un informe y no genera valor.

## 9. Conclusiones

1. **KDD es un proceso, no un algoritmo.** La calidad del conocimiento depende
   de la articulación de las siete fases, no de la sofisticación de la técnica.
2. **La omisión o mala ejecución de fases reduce la calidad del análisis.** El
   experimento de degradación lo demuestra, pero con matices: la falta de
   estandarización (ARI = 0,160) y la fijación de k sin evaluación (ARI = 0,020)
   cambian la partición; duplicados y ruido preservan la membresía (ARI > 0,96)
   pero sesgan los totales. La calidad no se mide con una sola métrica.
3. **La fase 6 es la que convierte un modelo en conocimiento.** Sin métricas,
   estabilidad y validación con el negocio, el clustering es solo una tabla de
   grupos.
4. **El caso produjo conocimiento accionable.** Se identificaron cinco
   segmentos; 17,93 % de los clientes concentra 73,50 % del ingreso; el segmento
   VIP Elite (1,29 % de los clientes) explica 34,78 % del ingreso; los Cazadores
   de Ofertas tienen 2,5 veces el descuento medio y aportan 4,26 % del ingreso.
5. **La aplicación práctica permite ventajas competitivas reales** cuando el
   conocimiento se integra en decisiones de retención, precio y asignación de
   presupuesto. En este caso, la prioridad es proteger a los VIP y reactivar de
   forma selectiva a los clientes de bajo valor, no aplicar campañas masivas.

La respuesta a la pregunta de investigación es, entonces, que las fases de KDD
se aplican de forma iterativa y que su impacto en la generación de conocimiento
útil depende de tres condiciones: datos bien seleccionados y tratados,
transformaciones justificadas y evaluación que conecte el resultado con una
decisión.

## 10. Recomendaciones

1. **No omitir fases.** Documentar cada decisión de selección, limpieza,
   transformación y evaluación, incluso cuando el dataset no presente problemas.
2. **Priorizar la calidad de los datos.** Validar fórmulas de negocio (descuento,
   impuesto, envío), estados de pedido y fechas de corte antes de modelar.
3. **Validar con expertos del dominio.** Las etiquetas y los umbrales deben ser
   revisados por el área comercial; la estadística no reemplaza el conocimiento
   del negocio.
4. **Integrar KDD en los sistemas de información.** Publicar los segmentos en un
   tablero, definir responsables y monitorear los KPI por segmento.
5. **Ejecutar un piloto antes de escalar.** Medir el retorno de las acciones
   sobre VIP Elite, VIP Leales y Cazadores de Ofertas antes de asignar
   presupuesto masivo.
6. **Revisar el modelo periódicamente.** La recencia y los segmentos cambian; el
   pipeline debe recalcularse con una fecha de corte y comparar la estabilidad.
7. **Ampliar la comparación de algoritmos.** Contrastar K-Means con métodos
   jerárquicos o DBSCAN para evaluar si la estructura de segmentos se mantiene.
8. **Cuidar la privacidad.** Aunque el dataset sea simulado, el pipeline debe
   excluir nombres y datos personales del conjunto modelado.
9. **No confundir estabilidad con validez.** Un ARI alto indica que la partición
   se reproduce, no que los totales sean correctos; auditar ingresos, pedidos y
   perfiles en cada ejecución.

## 11. Referencias

Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.
*Proceedings of the 20th International Conference on Very Large Data Bases*,
487–499.

Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE
Transactions on Pattern Analysis and Machine Intelligence, 1*(2), 224–227.
https://doi.org/10.1109/TPAMI.1979.4766909

Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From data mining to
knowledge discovery in databases. *AI Magazine, 17*(3), 37–54.
https://doi.org/10.1609/aimag.v17i3.1230

Han, J., Kamber, M., & Pei, J. (2012). *Data mining: Concepts and techniques*
(3rd ed.). Morgan Kaufmann.

Hernández Orallo, J., Ramírez Quintana, M. J., & Ferri Ramírez, C. (2004).
*Introducción a la minería de datos*. Pearson Educación.

Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of
Classification, 2*(1), 193–218. https://doi.org/10.1007/BF01908075

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern
Recognition Letters, 31*(8), 651–666.
https://doi.org/10.1016/j.patrec.2009.09.011

Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data: An introduction
to cluster analysis*. Wiley.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and
validation of cluster analysis. *Journal of Computational and Applied
Mathematics, 20*, 53–65.
https://doi.org/10.1016/0377-0427(87)90125-7

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data
mining. *Proceedings of the 4th International Conference on the Practical
Applications of Knowledge Discovery and Data Mining*, 29–39.

---

## Anexo A. Reproducibilidad

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
python -m pytest -q
```

El pipeline genera:

- `docs/results/kdd_log.json`: métricas y decisiones por fase.
- `docs/results/reporte_kdd.md`: reporte técnico.
- `docs/results/segmentos.json`: perfiles y evaluación.
- `docs/results/conocimiento_acciones.md`: acciones por segmento.
- `docs/results/experimento_degradacion.json`: evidencia del experimento.
- `docs/images/*.png`: figuras para el documento.

## Anexo B. Mapa de evidencia

| Afirmación del ensayo | Archivo de evidencia |
|---|---|
| El dataset es simulado | `docs/results/perfil_dataset.json` |
| La recencia usa fecha de corte | `docs/results/reporte_transformacion.json` |
| `TotalAmount` ignora el descuento | `docs/results/reporte_limpieza.json` |
| k se eligió con métricas | `docs/results/kdd_log.json`, `seleccion_k.png` |
| Las etiquetas son únicas | `docs/results/segmentos.json` |
| La solución es estable | `docs/results/evaluacion_clustering.json` |
| Omitir fases degrada el resultado | `docs/results/experimento_degradacion.json` |
| Cada segmento tiene acciones | `docs/results/conocimiento_acciones.md` |

## Anexo C. Lista de verificación del formato de entrega

- [ ] Completar nombre, facultad, asignatura y fecha en el encabezado.
- [ ] Convertir este documento a PDF.
- [ ] Revisar ortografía y redacción.
- [ ] Verificar que el PDF tenga entre 8 y 12 páginas.
- [ ] Adjuntar las figuras de `docs/images/` si el formato lo permite.
- [ ] Subir el PDF al espacio de entrega antes del plazo.
