# Aplicación práctica de las fases de KDD — Segmentación de clientes

Este repositorio implementa un proceso **KDD (Knowledge Discovery in Databases)**
completo sobre un dataset de ventas de una tienda en línea. El objetivo no es
solo ejecutar K-Means, sino demostrar cómo cada fase del proceso aporta (o
degrada) el conocimiento final.

El ensayo académico con el marco teórico, la discusión y las referencias está en
[`propuesta_PDF.md`](propuesta_PDF.md). La evidencia técnica generada por el
pipeline está en [`docs/results/`](docs/results/) y las figuras en
[`docs/images/`](docs/images/).

## Pregunta y objetivo

**Pregunta:** ¿qué segmentos de clientes existen según su valor, frecuencia,
sensibilidad al descuento y recencia, y cómo debe priorizar la organización sus
acciones de retención y crecimiento?

**Objetivo:** segmentar la base de clientes y traducir los segmentos en acciones
comerciales priorizadas, cuantificando el ingreso en riesgo y el potencial de
crecimiento.

## Las siete fases y su implementación

| Fase KDD | Implementación | Evidencia |
|---|---|---|
| 1. Comprensión del dominio | Pregunta, objetivo, KPI y criterios de éxito en `settings.py` | `docs/results/kdd_log.json` |
| 2. Selección de datos | `DataLoader` + perfilado y detección de dataset simulado | `docs/results/perfil_dataset.json` |
| 3. Preprocesamiento | `DataCleaner`: duplicados, nulos, consistencia, filtro de estados, ingreso neto | `docs/results/reporte_limpieza.json` |
| 4. Transformación | `CustomerTransformer`: agregación por cliente, recencia con fecha de corte, RFM, winsorización, estandarización, ACP | `docs/results/reporte_transformacion.json` |
| 5. Minería | `KMeansModel` + selección de k con codo, silhouette, Davies-Bouldin y Calinski-Harabasz | `docs/results/kdd_log.json`, `docs/images/seleccion_k.png` |
| 6. Interpretación y evaluación | `CustomerSegmenter`: perfiles, etiquetas únicas, estabilidad bootstrap (ARI), experimento de degradación | `docs/results/evaluacion_clustering.json`, `docs/results/experimento_degradacion.json` |
| 7. Uso del conocimiento | `knowledge.py`: acciones, KPI, riesgos e impacto estimado por segmento | `docs/results/conocimiento_acciones.md` |

## Estructura del proyecto

```
.
├── amazon.csv                  # Dataset de ventas
├── main.py                     # Orquestador de las siete fases
├── propuesta_PDF.md            # Ensayo académico (base para el PDF)
├── requirements.txt
├── src/
│   ├── config/settings.py      # Parámetros y decisiones metodológicas
│   ├── core/                   # Formato, figuras, utilidades
│   ├── data/                   # Carga, limpieza, transformación, diccionario
│   ├── analysis/               # EDA, estadística, evaluación, asociación
│   ├── clustering/             # PCA, K-Means, segmentación
│   ├── experiments/            # Experimento controlado de degradación
│   ├── kdd/                    # Registro de fases y capa de conocimiento
│   ├── reports/                # Reporte técnico automático
│   └── visualization/          # Gráficas descriptivas
├── tests/                      # Pruebas con pytest
└── docs/
    ├── images/                 # Figuras del documento
    └── results/                # Métricas, perfiles y reportes
```

## Instalación y ejecución

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python main.py                 # pipeline completo + experimento
python main.py --no-experiment # sin experimento de degradación
python main.py --quick         # evaluación reducida (desarrollo)
python -m pytest -q            # pruebas
```

Salidas principales:

- `docs/results/kdd_log.json`: métricas, decisiones y advertencias por fase.
- `docs/results/reporte_kdd.md`: reporte técnico completo.
- `docs/results/segmentos.json`: perfiles y evaluación de los segmentos.
- `docs/results/conocimiento_acciones.md`: acciones y KPI por segmento.
- `docs/results/experimento_degradacion.json`: impacto de omitir fases.
- `docs/images/*.png`: figuras listas para el documento.

## Decisiones metodológicas que el documento debe defender

1. **El dataset es simulado.** Presenta 100 000 filas exactas, cero nulos, cero
   duplicados, ~20 000 filas por año y un solo producto por pedido. Se declara
   en la metodología; la actividad permite datos reales o simulados.
2. **La recencia usa la fecha máxima del dataset** (`2024-12-29`), no
   `pd.Timestamp.now()`. Así el resultado es reproducible y no cambia cada día.
3. **Se excluyen pedidos cancelados, devueltos y pendientes** del conjunto
   analítico (10 180 filas). Incluirlos inflaría ingreso y recencia.
4. **`TotalAmount` ignora el descuento.** Se usa `Ingreso_Neto =
   Quantity × UnitPrice × (1 − Discount)` y se cuantifica la sobreestimación
   (USD 9 241 433,05 en el dataset completo).
5. **El ACP no reemplaza a las variables originales.** Con 2 componentes se
   retiene alrededor del 69 % de la varianza; K-Means trabaja sobre las cinco
   variables estandarizadas y el ACP se usa para visualización y diagnóstico.
6. **k no se fija a priori.** Se evalúa k = 2..10 con codo, silhouette,
   Davies-Bouldin, Calinski-Harabasz y tamaño mínimo de cluster; la regla de
   selección está escrita en el código y en el reporte. Se elige k = 5
   (silhouette = 0,318; Davies-Bouldin = 0,884). Ningún k de 3 a 6 alcanzó el
   5 % mínimo por cluster, por lo que el umbral se relajó al 1 % técnico y la
   relajación quedó registrada; no se fusionó el segmento VIP Elite (1,29 % de
   los clientes, 34,78 % del ingreso) para cumplir una cuota.
7. **Las etiquetas son únicas.** Dos clusters no pueden compartir etiqueta; si
   comparten cuadrante, se diferencian por frecuencia, ticket o descuento.
8. **No se aplica Apriori clásico** porque cada pedido tiene un solo producto.
   Se documenta la limitación y se calculan reglas de co-compra por cliente como
   complemento exploratorio.
9. **La evaluación no se limita a la inercia.** Se reportan silhouette,
   Davies-Bouldin, Calinski-Harabasz y estabilidad bootstrap con ARI.
10. **El experimento de degradación** inyecta duplicados, nulos, ruido y errores
    de transformación para medir con ARI cuánto se alejan los segmentos de la
    solución de referencia.

## Resultados principales (k = 5)

| Segmento | Clientes | % clientes | % ingreso | Ticket | Frecuencia | Descuento | Recencia (días) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VIP Leales | 4 819 | 16,63 | 38,72 | 1 714,39 | 5,03 | 6,88 % | 347,2 |
| VIP Elite | 375 | 1,29 | 34,78 | 1 621,41 | 59,46 | 7,45 % | 36,2 |
| Potenciales de Crecimiento | 11 270 | 38,90 | 16,32 | 622,44 | 2,26 | 5,17 % | 349,0 |
| Ocasionales Inactivos | 7 408 | 25,57 | 5,92 | 581,74 | 1,35 | 3,55 % | 1 266,5 |
| Cazadores de Ofertas | 5 099 | 17,60 | 4,26 | 525,88 | 1,53 | 18,30 % | 815,0 |

- 17,93 % de los clientes (VIP Elite + VIP Leales) genera 73,50 % del ingreso.
- Estabilidad bootstrap: ARI medio = 0,981 (mínimo 0,969, 10 remuestreos).
- Experimento de degradación: sin estandarización ARI = 0,160; k = 2 ARI = 0,020;
  nulos imputados con cero ARI = 0,768. Duplicados y ruido preservan la partición
  (ARI > 0,96) pero sesgan los totales.

## Limitaciones

- El dataset es simulado; los hallazgos ilustran el proceso KDD, no una empresa
  real.
- El análisis es descriptivo y no establece causalidad.
- Las etiquetas y el impacto estimado son interpretaciones que deben validarse
  con expertos de negocio y con datos nuevos.
- La estabilidad bootstrap mide reproducibilidad del algoritmo, no validez
  externa.

## Referencias principales

- Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From data mining to
  knowledge discovery in databases. *AI Magazine, 17*(3), 37–54.
- Han, J., Kamber, M., & Pei, J. (2012). *Data mining: Concepts and techniques*
  (3rd ed.). Morgan Kaufmann.
- Hernández Orallo, J., Ramírez Quintana, M. J., & Ferri Ramírez, C. (2004).
  *Introducción a la minería de datos*. Pearson.
