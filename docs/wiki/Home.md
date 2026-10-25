# Proyecto KDD — Segmentación de clientes de una tienda en línea

Este repositorio implementa las **siete fases del proceso KDD** (*Knowledge
Discovery in Databases*) sobre un dataset de ventas de una tienda en línea. El
objetivo no es solo ejecutar K-Means, sino documentar cómo cada fase transforma
los datos y qué decisiones metodológicas permiten (o impiden) convertir los
resultados en conocimiento accionable.

El ensayo académico completo está en
[`propuesta_PDF.md`](../../propuesta_PDF.md); la evidencia técnica generada por
el pipeline está en [`docs/results/`](../results/) y las figuras en
[`docs/images/`](../images/).

## Pregunta de investigación

¿Cómo se aplican las fases del proceso KDD en un caso práctico de minería de
datos y cuál es su impacto en la generación de conocimiento útil para la toma de
decisiones?

## Las siete fases y su implementación

| Fase KDD | Módulo principal | Evidencia generada |
|---|---|---|
| 1. Comprensión del dominio | `main.py`, `src/config/settings.py` | Pregunta, objetivos y criterios de éxito |
| 2. Selección de datos | `src/data/data_loader.py` | `perfil_dataset.json` |
| 3. Limpieza y preprocesamiento | `src/data/data_cleaner.py` | `reporte_limpieza.json` |
| 4. Transformación y reducción | `src/data/transformation.py`, `src/clustering/pca.py` | `reporte_transformacion.json`, `acp_varianza.png` |
| 5. Minería de datos | `src/clustering/kmeans.py`, `src/analysis/evaluation.py`, `src/analysis/association.py` | `seleccion_k.png`, `reglas_asociacion.json` |
| 6. Interpretación y evaluación | `src/clustering/segmentation.py`, `src/experiments/degradation.py` | `evaluacion_clustering.json`, `experimento_degradacion.json` |
| 7. Uso del conocimiento | `src/kdd/knowledge.py`, `src/reports/kdd_report.py` | `conocimiento_acciones.md`, `reporte_kdd.md` |

El registro cronológico de las fases, con decisiones, advertencias y métricas,
se guarda en [`kdd_log.json`](../results/kdd_log.json) y se renderiza en
[`kdd_fases.md`](../results/kdd_fases.md).

## Estructura del proyecto

```text
Analisis de Datos/
├── main.py                     # Orquestador de las siete fases
├── amazon.csv                  # Dataset de trabajo
├── propuesta_PDF.md            # Ensayo académico (marco teórico y resultados)
├── src/
│   ├── config/                 # Parámetros metodológicos centralizados
│   ├── data/                   # Carga, perfilado, limpieza y transformación
│   ├── clustering/             # ACP, K-Means y segmentación
│   ├── analysis/               # Estadística, evaluación y reglas de asociación
│   ├── experiments/            # Experimento controlado de degradación
│   ├── kdd/                    # Registro de fases y traducción a acciones
│   ├── reports/                # Reporte técnico automático
│   ├── visualization/          # Gráficos exploratorios
│   └── core/                   # Utilidades de formato y guardado de figuras
├── docs/
│   ├── results/                # Evidencia JSON/Markdown del pipeline
│   ├── images/                 # Figuras generadas
│   └── wiki/                   # Esta documentación
└── tests/                      # Pruebas de datos, clustering y evaluación
```

## Ejecución

```bash
python -m pip install -r requirements.txt

# Pipeline completo (incluye selección de k y experimento de degradación)
python main.py

# Modo rápido: k fijo, sin experimento
python main.py --quick --no-experiment
```

El pipeline es reproducible: la recencia se calcula contra la fecha máxima del
dataset (`2024-12-29`), no contra la fecha del sistema. Las semillas y los
umbrales están en `src/config/settings.py`.

## Decisiones metodológicas clave

1. **El dataset es sintético.** El perfilado detecta 100 000 filas exactas, cero
   nulos, cero duplicados, filas anuales casi uniformes y un único producto por
   pedido. Se declara como simulación y se analiza como caso de estudio, no como
   evidencia del mercado real.
2. **`TotalAmount` ignora el descuento.** Se recalcula
   `Ingreso_Neto = Quantity × UnitPrice × (1 − Discount)` y se cuantifica la
   sobreestimación del dataset original (USD 9 241 433,05).
3. **Las devoluciones y cancelaciones se excluyen** del análisis de valor. Los
   pedidos sin cliente identificable se conservan en el perfilado, pero no en la
   base analítica.
4. **K-Means trabaja con las cinco variables estandarizadas** (ingreso, ticket,
   frecuencia, recencia y descuento), no con dos componentes del ACP. El ACP se
   usa para visualización y diagnóstico.
5. **`k` no se fija a priori.** Se evalúa `k = 2..10` con codo, silhouette,
   Davies-Bouldin, Calinski-Harabasz y tamaño mínimo de cluster. Se elige `k = 5`
   (silhouette = 0,318; Davies-Bouldin = 0,884). Ningún `k` de 3 a 6 alcanzó el
   5 % mínimo por cluster, por lo que el umbral se relajó al 1 % técnico y se
   conservó el micro-segmento **VIP Elite** (1,29 % de clientes, 34,78 % del
   ingreso). La relajación queda registrada en el reporte.
6. **La estabilidad se mide con bootstrap (ARI).** El agrupamiento final alcanza
   un ARI medio de 0,981, por encima del umbral de 0,70.
7. **Las reglas de asociación son un complemento exploratorio.** Cada pedido
   contiene un solo producto, por lo que Apriori a nivel de pedido no es
   aplicable; se calculan co-ocurrencias a nivel de cliente y se justifica la
   decisión en el informe.

## Evidencia y resultados

- [`reporte_kdd.md`](../results/reporte_kdd.md): reporte técnico completo.
- [`segmentos.json`](../results/segmentos.json): perfiles, tamaños e ingresos.
- [`evaluacion_clustering.json`](../results/evaluacion_clustering.json):
  métricas, estabilidad y comparación de alternativas.
- [`experimento_degradacion.json`](../results/experimento_degradacion.json):
  efecto de duplicados, nulos y ruido sobre el ARI.
- [`conocimiento_acciones.md`](../results/conocimiento_acciones.md): acciones,
  KPI y riesgos por segmento.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas cubren perfilado y limpieza, métricas de selección de `k`,
estabilidad bootstrap, etiquetado único de segmentos y registro de fases.

## Limitaciones

- El dataset es sintético y no representa el comportamiento real de Amazon.
- Cada pedido contiene un solo producto, lo que impide un análisis de canasta
  clásico.
- La segmentación es descriptiva, no causal: los segmentos orientan decisiones,
  pero no demuestran el efecto de una campaña.
- El ingreso se analiza en USD corrientes, sin ajuste inflacionario.
