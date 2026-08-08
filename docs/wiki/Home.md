# Amazon Sales Data Analysis - Wiki

This wiki documents the architecture, modules, and usage of the Amazon Sales Data Analysis project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Modules Overview](#modules-overview)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Configuration](#configuration)
6. [Running Tests](#running-tests)
7. [Contributing](#contributing)
8. [License](#license)

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- `pip`

### Installation

```bash
git clone <repository-url>
cd analisis-de-datos

python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux / macOS

pip install -r requirements.txt
```

### Running the Project

Run the full analysis pipeline:

```bash
python main.py
```

An interactive version of the same workflow is provided as a Jupyter notebook:

```bash
jupyter notebook notebooks/analisis_exploratorio.ipynb
```

---

## Project Structure

```
analisis-de-datos/
├── main.py                 # Entry point and pipeline orchestration
├── amazon.csv              # Dataset used for the analysis
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT license
├── src/
│   ├── config/             # Centralized configuration
│   ├── core/               # Formatters and display helpers
│   ├── data/               # DataLoader and DataCleaner
│   ├── analysis/           # EDA and statistical analyzer
│   ├── visualization/      # Distribution and chart plotter
│   ├── clustering/         # PCA, K-Means, and segmentation
│   └── reports/            # Summary report generator
├── tests/                  # pytest test suite
├── notebooks/              # Jupyter notebooks
└── docs/wiki/              # This documentation
```

---

## Modules Overview

### Data Layer (`src/data/`)

- **DataLoader** - loads the CSV with configured data types and date parsing, and exposes dataset information.
- **DataCleaner** - removes duplicates, converts date columns, drops irrelevant columns, and reports missing values.

### Analysis Layer (`src/analysis/`)

- **ExploratoryDataAnalysis** - numeric/categorical statistics and top/bottom rankings by product, customer, category, and state.
- **StatisticalAnalyzer** - descriptive statistics, customer aggregation, inertia analysis, cluster profiles, and cluster labeling.

### Visualization Layer (`src/visualization/`)

- **DistributionPlotter** - histograms and boxplots for numeric variables; categorical distributions with income overlay.
- **ChartPlotter** - time series, correlation heatmaps, scatter matrix, ID revenue distribution, and top/bottom bar charts.

### Clustering Layer (`src/clustering/`)

- **PCAModel** - wraps `StandardScaler` and `PCA`, exposing loadings, explained variance, and scatter plots.
- **KMeansModel** - wraps `sklearn` K-Means with configurable parameters, plus the elbow method and inertia decomposition.
- **CustomerSegmenter** - orchestrates aggregation, standardization, PCA, K-Means, labeling, and segment visualizations.

### Reports Layer (`src/reports/`)

- **ReportGenerator** - prints data information and ranking summaries to the console.

---

## Machine Learning Pipeline

`CustomerSegmenter` implements the end-to-end segmentation workflow:

1. Aggregate sales per customer: total income, average ticket, frequency, average discount, and days since last purchase.
2. Standardize features (center and scale).
3. Reduce dimensionality with PCA (default 2 components).
4. Cluster with K-Means (k-means++, multiple initialization, configurable k).
5. Compute relative cluster profiles against the global mean.
6. Assign business labels automatically.

### Segment Labels

| Segment | Relative profile condition |
|---|---|
| Premium (High Value) | Income > 1.2x, frequency > 1.2x, recency < 0.8x |
| Frequent (Loyal) | Frequency > 1.1x |
| Occasional (Price Sensitive) | Average discount > 1.05x |
| Inactive (At Risk) | Everything else |

Each segment is diagnosed through radar charts, cluster distribution pies, PCA scatter with correlation circles, and product-level top/bottom analysis.

---

## Results & Conclusions

The following figures are produced by a full run of `python main.py` over `amazon.csv`. The full numeric summary is available in `docs/conclusions_summary.json`, and the detailed analysis lives in the project [README](../../README.md#results--conclusions).

| Metric | Value |
|---|---|
| Records | 100,000 |
| Unique customers | 30,243 |
| Total revenue | USD 132,424,396.57 |
| Variance explained by PC1 + PC2 | 66.0% |

![Customer segments in PCA space](../images/segmentation_pca_clusters.png)

*Customer segments projected on PC1/PC2 with the loading circle.*

Key findings:

- The ~19.5% of customers in the two premium clusters generate about 75% of total revenue.
- Around 80% of customers are low-frequency and low-spend (Occasional and Inactive), representing the main retention opportunity.
- Two clusters currently receive the same automatic label (Premium), a candidate for refining the labeling thresholds.

---

## Configuration

`src/config/settings.py` centralizes all tunable parameters:

- `DATASET`, `DTYPES`, `COLS_TO_DROP` - data source and cleaning behavior
- `NUMERIC_VARS`, `CATEGORICAL_VARS`, `ID_VARS` - analysis targets
- `CLUSTERING_VARS` - the behavioral features used by the clustering pipeline
- `KMEANS_CONFIG` - number of clusters, initialization, `n_init`, `max_iter`, random state
- `PCA_CONFIG` - number of components and random state
- `PLOT_STYLE`, `FIGURE_SIZES` - global matplotlib styling

Change these values to alter the analysis without touching pipeline code.

---

## Running Tests

```bash
pytest tests/
```

Run a specific test file:

```bash
pytest tests/test_clustering.py
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and keep the test suite green.
4. Update the documentation if the public behavior changes.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.