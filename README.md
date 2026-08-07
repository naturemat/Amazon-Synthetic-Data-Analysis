# Amazon Sales Data Analysis

A modular Python project for exploratory data analysis (EDA), visualization, and automatic customer segmentation of Amazon sales data using a machine learning pipeline based on PCA and K-Means clustering.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-pep8-000000.svg)](https://peps.python.org/pep-0008/)

## Overview

The project turns a raw sales CSV into business insight through a layered, reusable architecture:

- **Data pipeline** - load, clean, and prepare the dataset
- **Exploratory analysis** - descriptive statistics and top/bottom rankings
- **Visualization** - distributions, time series, correlations, and segmentation plots
- **Customer segmentation (ML)** - a reproducible machine learning workflow that groups customers into actionable segments using dimensionality reduction (PCA) and K-Means clustering

## Core Features

- Duplicate removal, date parsing, and handling of irrelevant or missing columns
- Numeric and categorical descriptive statistics
- Top/bottom analysis by product, category, customer, state, and seller
- Monthly revenue time series analysis
- Correlation matrix and bivariate scatter plot matrix
- Customer behavioral profiling (similar to RFM)
- PCA for dimensionality reduction
- K-Means clustering with the elbow method and inertia analysis
- Automatic segment interpretation based on relative cluster profiles

## The Machine Learning Pipeline

The segmentation workflow in `src/clustering/` is fully reproducible and configuration-driven:

```
Clean data
    -> Agregate per customer (TotalAmount, Ticket, Frequency, Discount, Recency)
    -> StandardScaler (center and scale)
    -> PCA dimensionality reduction (default: 2 components)
    -> K-Means clustering (configurable k, k-means++, multiple init)
    -> Relative cluster profile (vs. global average)
    -> Automatic segment labeling
```

### Segment Labels

After clustering, each group is automatically labeled depending on its relative profile:

| Segment | Profile |
|---|---|
| Premium (High Value) | High income, high frequency, recent purchase |
| Frequent (Loyal) | Above-average purchase frequency |
| Occasional (Price Sensitive) | High average discount ratio |
| Inactive (At Risk) | Long time since last purchase |

Cluster validation is supported through the elbow method, total inertia decomposition (intra/inter-class), and PCA loadings with a correlation circle.

## Getting Started

### Prerequisites

- Python 3.8 or later
- `pip`

### Installation

```bash
git clone <repository-url>
cd analisis-de-datos

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Usage

Run the complete analysis pipeline (data loading, cleaning, EDA, visualizations, and segmentation):

```bash
python main.py
```

`main.py` also exposes reusable entry points for testing and partial pipelines:

- `run_full_analysis()` - the entire workflow
- `run_data_loading_only()` - load and inspect the dataset
- `run_visualization_only(df)` - distribution and time series plots
- `run_segmentation_only(df, original_df)` - complete segmentation only

### Running Tests

```bash
pytest tests/
```

### Jupyter Notebooks

An interactive implementation of the pipeline is available:

```bash
jupyter notebook notebooks/analisis_exploratorio.ipynb
```

## Project Structure

The repository follows a layered architecture, where each layer has the single responsibility:

```
analisis-de-datos/
├── main.py                 # Entry point and pipeline orchestration
├── amazon.csv              # Dataset used for the analysis
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT license
├── README.md
├── src/
│   ├── config/             # Centralized configuration (columns, models, plot styling)
│   ├── core/               # Formatters and display helpers
│   ├── data/               # DataLoader and DataCleaner
│   ├── analysis/           # EDA and statistical analyzer
│   ├── visualization/      # Distribution and chart plotter
│   ├── clustering/         # PCAModel, KMeansModel, CustomerSegmenter
│   └── reports/            # Summary report generator
├── tests/                  # pytest test suite
├── notebooks/              # Jupyter notebooks
└── docs/wiki/              # Project documentation
```

### Layer Reference

| Directory | Responsibility |
|---|---|
| `src/config/` | Centralized settings: columns, data types, clustering params, plot styling |
| `src/core/` | Number/currency formatters and environment configuration |
| `src/data/` | `DataLoader` (CSV ingestion with typed dtypes) and `DataCleaner` |
| `src/analysis/` | `ExploratoryDataAnalysis` and `StatisticalAnalyzer` |
| `src/visualization/` | `DistributionPlotter` and `ChartPlotter` |
| `src/clustering/` | `PCAModel`, `KMeansModel`, and the `CustomerSegmenter` pipeline |
| `src/reports/` | `ReportGenerator` for console-readable summaries |

## Configuration

All knobs are centralized in `src/config/settings.py`:

| Setting | Purpose |
|---|---|
| `DATASET`, `DTYPES` | Source CSV path and column data types |
| `COLS_TO_DROP` | Columns removed during cleaning |
| `NUMERIC_VARS`, `CATEGORICAL_VARS` | Variables used in distribution analysis |
| `ID_VARS` | Identity columns for top/bottom analysis |
| `CLUSTERING_VARS` | Behavioral features feed into the clustering pipeline |
| `KMEANS_CONFIG`, `PCA_CONFIG` | Model hyperparameters (k, initialization, random state) |
| `PLOT_STYLE`, `FIGURE_SIZES` | Global plot styling |

## Documentation

Extended guide and module reference are available in `docs/wiki/Home.md`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Mateo Cobo