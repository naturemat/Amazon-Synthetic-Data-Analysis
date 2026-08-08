"""
Amazon Sales Data Analysis - Main Entry Point

This is the main entry point for the modular Amazon Sales Data Analysis project.
It orchestrates the entire analysis pipeline using the modular architecture.
"""

import sys
import pandas as pd

# Core configuration
from src.core.formatters import (
    human_format,
    configure_pandas_display,
    configure_matplotlib_style,
)

# Data layer
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner

# Analysis layer
from src.analysis.eda import ExploratoryDataAnalysis

# Visualization layer
from src.visualization.plots import DistributionPlotter
from src.visualization.charts import ChartPlotter

# Clustering layer
from src.clustering.segmentation import CustomerSegmenter
from src.clustering.pca import PCAModel
from src.clustering.kmeans import KMeansModel

# Config
from src.config.settings import (
    NUMERIC_VARS,
    CATEGORICAL_VARS,
    ID_VARS,
    CLUSTERING_VARS,
    KMEANS_CONFIG,
)


def run_full_analysis():
    """Run the complete analysis pipeline."""
    
    # Configure display and plotting
    configure_pandas_display()
    configure_matplotlib_style()
    
    # ========================================
    # 1. DATA LOADING
    # ========================================
    print("\n" + "="*80)
    print("1. CARGA DE DATOS")
    print("="*80)
    
    loader = DataLoader("amazon.csv")
    DF = loader.load()
    loader.get_info()
    
    # ========================================
    # 2. DATA CLEANING
    # ========================================
    print("\n" + "="*80)
    print("2. LIMPIEZA DE DATOS")
    print("="*80)
    
    cleaner = DataCleaner(DF)
    cleaner.remove_duplicates()
    cleaner.convert_date_column()
    cleaner.drop_columns()
    cleaner.check_missing_values()
    cleaner.get_total_income()
    
    df = cleaner.get_processed_data()
    
    # ========================================
    # 3. BASIC STATISTICS
    # ========================================
    print("\n" + "="*80)
    print("3. ESTADÍSTICAS BÁSICAS")
    print("="*80)
    
    eda = ExploratoryDataAnalysis(df, DF)
    eda.basic_statistics_numeric()
    eda.basic_statistics_categorical()
    
    # ========================================
    # 4. DISTRIBUTION PLOTS
    # ========================================
    print("\n" + "="*80)
    print("4. GRÁFICAS DE DISTRIBUCIÓN")
    print("="*80)
    
    dist_plotter = DistributionPlotter(df)
    dist_plotter.plot_numeric_distributions(NUMERIC_VARS)
    dist_plotter.plot_categorical_distributions(CATEGORICAL_VARS)
    
    # ========================================
    # 5. TIME SERIES ANALYSIS
    # ========================================
    print("\n" + "="*80)
    print("5. GRÁFICAS DE INGRESO X TIEMPO")
    print("="*80)
    
    chart_plotter = ChartPlotter(df)
    chart_plotter.plot_time_series()
    
    # ========================================
    # 6. ID VARIABLE DISTRIBUTION
    # ========================================
    print("\n" + "="*80)
    print("6. DISTRIBUCIÓN DE INGRESOS X ID + TOP/BOTTOM")
    print("="*80)
    
    chart_plotter.plot_id_distribution(ID_VARS)
    
    # ========================================
    # 7. TOP/BOTTOM BY ID
    # ========================================
    print("\n" + "="*80)
    print("7. TOP 10 Y BOTTOM 10 POR ID")
    print("="*80)
    
    # ========================================
    # 8. TOP/BOTTOM BY CATEGORY
    # ========================================
    print("\n" + "="*80)
    print("8. TOP 1 Y BOTTOM 1 PRODUCTOS POR CATEGORÍA")
    print("="*80)
    
    df_plot_cat = eda.get_top_bottom_products_by_category()
    chart_plotter.plot_top_bottom_by_category(df_plot_cat)
    
    # ========================================
    # 9. TOP/BOTTOM BY STATE
    # ========================================
    print("\n" + "="*80)
    print("9. TOP 1 Y BOTTOM 1 PRODUCTOS POR ESTADO")
    print("="*80)
    
    df_plot_state = eda.get_top_bottom_products_by_state()
    chart_plotter.plot_top_bottom_by_state(df_plot_state)
    
    # ========================================
    # 10. TOP PRODUCTS BY QUANTITY
    # ========================================
    print("\n" + "="*80)
    print("10. TOP 5 PRODUCTOS MÁS VENDIDOS")
    print("="*80)
    
    eda.get_top_products_by_quantity()
    
    # ========================================
    # 11. TOP PRODUCTS BY INCOME
    # ========================================
    print("\n" + "="*80)
    print("11. TOP 5 PRODUCTOS CON MÁS INGRESOS")
    print("="*80)
    
    eda.get_top_products_by_income()
    
    # ========================================
    # 12. TOP CUSTOMERS BY INCOME
    # ========================================
    print("\n" + "="*80)
    print("12. TOP 5 CLIENTES CON MÁS INGRESOS")
    print("="*80)
    
    eda.get_top_customers_by_income()
    
    # ========================================
    # 13. CORRELATION MATRIX
    # ========================================
    print("\n" + "="*80)
    print("13. MATRIZ DE CORRELACIONES")
    print("="*80)
    
    chart_plotter.plot_correlation_matrix(NUMERIC_VARS)
    
    # ========================================
    # 14. SCATTER PLOTS
    # ========================================
    print("\n" + "="*80)
    print("14. GRÁFICAS DE DISPERSIÓN")
    print("="*80)
    
    chart_plotter.plot_scatter_matrix(NUMERIC_VARS, save=True)
    
    # ========================================
    # 15. CUSTOMER AGGREGATION
    # ========================================
    print("\n" + "="*80)
    print("15. SELECCIÓN DE VARIABLES PARA CLUSTERIZACIÓN")
    print("="*80)
    
    # ========================================
    # 16. STANDARDIZATION
    # ========================================
    print("\n" + "="*80)
    print("16. ESTANDARIZACIÓN - CENTRAR Y REDUCIR")
    print("="*80)
    
    # ========================================
    # 17. PCA
    # ========================================
    print("\n" + "="*80)
    print("17. ACP")
    print("="*80)
    
    # ========================================
    # 18. ELBOW METHOD
    # ========================================
    print("\n" + "="*80)
    print("18. MÉTODO DEL CODO JAMBÚ")
    print("="*80)
    
    # ========================================
    # 19. K-MEANS
    # ========================================
    print("\n" + "="*80)
    print("19. K-MEANS")
    print("="*80)
    
    # ========================================
    # 20. CUSTOMER SEGMENTATION
    # ========================================
    print("\n" + "="*80)
    print("20. SEGMENTACIÓN DE CLIENTES")
    print("="*80)
    
    # Create full dataset for segmentation
    df_full = df.merge(
        DF[['ProductID', 'ProductName']].drop_duplicates(),
        on='ProductID',
        how='left'
    )
    
    # Perform segmentation
    segmenter = CustomerSegmenter(df, DF, CLUSTERING_VARS)
    segmenter.prepare_customer_data()
    segmenter.perform_clustering(n_clusters=KMEANS_CONFIG["n_clusters"])
    segmenter.assign_segment_labels()
    
    # Get PCA data for plotting
    clientes_agg = segmenter.get_segmented_customers()
    pca_model = PCAModel(n_components=2)
    X_pca = pca_model.fit_transform(clientes_agg[CLUSTERING_VARS])
    
    # Plot segmentation
    segmenter.plot_pca_with_clusters(X_pca)
    segmenter.plot_cluster_distribution()
    segmenter.plot_radar_chart()
    segmenter.plot_cluster_details()
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO")
    print("="*80)
    print("El análisis se ha completado exitosamente.")


def run_data_loading_only():
    """Run only data loading (for testing)."""
    configure_pandas_display()
    loader = DataLoader("amazon.csv")
    DF = loader.load()
    loader.get_info()
    return DF


def run_visualization_only(df: pd.DataFrame):
    """Run only visualizations (requires preprocessed df)."""
    configure_matplotlib_style()
    
    dist_plotter = DistributionPlotter(df)
    dist_plotter.plot_numeric_distributions()
    dist_plotter.plot_categorical_distributions()
    
    chart_plotter = ChartPlotter(df)
    chart_plotter.plot_time_series()


def run_segmentation_only(df: pd.DataFrame, original_df: pd.DataFrame):
    """Run only customer segmentation."""
    configure_matplotlib_style()
    
    segmenter = CustomerSegmenter(df, original_df, CLUSTERING_VARS)
    segmenter.prepare_customer_data()
    segmenter.perform_clustering(n_clusters=4)
    segmenter.assign_segment_labels()
    
    clientes_agg = segmenter.get_segmented_customers()
    pca_model = PCAModel(n_components=2)
    X_pca = pca_model.fit_transform(clientes_agg[CLUSTERING_VARS])
    
    segmenter.plot_pca_with_clusters(X_pca)
    segmenter.plot_cluster_distribution()
    segmenter.plot_radar_chart()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_full_analysis()
