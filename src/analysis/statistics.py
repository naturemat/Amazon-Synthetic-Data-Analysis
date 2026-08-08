"""
Statistical analysis module for Amazon Sales Data Analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional


class StatisticalAnalyzer:
    """Handles statistical analysis operations."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the StatisticalAnalyzer with a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to analyze
        """
        self._df = df
    
    def calculate_descriptive_stats(self, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Calculate descriptive statistics.
        
        Parameters
        ----------
        columns : list, optional
            Columns to include
        
        Returns
        -------
        pd.DataFrame
            Descriptive statistics
        """
        if columns:
            return self._df[columns].describe()
        return self._df.describe()
    
    def calculate_percentiles(self, columns: list, percentiles: list = None) -> pd.DataFrame:
        """
        Calculate percentiles for specified columns.
        
        Parameters
        ----------
        columns : list
            Columns to analyze
        percentiles : list, optional
            Percentiles to calculate
        
        Returns
        -------
        pd.DataFrame
            Percentile values
        """
        percentiles = percentiles or [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        return self._df[columns].describe(percentiles=percentiles)
    
    def aggregate_by_customer(self) -> pd.DataFrame:
        """
        Aggregate data by customer for clustering.
        
        Returns
        -------
        pd.DataFrame
            Customer-level aggregated data
        """
        clientes_agg = self._df.groupby('CustomerID').agg({
            'TotalAmount': ['sum', 'mean', 'count'],
            'Quantity': 'sum',
            'Discount': 'mean',
            'UnitPrice': 'mean',
            'OrderDate': 'max'
        }).round(2)
        
        clientes_agg.columns = [
            'Ingreso_Total', 'Ticket_Promedio', 'Frecuencia',
            'Cantidad_Total', 'Descuento_Promedio',
            'Precio_Promedio', 'Ultima_Compra'
        ]
        
        clientes_agg['Dias_Ultima_Compra'] = (
            pd.Timestamp.now() - clientes_agg['Ultima_Compra']
        ).dt.days
        
        return clientes_agg
    
    def calculate_inertia_analysis(self, X_pca: np.ndarray, kmeans_model) -> dict:
        """
        Calculate inertia analysis for clustering.
        
        Parameters
        ----------
        X_pca : np.ndarray
            PCA-transformed data
        kmeans_model : KMeans
            Fitted K-Means model
        
        Returns
        -------
        dict
            Inertia analysis results
        """
        inertia_intra = kmeans_model.inertia_
        global_center = X_pca.mean(axis=0)
        inertia_total = np.sum((X_pca - global_center) ** 2)
        inertia_inter = inertia_total - inertia_intra
        
        return {
            'inertia_total': inertia_total,
            'inertia_intra': inertia_intra,
            'inertia_inter': inertia_inter,
            'pct_intra': 100 * inertia_intra / inertia_total,
            'pct_inter': 100 * inertia_inter / inertia_total,
        }
    
    def calculate_cluster_profile(self, clientes_agg: pd.DataFrame, 
                                   clustering_vars: list) -> pd.DataFrame:
        """
        Calculate cluster profiles.
        
        Parameters
        ----------
        clientes_agg : pd.DataFrame
            Aggregated customer data with cluster labels
        clustering_vars : list
            Variables used for clustering
        
        Returns
        -------
        pd.DataFrame
            Cluster profile (mean values)
        """
        return clientes_agg.groupby('Segmento')[clustering_vars].mean()
    
    def calculate_relative_profile(self, perfil_segmentos: pd.DataFrame, 
                                   clientes_agg: pd.DataFrame, 
                                   clustering_vars: list) -> pd.DataFrame:
        """
        Calculate relative profile (vs global average).
        
        Parameters
        ----------
        perfil_segmentos : pd.DataFrame
            Cluster profiles (means)
        clientes_agg : pd.DataFrame
            Full customer data
        clustering_vars : list
            Clustering variables
        
        Returns
        -------
        pd.DataFrame
            Relative profile
        """
        return perfil_segmentos / clientes_agg[clustering_vars].mean()
    
    def assign_segment_labels(self, perfil_relativo: pd.DataFrame) -> dict:
        """
        Assign labels to segments based on their profile.
        
        Parameters
        ----------
        perfil_relativo : pd.DataFrame
            Relative profile (vs average)
        
        Returns
        -------
        dict
            Mapping of segment to label
        """
        segmento_labels = {}
        
        for seg, row in perfil_relativo.iterrows():
            # 1. Premium: alto ingreso + alta frecuencia + reciente
            if (row['Ingreso_Total'] > 1.2 and 
                row['Frecuencia'] > 1.2 and 
                row['Dias_Ultima_Compra'] < 0.8):
                segmento_labels[seg] = 'PREMIUM (Alto Valor)'
            
            # 2. Frecuentes: alta frecuencia
            elif row['Frecuencia'] > 1.1:
                segmento_labels[seg] = 'FRECUENTES (Leales)'
            
            # 3. Ocasionales / Sensibles al precio
            elif row['Descuento_Promedio'] > 1.05:
                segmento_labels[seg] = 'OCASIONALES (Sensibles Precio)'
            
            # 4. Inactivos
            else:
                segmento_labels[seg] = 'INACTIVOS (Riesgo Pérdida)'
        
        return segmento_labels
    
    def calculate_cluster_statistics(self, df_full: pd.DataFrame, 
                                      clientes_agg: pd.DataFrame, 
                                      k_opt: int) -> pd.DataFrame:
        """
        Calculate detailed statistics for each cluster.
        
        Parameters
        ----------
        df_full : pd.DataFrame
            Full dataset with ProductName
        clientes_agg : pd.DataFrame
            Aggregated customer data with segment labels
        k_opt : int
            Number of clusters
        
        Returns
        -------
        pd.DataFrame
            Cluster statistics
        """
        cluster_stats = []
        
        for c in range(k_opt):
            cluster_customers = clientes_agg[clientes_agg['Segmento'] == c].index
            df_cluster = df_full[df_full['CustomerID'].isin(cluster_customers)]
            
            n_records = len(df_cluster)
            pct_total = n_records / len(df_full) * 100
            total_ingresos = df_cluster['TotalAmount'].sum()
            
            min_price = df_cluster['UnitPrice'].min()
            max_price = df_cluster['UnitPrice'].max()
            
            # Top/Bottom products by income
            prod_income = df_cluster.groupby('ProductName')['TotalAmount'].sum()
            top3_prod = prod_income.sort_values(ascending=False).head(3).to_dict()
            bottom3_prod = prod_income.sort_values(ascending=True).head(3).to_dict()
            
            # Top/Bottom categories by income
            cat_income = df_cluster.groupby('Category')['TotalAmount'].sum()
            top3_cat = cat_income.sort_values(ascending=False).head(3).to_dict()
            bottom3_cat = cat_income.sort_values(ascending=True).head(3).to_dict()
            
            # Top/Bottom products by quantity
            prod_qty = df_cluster.groupby('ProductName')['Quantity'].sum()
            top3_qty_prod = prod_qty.sort_values(ascending=False).head(3).to_dict()
            bottom3_qty_prod = prod_qty.sort_values(ascending=True).head(3).to_dict()
            
            # Top/Bottom categories by quantity
            cat_qty = df_cluster.groupby('Category')['Quantity'].sum()
            top3_qty_cat = cat_qty.sort_values(ascending=False).head(3).to_dict()
            bottom3_qty_cat = cat_qty.sort_values(ascending=True).head(3).to_dict()
            
            # User statistics
            user_stats = df_cluster.groupby('CustomerID').agg({
                'Quantity': 'sum', 
                'TotalAmount': 'sum'
            })
            
            top_user_qty = user_stats['Quantity'].idxmax()
            top_user_qty_value = user_stats.loc[top_user_qty, 'Quantity']
            top_user_qty_income = user_stats.loc[top_user_qty, 'TotalAmount']
            
            bottom_user_qty = user_stats['Quantity'].idxmin()
            bottom_user_qty_value = user_stats.loc[bottom_user_qty, 'Quantity']
            bottom_user_qty_income = user_stats.loc[bottom_user_qty, 'TotalAmount']
            
            top_user_income = user_stats['TotalAmount'].idxmax()
            top_user_income_value = user_stats.loc[top_user_income, 'TotalAmount']
            top_user_income_qty = user_stats.loc[top_user_income, 'Quantity']
            
            bottom_user_income = user_stats['TotalAmount'].idxmin()
            bottom_user_income_value = user_stats.loc[bottom_user_income, 'TotalAmount']
            bottom_user_income_qty = user_stats.loc[bottom_user_income, 'Quantity']
            
            n_customers = df_cluster['CustomerID'].nunique()
            n_products = df_cluster['ProductID'].nunique()
            n_sellers = df_cluster['SellerID'].nunique()
            
            cluster_stats.append({
                'Cluster': c,
                'Registros': n_records,
                'Porcentaje_total': pct_total,
                'Clientes_unicos': n_customers,
                'Productos_unicos': n_products,
                'Vendedores_unicos': n_sellers,
                'Ingresos_totales': total_ingresos,
                'Precio_min': min_price,
                'Precio_max': max_price,
                'Top3_Productos_Ingreso': top3_prod,
                'Bottom3_Productos_Ingreso': bottom3_prod,
                'Top3_Categorias_Ingreso': top3_cat,
                'Bottom3_Categorias_Ingreso': bottom3_cat,
                'Top3_Productos_Cantidad': top3_qty_prod,
                'Bottom3_Productos_Cantidad': bottom3_qty_prod,
                'Top3_Categorias_Cantidad': top3_qty_cat,
                'Bottom3_Categorias_Cantidad': bottom3_qty_cat,
                'Usuario_Mas_Compras': {
                    'CustomerID': top_user_qty, 
                    'Cantidad': top_user_qty_value, 
                    'Ingresos': top_user_qty_income
                },
                'Usuario_Menos_Compras': {
                    'CustomerID': bottom_user_qty, 
                    'Cantidad': bottom_user_qty_value, 
                    'Ingresos': bottom_user_qty_income
                },
                'Usuario_Mas_Ingreso': {
                    'CustomerID': top_user_income, 
                    'Ingresos': top_user_income_value, 
                    'Cantidad': top_user_income_qty
                },
                'Usuario_Menos_Ingreso': {
                    'CustomerID': bottom_user_income, 
                    'Ingresos': bottom_user_income_value, 
                    'Cantidad': bottom_user_income_qty
                },
            })
        
        return pd.DataFrame(cluster_stats)


def analyze(df: pd.DataFrame) -> StatisticalAnalyzer:
    """
    Convenience function to create a StatisticalAnalyzer.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze
    
    Returns
    -------
    StatisticalAnalyzer
        The analyzer instance
    """
    return StatisticalAnalyzer(df)
