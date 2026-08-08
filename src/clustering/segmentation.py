"""
Customer segmentation module for Amazon Sales Data Analysis.
Combines PCA, K-Means, and visualization for customer segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple

from .pca import PCAModel
from .kmeans import KMeansModel
from ..config.settings import CLUSTERING_VARS
from ..core.formatters import get_formatter, human_format_detailed


class CustomerSegmenter:
    """Handles customer segmentation using PCA and K-Means."""
    
    def __init__(self, df: pd.DataFrame, original_df: pd.DataFrame,
                 clustering_vars: Optional[List[str]] = None):
        """
        Initialize the CustomerSegmenter.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataframe
        original_df : pd.DataFrame
            Original dataframe (for joins)
        clustering_vars : list, optional
            Variables to use for clustering
        """
        self._df = df
        self._original_df = original_df
        self._clustering_vars = clustering_vars or CLUSTERING_VARS
        
        self._clientes_agg: Optional[pd.DataFrame] = None
        self._pca_model: Optional[PCAModel] = None
        self._kmeans_model: Optional[KMeansModel] = None
        self._segment_labels: Dict[int, str] = {}
    
    def prepare_customer_data(self) -> pd.DataFrame:
        """
        Aggregate data by customer for clustering.
        
        Returns
        -------
        pd.DataFrame
            Aggregated customer data
        """
        print("\n" + "="*80)
        print("SELECCIÓN DE VARIABLES PARA CLUSTERIZACIÓN")
        print("="*80)
        
        self._clientes_agg = self._df.groupby('CustomerID').agg({
            'TotalAmount': ['sum', 'mean', 'count'],
            'Quantity': 'sum',
            'Discount': 'mean',
            'UnitPrice': 'mean',
            'OrderDate': 'max'
        }).round(2)
        
        self._clientes_agg.columns = [
            'Ingreso_Total', 'Ticket_Promedio', 'Frecuencia',
            'Cantidad_Total', 'Descuento_Promedio',
            'Precio_Promedio', 'Ultima_Compra'
        ]
        
        self._clientes_agg['Dias_Ultima_Compra'] = (
            pd.Timestamp.now() - self._clientes_agg['Ultima_Compra']
        ).dt.days
        
        print(f"Clientes analizados: {len(self._clientes_agg)}")
        
        return self._clientes_agg
    
    def perform_clustering(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        Perform PCA and K-Means clustering.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        
        Returns
        -------
        pd.DataFrame
            Customer data with segment labels
        """
        if self._clientes_agg is None:
            self.prepare_customer_data()
        
        # Standardization
        print("\n" + "="*80)
        print("ESTANDARIZACIÓN - CENTRAR Y REDUCIR")
        print("="*80)
        
        self._pca_model = PCAModel(n_components=2)
        X_pca = self._pca_model.fit_transform(self._clientes_agg[self._clustering_vars])
        
        print(self._pca_model.scaler.mean_)
        print("\n")
        print(pd.DataFrame(
            self._pca_model.scaler.transform(self._clientes_agg[self._clustering_vars].fillna(0)),
            columns=self._clustering_vars,
            index=self._clientes_agg.index
        ).describe().loc[['mean', 'std']])
        
        # PCA summary
        self._pca_model.print_summary()
        
        loadings = self._pca_model.get_loadings(self._clustering_vars)
        print(loadings)
        
        # Plot PCA without clustering
        print("\nGRÁFICA ACP DE CLIENTES (Sin Clustering)")
        self._pca_model.plot_pca_scatter()
        
        # K-Means
        print("\n" + "="*80)
        print("K-MEANS")
        print("="*80)
        
        self._kmeans_model = KMeansModel(n_clusters=n_clusters)
        labels = self._kmeans_model.fit_predict(X_pca)
        
        self._clientes_agg['Segmento'] = labels
        
        # Inertia analysis
        self._kmeans_model.print_inertia_analysis(X_pca)
        
        return self._clientes_agg
    
    def assign_segment_labels(self) -> Dict[int, str]:
        """
        Assign labels to segments based on their profile.
        
        Returns
        -------
        dict
            Mapping of segment to label
        """
        if self._clientes_agg is None:
            raise ValueError("Run perform_clustering first")
        
        perfil_segmentos = self._clientes_agg.groupby('Segmento')[self._clustering_vars].mean()
        perfil_relativo = perfil_segmentos / self._clientes_agg[self._clustering_vars].mean()
        
        self._segment_labels = {}
        
        for seg, row in perfil_relativo.iterrows():
            # Premium: alto ingreso + alta frecuencia + reciente
            if (row['Ingreso_Total'] > 1.2 and 
                row['Frecuencia'] > 1.2 and 
                row['Dias_Ultima_Compra'] < 0.8):
                self._segment_labels[seg] = 'PREMIUM (Alto Valor)'
            
            # Frecuentes: alta frecuencia
            elif row['Frecuencia'] > 1.1:
                self._segment_labels[seg] = 'FRECUENTES (Leales)'
            
            # Ocasionales / Sensibles al precio
            elif row['Descuento_Promedio'] > 1.05:
                self._segment_labels[seg] = 'OCASIONALES (Sensibles Precio)'
            
            # Inactivos
            else:
                self._segment_labels[seg] = 'INACTIVOS (Riesgo Pérdida)'
        
        print("\nEtiquetas asignadas automáticamente (flexible):")
        for k, v in self._segment_labels.items():
            print(f"Cluster {k}: {v}")
        
        return self._segment_labels
    
    def plot_pca_with_clusters(self, X_pca: np.ndarray) -> None:
        """
        Plot PCA with cluster visualization and correlation circle.
        
        Parameters
        ----------
        X_pca : np.ndarray
            PCA-transformed data
        """
        print("PCA + CLUSTERS + CÍRCULO DE CORRELACIONES")
        
        loadings = self._pca_model.get_loadings(self._clustering_vars)
        
        centroids = self._kmeans_model.centroids
        explained_var = self._pca_model.explained_variance_ratio
        
        fig, ax = plt.subplots(figsize=(9, 8))
        
        # PCA + Clusters
        for c in range(self._kmeans_model.n_clusters):
            ax.scatter(
                X_pca[self._clientes_agg['Segmento'] == c, 0],
                X_pca[self._clientes_agg['Segmento'] == c, 1],
                label=self._segment_labels.get(c, f'Cluster {c}'),
                alpha=0.6
            )
        
        # Centroids
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='X',
            s=200,
            color='black',
            label='Centroides'
        )
        
        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
        ax.set_title("Segmentación de Clientes (PCA + Círculo de Correlaciones)")
        ax.grid(alpha=0.3)
        
        # Save PCA limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Correlation circle
        circle_scale = min(
            (xlim[1] - xlim[0]),
            (ylim[1] - ylim[0])
        ) * 0.9
        
        theta = np.linspace(0, 2*np.pi, 300)
        
        ax.plot(
            circle_scale * np.cos(theta),
            circle_scale * np.sin(theta),
            linestyle='--',
            color='gray',
            alpha=0.6
        )
        
        # Loading vectors
        for var in loadings.index:
            x = loadings.loc[var, 'PC1'] * circle_scale
            y = loadings.loc[var, 'PC2'] * circle_scale
            
            ax.arrow(
                0, 0, x, y,
                color='black',
                width=0.0,
                head_width=circle_scale * 0.06,
                length_includes_head=True,
                alpha=0.85
            )
            
            ax.text(
                x * 1.1,
                y * 1.1,
                var,
                fontsize=9,
                ha='center',
                va='center'
            )
        
        # Base axes
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        
        # Force PCA range
        ax.set_xlim(-10, 30)
        ax.set_ylim(-10, 10)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_distribution(self) -> None:
        """Plot pie charts for cluster distribution."""
        # Record distribution by cluster
        df_full = self._df.merge(
            self._original_df[['ProductID', 'ProductName']].drop_duplicates(),
            on='ProductID',
            how='left'
        )
        
        record_counts = []
        
        for c in range(self._kmeans_model.n_clusters):
            cluster_customers = self._clientes_agg[self._clientes_agg['Segmento'] == c].index
            n_records = df_full[df_full['CustomerID'].isin(cluster_customers)].shape[0]
            record_counts.append(n_records)
        
        record_counts = np.array(record_counts)
        record_labels = [f"Cluster {i}\n{self._segment_labels[i]}" for i in range(self._kmeans_model.n_clusters)]
        
        # Pie chart - records
        plt.figure(figsize=(8, 8))
        plt.pie(
            record_counts,
            labels=record_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab10.colors[:self._kmeans_model.n_clusters],
            wedgeprops={'edgecolor': 'black'}
        )
        plt.title("Distribución de registros por segmento (K-Means)")
        plt.show()
        
        # Pie chart - segments
        segment_counts = self._clientes_agg['Segmento'].value_counts().sort_index()
        segment_labels = [f"Cluster {i}\n{self._segment_labels[i]}" for i in segment_counts.index]
        
        plt.figure(figsize=(8, 8))
        plt.pie(
            segment_counts,
            labels=segment_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab10.colors[:len(segment_counts)],
            wedgeprops={'edgecolor': 'black'}
        )
        plt.title("Distribución de clientes por segmentos (K-Means)")
        plt.show()
    
    def plot_radar_chart(self) -> None:
        """Plot radar chart for cluster profiles."""
        # Normalized cluster means
        cluster_means = self._clientes_agg.groupby('Segmento')[self._clustering_vars].mean()
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        # Angles
        num_vars = len(self._clustering_vars)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        ax.set_rscale('linear')
        ax.set_ylim(0, 1.1)
        
        plt.xticks(angles[:-1], self._clustering_vars, fontsize=10)
        
        colors = plt.cm.tab10.colors
        for i, row in cluster_means_norm.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, color=colors[i], linewidth=2, 
                   label=f"Cluster {i} ({self._segment_labels[i]})")
            ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        ax.yaxis.grid(True, color='gray', linestyle='--', alpha=0.5)
        ax.xaxis.grid(True, color='gray', linestyle='--', alpha=0.5)
        
        plt.title("Perfil de clusters (Radar / Estrella)", size=14, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_details(self) -> None:
        """Plot detailed charts for each cluster."""
        df_full = self._df.merge(
            self._original_df[['ProductID', 'ProductName']].drop_duplicates(),
            on='ProductID',
            how='left'
        )

        for c in range(self._kmeans_model.n_clusters):
            df_cluster = df_full[self._df['CustomerID'].isin(
                self._clientes_agg[self._clientes_agg['Segmento'] == c].index
            )]
            etiqueta = self._segment_labels[c]
            
            # Products by income
            prod_income = df_cluster.groupby('ProductName')['TotalAmount'].sum()
            top3 = prod_income.sort_values(ascending=False).head(3)
            bottom3 = prod_income.sort_values(ascending=True).head(3)
            combined = pd.concat([top3, bottom3])
            
            plt.figure(figsize=(12, 5))
            ax = plt.gca()
            sns.barplot(x=combined.index, y=combined.values, palette=['red']*3 + ['blue']*3, ax=ax)
            plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Productos por Ingreso')
            plt.ylabel('Total Amount')
            plt.xlabel('Producto')
            
            for p in ax.patches:
                ax.annotate(human_format_detailed(p.get_height()),
                           (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    def get_segmented_customers(self) -> pd.DataFrame:
        """Get customer data with segment labels."""
        if self._clientes_agg is None:
            raise ValueError("Run perform_clustering first")
        
        return self._clientes_agg.copy()


def segment_customers(df: pd.DataFrame, original_df: pd.DataFrame,
                      n_clusters: int = 4) -> Tuple[CustomerSegmenter, pd.DataFrame]:
    """
    Convenience function to perform customer segmentation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    original_df : pd.DataFrame
        Original dataframe
    n_clusters : int
        Number of clusters
    
    Returns
    -------
    tuple
        (CustomerSegmenter, segmented_data)
    """
    segmenter = CustomerSegmenter(df, original_df)
    segmenter.prepare_customer_data()
    segmenter.perform_clustering(n_clusters)
    segmenter.assign_segment_labels()
    
    return segmenter, segmenter.get_segmented_customers()
