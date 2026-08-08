"""
Visualization module - Complex charts for Amazon Sales Data Analysis.
Handles time series, correlation matrices, scatter plots, pie charts, and radar charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import itertools
from typing import Optional, List, Tuple
from matplotlib.ticker import FuncFormatter

from ..core.formatters import get_formatter, human_format, human_format_detailed
from ..config.settings import NUMERIC_VARS


class ChartPlotter:
    """Handles complex chart visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ChartPlotter.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to plot
        """
        self._df = df
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the dataframe."""
        return self._df
    
    def plot_time_series(self, date_column: str = "OrderDate", 
                         value_column: str = "TotalAmount") -> None:
        """
        Plot monthly revenue time series.
        
        Parameters
        ----------
        date_column : str
            Date column name
        value_column : str
            Value column to aggregate
        """
        
        # Monthly series
        serie_mensual = self._df.set_index(date_column)[value_column].resample('ME').sum()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Colors by year
        years = serie_mensual.index.year
        unique_years = sorted(years.unique())
        colors_map = {year: color for year, color in zip(unique_years, plt.cm.tab10.colors)}
        
        # Line connecting all points
        ax.plot(serie_mensual.index, serie_mensual.values, color='gray', linewidth=1.5, alpha=0.6)
        
        # Each point with color by year
        for date, value in serie_mensual.items():
            ax.plot(date, value, marker='o', color=colors_map[date.year], markersize=6)
        
        # Labels: first letter of month
        for date, value in serie_mensual.items():
            month_initial = date.strftime('%b')[0]
            ax.text(date, value + serie_mensual.max()*0.01, month_initial,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # X axis configuration
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Año')
        ax.set_ylabel('TotalAmount')
        ax.set_title('TotalAmount mensual por año con iniciales de mes')
        ax.yaxis.set_major_formatter(get_formatter())
        
        # Grid only horizontal
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, columns: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot correlation matrix heatmap.
        
        Parameters
        ----------
        columns : list, optional
            Columns to include
        figsize : tuple
            Figure size
        """
        print("\n" + "="*80)
        print("MATRIZ DE CORRELACIONES")
        print("="*80)
        
        df_numeric = self._df.select_dtypes(include=['number'])
        
        if columns:
            df_numeric = df_numeric[columns]
        
        corr_matrix = df_numeric.corr()
        
        # Bicolor heatmap
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.cm.RdBu
        
        im = ax.imshow(
            corr_matrix,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation='nearest'
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlación')
        
        # Ticks
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Values inside cells
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(
                    j, i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=9
                )
        
        ax.set_title("Matriz de Correlaciones (Bicolor)")
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_matrix(self, columns: Optional[List[str]] = None,
                            save: bool = False) -> None:
        """
        Plot scatter plot matrix.
        
        Parameters
        ----------
        columns : list, optional
            Columns to plot
        save : bool
            Whether to save the figure
        """
        
        columns = columns or NUMERIC_VARS
        
        if not all(c in self._df.columns for c in columns):
            print('Faltan algunas de las columnas requeridas para el gráfico de pares.')
            return
        
        data = self._df[columns].dropna()
        
        # All combinations without repetition
        pairs = list(itertools.combinations(columns, 2))
        
        # Create figure 3x5
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for ax, (y, x) in zip(axes, pairs):
            sns.scatterplot(
                data=data,
                x=x,
                y=y,
                ax=ax
            )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        
        # Remove empty axes
        for i in range(len(pairs), len(axes)):
            fig.delaxes(axes[i])
        
        fig.suptitle(
            'Diagramas de Dispersión Bivariados (3 filas × 5 columnas)',
            fontsize=16
        )
        
        fig.tight_layout()
        
        if save:
            fig.savefig("Diagrama_dispersion_3x5_bivariado.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_id_distribution(self, id_vars: List[str], top_n: int = 30) -> None:
        """
        Plot distribution for ID variables, limited to the most frequent IDs.

        Parameters
        ----------
        id_vars : list
            ID columns to plot
        top_n : int
            Number of most frequent IDs to display
        """

        for col in id_vars:
            if col not in self._df.columns:
                continue

            freq = self._df[col].value_counts()
            top_ids = freq.head(top_n).index

            df_subset = self._df[self._df[col].isin(top_ids)]

            income = (
                df_subset.groupby(col)['TotalAmount']
                .sum()
                .reindex(top_ids)
            )

            fig, ax1 = plt.subplots(figsize=(10, 4))

            # Bars: distribution of the top-N IDs
            sns.countplot(
                data=df_subset,
                x=col,
                order=top_ids,
                ax=ax1,
                color='#9ecae1'
            )
            ax1.set_ylabel('Frecuencia')
            ax1.set_xlabel('')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_title(f'Distribución e Ingresos por {col} (Top {top_n})')
            ax1.grid(axis='y', linestyle='--', alpha=0.4)

            # Red dots: income per top-N ID
            ax2 = ax1.twinx()
            ax2.scatter(
                range(len(top_ids)),
                income.values,
                color='red',
                alpha=0.6,
                s=25,
                zorder=5
            )

            ax2.yaxis.set_major_formatter(get_formatter())
            ax2.set_ylabel('Ingresos')

            plt.tight_layout()
            plt.show()
    
    def plot_top_bottom_bar(self, combined: pd.Series, 
                            labels: pd.Series,
                            title: str,
                            top_color: str = 'red',
                            bottom_color: str = 'blue') -> None:
        """
        Plot top/bottom bar chart.
        
        Parameters
        ----------
        combined : pd.Series
            Combined top and bottom values
        labels : pd.Series
            Labels for the bars
        title : str
            Chart title
        top_color : str
            Color for top values
        bottom_color : str
            Color for bottom values
        """
        plt.figure(figsize=(14, 5))
        ax = sns.barplot(
            x=labels,
            y=combined.values,
            palette=[top_color] * 10 + [bottom_color] * 10
        )
        
        plt.title(title)
        plt.xlabel('')
        ax.yaxis.set_major_formatter(get_formatter())
        plt.ylabel('Ingresos')
        plt.xticks(rotation=90)
        
        # Labels on bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height:,.1f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=9,
                xytext=(0, 3),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_bottom_by_category(self, df_plot: pd.DataFrame) -> None:
        """
        Plot top/bottom products by category.
        
        Parameters
        ----------
        df_plot : pd.DataFrame
            Data with Category, Product, TotalAmount, Type columns
        """
        
        df_plot['Label'] = df_plot['Category'] + ' - ' + df_plot['Product']
        
        palette = {'Top 1': 'red', 'Bottom 1': 'blue'}
        
        plt.figure(figsize=(16, 6))
        ax = sns.barplot(
            x='Label',
            y='TotalAmount',
            hue='Type',
            data=df_plot,
            palette=palette
        )
        
        ax.yaxis.set_major_formatter(get_formatter())
        plt.xticks(rotation=90, ha='right')
        plt.ylabel('Ingresos')
        plt.xlabel('Producto - Categoría')
        plt.title('Top 1 y Bottom 1 productos por categoría')
        
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                human_format(height, None),
                (p.get_x() + p.get_width()/2, height),
                ha='center',
                va='bottom',
                fontsize=9,
                xytext=(0, 3),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_bottom_by_state(self, df_plot: pd.DataFrame) -> None:
        """
        Plot top/bottom products by state.
        
        Parameters
        ----------
        df_plot : pd.DataFrame
            Data with State, Product, TotalAmount, Type columns
        """
        
        df_plot['Label'] = df_plot['Product'] + '\n(' + df_plot['State'] + ')'
        
        palette = {'Top 1': 'red', 'Bottom 1': 'blue'}
        
        plt.figure(figsize=(18, 6))
        ax = sns.barplot(
            x='Label',
            y='TotalAmount',
            hue='Type',
            data=df_plot,
            palette=palette
        )
        
        plt.title('Top 1 y Bottom 1 productos por estado')
        plt.xlabel('Producto (Estado)')
        plt.ylabel('Ingresos')
        ax.yaxis.set_major_formatter(get_formatter())
        
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                human_format(height, None),
                (p.get_x() + p.get_width()/2, height),
                ha='center',
                va='bottom',
                fontsize=9,
                xytext=(0, 3),
                textcoords='offset points'
            )
        
        plt.xticks(rotation=90, ha='center')
        plt.tight_layout()
        plt.show()


def plot_time_series(df: pd.DataFrame, date_column: str = "OrderDate", 
                     value_column: str = "TotalAmount") -> None:
    """Convenience function to plot time series."""
    plotter = ChartPlotter(df)
    plotter.plot_time_series(date_column, value_column)


def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Convenience function to plot correlation matrix."""
    plotter = ChartPlotter(df)
    plotter.plot_correlation_matrix(columns)


def plot_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                        save: bool = False) -> None:
    """Convenience function to plot scatter matrix."""
    plotter = ChartPlotter(df)
    plotter.plot_scatter_matrix(columns, save)
