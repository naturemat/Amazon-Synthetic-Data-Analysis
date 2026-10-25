"""
Visualization module - Basic plots for Amazon Sales Data Analysis.
Handles distribution plots, histograms, and boxplots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

from ..core.formatters import get_formatter, human_format
from ..core.plotting import save_figure
from ..config.settings import NUMERIC_VARS, CATEGORICAL_VARS


class DistributionPlotter:
    """Handles distribution plots (histograms, boxplots)."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DistributionPlotter.
        
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
    
    def plot_numeric_distributions(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot distribution for all numeric variables.
        
        Parameters
        ----------
        columns : list, optional
            Columns to plot. Defaults to NUMERIC_VARS.
        """
        columns = columns or NUMERIC_VARS
        
        print("\n" + "="*80)
        print("GRÁFICAS DE DISTRIBUCIÓN - VARIABLES NUMÉRICAS")
        print("="*80)
        
        for col in columns:
            if col not in self._df.columns:
                continue
                
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            sns.histplot(self._df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Distribución de {col}')
            
            # Boxplot with outliers in red
            sns.boxplot(
                y=self._df[col],
                ax=axes[1],
                flierprops=dict(
                    marker='o',
                    markerfacecolor='red',
                    markeredgecolor='red',
                    markersize=5
                )
            )
            axes[1].set_title(f'Boxplot de {col}')
            
            plt.tight_layout()
            save_figure(fig, f"kdd_distribucion_num_{col}")
    
    def plot_categorical_distributions(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot distribution for categorical variables with income overlay.
        
        Parameters
        ----------
        columns : list, optional
            Columns to plot. Defaults to CATEGORICAL_VARS.
        """
        columns = columns or CATEGORICAL_VARS
        
        print("GRÁFICAS DE DISTRIBUCIÓN - VARIABLES CATEGÓRICAS")
        
        for col in columns:
            if col not in self._df.columns:
                continue
            
            freq = self._df[col].value_counts(sort=False)
            income = self._df.groupby(col, observed=True)['TotalAmount'].sum().reindex(freq.index)
            
            fig, ax1 = plt.subplots(figsize=(12, 5))
            
            # Distribution (frequency)
            sns.barplot(
                x=freq.index,
                y=freq.values,
                ax=ax1,
                color='#6BAED6'
            )
            ax1.set_ylabel('Frecuencia')
            ax1.set_xlabel(col)
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Income (line)
            ax2 = ax1.twinx()
            ax2.plot(
                freq.index,
                income.values,
                color='#E31A1C',
                marker='o',
                linewidth=2
            )
            ax2.yaxis.set_major_formatter(get_formatter())
            ax2.set_ylabel('Ingresos')
            
            # Labels for Brand
            if col == 'Brand':
                income_sorted = income.sort_values(ascending=False)
                top5 = income_sorted.head(5).index
                bottom5 = income_sorted.tail(5).index
                show_labels = set(top5).union(bottom5)
            else:
                show_labels = freq.index
            
            # Income labels
            for x, y in zip(freq.index, income.values):
                if x in show_labels:
                    ax2.annotate(
                        human_format(y, None),
                        (x, y),
                        textcoords='offset points',
                        xytext=(0, 6),
                        ha='center',
                        fontsize=9,
                        color='black'
                    )
            
            ax1.set_title(f'Distribución e Ingresos por {col}')
            
            ax1.tick_params(
                axis='x',
                rotation=90 if col in ['Brand', 'Category', 'City', 'State'] else 45
            )
            
            plt.tight_layout()
            save_figure(fig, f"kdd_distribucion_cat_{col}")


def plot_numeric_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Convenience function to plot numeric distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    columns : list, optional
        Columns to plot
    """
    plotter = DistributionPlotter(df)
    plotter.plot_numeric_distributions(columns)


def plot_categorical_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Convenience function to plot categorical distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    columns : list, optional
        Columns to plot
    """
    plotter = DistributionPlotter(df)
    plotter.plot_categorical_distributions(columns)
