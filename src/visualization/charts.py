"""
Visualization module - Complex charts for Amazon Sales Data Analysis.

Handles time series, correlation matrices, scatter plots, pie charts, and radar
charts. Figures are written to ``docs/images`` instead of blocking the pipeline.
"""

import itertools
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config.settings import NUMERIC_VARS, SCATTER_MATRIX_SAMPLE
from ..core.formatters import get_formatter, human_format, human_format_detailed
from ..core.plotting import save_figure


class ChartPlotter:
    """Handles complex chart visualizations."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ChartPlotter.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to plot.
        """
        self._df = df

    @property
    def data(self) -> pd.DataFrame:
        """Get the dataframe."""
        return self._df

    def plot_time_series(self, date_column: str = "OrderDate",
                         value_column: str = "TotalAmount") -> None:
        """
        Plot the monthly revenue time series.

        Parameters
        ----------
        date_column : str
            Date column name.
        value_column : str
            Value column to aggregate.
        """
        monthly = self._df.set_index(date_column)[value_column].resample("ME").sum()

        fig, ax = plt.subplots(figsize=(12, 5))

        years = monthly.index.year
        unique_years = sorted(years.unique())
        colors_map = {year: color for year, color in zip(unique_years, plt.cm.tab10.colors)}

        ax.plot(monthly.index, monthly.values, color="gray", linewidth=1.5, alpha=0.6)

        for date, value in monthly.items():
            ax.plot(date, value, marker="o", color=colors_map[date.year], markersize=6)

        for date, value in monthly.items():
            month_initial = date.strftime("%b")[0]
            ax.text(date, value + monthly.max() * 0.01, month_initial,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Año")
        ax.set_ylabel("Ingreso mensual (USD)")
        ax.set_title("Ingreso mensual por año (inicial del mes en cada punto)")
        ax.yaxis.set_major_formatter(get_formatter())
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        save_figure(fig, "kdd_serie_ingresos_mensual")

    def plot_correlation_matrix(self, columns: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot the correlation matrix heatmap.

        Parameters
        ----------
        columns : list, optional
            Columns to include.
        figsize : tuple
            Figure size.
        """
        print("\n" + "=" * 80)
        print("MATRIZ DE CORRELACIONES")
        print("=" * 80)

        df_numeric = self._df.select_dtypes(include=["number"])
        if columns:
            df_numeric = df_numeric[columns]

        corr_matrix = df_numeric.corr()

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_matrix, cmap=plt.cm.RdBu, vmin=-1, vmax=1,
                       interpolation="nearest")

        colorbar = plt.colorbar(im, ax=ax)
        colorbar.set_label("Correlación")

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_matrix.columns)

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=9)

        ax.set_title("Matriz de correlaciones")
        fig.tight_layout()
        save_figure(fig, "kdd_matriz_correlacion")

    def plot_scatter_matrix(self, columns: Optional[List[str]] = None,
                            save: bool = True) -> None:
        """
        Plot a bivariate scatter matrix (downsampled for speed).

        Parameters
        ----------
        columns : list, optional
            Columns to plot.
        save : bool
            Kept for backward compatibility; figures are always saved.
        """
        columns = columns or NUMERIC_VARS

        if not all(c in self._df.columns for c in columns):
            print("Faltan algunas de las columnas requeridas para el gráfico de pares.")
            return

        data = self._df[columns].dropna()
        if len(data) > SCATTER_MATRIX_SAMPLE:
            data = data.sample(SCATTER_MATRIX_SAMPLE, random_state=42)

        pairs = list(itertools.combinations(columns, 2))
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()

        for ax, (y, x) in zip(axes, pairs):
            sns.scatterplot(data=data, x=x, y=y, ax=ax, s=8, alpha=0.4)
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        for i in range(len(pairs), len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle("Diagramas de dispersión bivariados (3 × 5)", fontsize=16)
        fig.tight_layout()
        save_figure(fig, "kdd_dispersion_bivariada", dpi=200)

    def plot_id_distribution(self, id_vars: List[str], top_n: int = 30) -> None:
        """
        Plot the distribution of the most frequent IDs and their revenue.

        Parameters
        ----------
        id_vars : list
            ID columns to plot.
        top_n : int
            Number of most frequent IDs to display.
        """
        for col in id_vars:
            if col not in self._df.columns:
                continue

            freq = self._df[col].value_counts()
            top_ids = freq.head(top_n).index
            df_subset = self._df[self._df[col].isin(top_ids)]

            income = (
                df_subset.groupby(col)["TotalAmount"]
                .sum()
                .reindex(top_ids)
            )

            fig, ax1 = plt.subplots(figsize=(10, 4))

            sns.countplot(data=df_subset, x=col, order=top_ids,
                          ax=ax1, color="#9ecae1")
            ax1.set_ylabel("Frecuencia")
            ax1.set_xlabel("")
            ax1.tick_params(axis="x", rotation=45)
            ax1.set_title(f"Distribución e ingresos por {col} (Top {top_n})")
            ax1.grid(axis="y", linestyle="--", alpha=0.4)

            ax2 = ax1.twinx()
            ax2.scatter(range(len(top_ids)), income.values,
                        color="red", alpha=0.6, s=25, zorder=5)
            ax2.yaxis.set_major_formatter(get_formatter())
            ax2.set_ylabel("Ingresos")

            fig.tight_layout()
            save_figure(fig, f"kdd_distribucion_{col}")

    def plot_top_bottom_bar(self, combined: pd.Series,
                            labels: pd.Series,
                            title: str,
                            top_color: str = "red",
                            bottom_color: str = "blue") -> None:
        """Plot a top/bottom bar chart."""
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.barplot(x=labels, y=combined.values,
                    palette=[top_color] * 10 + [bottom_color] * 10, ax=ax)

        ax.set_title(title)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(get_formatter())
        ax.set_ylabel("Ingresos")
        plt.setp(ax.get_xticklabels(), rotation=90)

        for patch in ax.patches:
            height = patch.get_height()
            ax.annotate(f"{height:,.1f}",
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 3), textcoords="offset points")

        fig.tight_layout()
        save_figure(fig, "kdd_top_bottom")

    def plot_top_bottom_by_category(self, df_plot: pd.DataFrame) -> None:
        """Plot top/bottom products by category."""
        df_plot = df_plot.copy()
        df_plot["Label"] = df_plot["Category"] + " - " + df_plot["Product"]
        palette = {"Top 1": "red", "Bottom 1": "blue"}

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(x="Label", y="TotalAmount", hue="Type",
                    data=df_plot, palette=palette, ax=ax)

        ax.yaxis.set_major_formatter(get_formatter())
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
        ax.set_ylabel("Ingresos")
        ax.set_xlabel("Producto - Categoría")
        ax.set_title("Top 1 y Bottom 1 productos por categoría")

        for patch in ax.patches:
            height = patch.get_height()
            ax.annotate(human_format(height, None),
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 3), textcoords="offset points")

        fig.tight_layout()
        save_figure(fig, "kdd_top_bottom_categoria")

    def plot_top_bottom_by_state(self, df_plot: pd.DataFrame) -> None:
        """Plot top/bottom products by state."""
        df_plot = df_plot.copy()
        df_plot["Label"] = df_plot["Product"] + "\n(" + df_plot["State"] + ")"
        palette = {"Top 1": "red", "Bottom 1": "blue"}

        fig, ax = plt.subplots(figsize=(18, 6))
        sns.barplot(x="Label", y="TotalAmount", hue="Type",
                    data=df_plot, palette=palette, ax=ax)

        ax.set_title("Top 1 y Bottom 1 productos por estado")
        ax.set_xlabel("Producto (Estado)")
        ax.set_ylabel("Ingresos")
        ax.yaxis.set_major_formatter(get_formatter())

        for patch in ax.patches:
            height = patch.get_height()
            ax.annotate(human_format(height, None),
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 3), textcoords="offset points")

        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        fig.tight_layout()
        save_figure(fig, "kdd_top_bottom_estado")


def plot_time_series(df: pd.DataFrame, date_column: str = "OrderDate",
                     value_column: str = "TotalAmount") -> None:
    """Convenience function to plot a time series."""
    ChartPlotter(df).plot_time_series(date_column, value_column)


def plot_correlation_matrix(df: pd.DataFrame,
                            columns: Optional[List[str]] = None) -> None:
    """Convenience function to plot a correlation matrix."""
    ChartPlotter(df).plot_correlation_matrix(columns)


def plot_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        save: bool = True) -> None:
    """Convenience function to plot a scatter matrix."""
    ChartPlotter(df).plot_scatter_matrix(columns, save)
