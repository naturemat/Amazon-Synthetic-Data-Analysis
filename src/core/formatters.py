"""
Core utilities and formatters for the Amazon Sales Data Analysis project.
"""

import os

import numpy as np
import matplotlib

# Batch-friendly backend: figures are saved to disk. Set MPLBACKEND to override.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def human_format(x, pos=None):
    """
    Format numeric values to human-readable format (K, M, B).

    Parameters
    ----------
    x : float
        The value to format
    pos : int, optional
        Position (for matplotlib formatter)

    Returns
    -------
    str
        Formatted string (e.g., '1.5M', '500K')
    """
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.1f}k'
    else:
        return f'{x:.0f}'


def human_format_detailed(num, pos=None):
    """
    Format numbers with detailed units (K, M, B).

    Parameters
    ----------
    num : float
        The value to format
    pos : int, optional
        Position (for matplotlib formatter)

    Returns
    -------
    str
        Formatted string
    """
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.0f}B"


def configure_pandas_display():
    """Configure pandas display options for better output."""
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:.3f}'.format)


def configure_matplotlib_style():
    """Configure matplotlib style for all plots."""
    from ..config.settings import PLOT_STYLE

    try:
        plt.style.use(PLOT_STYLE)
    except OSError:  # older matplotlib versions
        plt.style.use("ggplot")


def get_formatter():
    """Return the human_format function as a matplotlib formatter."""
    return FuncFormatter(human_format)


def get_color_palette(n_colors=10):
    """Get a color palette for visualizations."""
    return plt.cm.tab10.colors[:n_colors]


def format_currency(value, locale='es_EC'):
    """
    Format value as currency (Latin America format).

    Parameters
    ----------
    value : float
        The monetary value
    locale : str
        Locale for formatting

    Returns
    -------
    str
        Formatted string
    """
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
