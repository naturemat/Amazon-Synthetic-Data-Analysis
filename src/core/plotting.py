"""
Figure helpers.

The project is meant to run as a batch pipeline: figures are written to
``docs/images`` instead of opening blocking windows. Set ``SHOW_PLOTS=1`` in the
environment to display them interactively as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

from ..config.settings import FIGURE_DPI, FIGURES_DIR, SHOW_PLOTS


def save_figure(fig=None, name: str = "figura", subdir: Optional[str] = None,
                show: Optional[bool] = None, dpi: Optional[int] = None,
                close: bool = True) -> str:
    """
    Save a Matplotlib figure to the project figures directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to save. Defaults to the current figure.
    name : str
        File name without extension.
    subdir : str, optional
        Optional subdirectory inside ``docs/images``.
    show : bool, optional
        Whether to also display the figure. Defaults to ``settings.SHOW_PLOTS``.
    dpi : int, optional
        Resolution. Defaults to ``settings.FIGURE_DPI``.
    close : bool
        Close the figure after saving to free memory.

    Returns
    -------
    str
        Path of the saved file.
    """
    fig = fig or plt.gcf()
    show = SHOW_PLOTS if show is None else show
    dpi = dpi or FIGURE_DPI

    target_dir = Path(FIGURES_DIR)
    if subdir:
        target_dir = target_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return str(path)


def close_all() -> None:
    """Close every open figure (useful at the end of a batch run)."""
    plt.close("all")
