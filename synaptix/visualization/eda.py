"""
Gráficas de análisis exploratorio (EDA) a partir de DataFrames.

Ejemplo
-------
>>> from synaptix.visualization import plot_distributions, plot_correlation
>>> plot_distributions(df)      # histogramas de todas las numéricas
>>> plot_correlation(df)        # heatmap de correlación
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_distributions",
    "plot_correlation",
    "plot_boxplots",
    "plot_scatter_matrix",
    "plot_missing",
]


def _numeric_columns(df: pd.DataFrame, columns: Optional[list]) -> list:
    if columns is not None:
        return columns
    return df.select_dtypes(include=[np.number]).columns.tolist()


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    bins: int = 30,
    color: str = "#4c9f70",
) -> None:
    """Histogramas de las columnas numéricas en una grilla.

    Parameters
    ----------
    df : DataFrame
        Datos a explorar.
    columns : list, optional
        Columnas a graficar. Por defecto, todas las numéricas.
    bins : int, default=30
        Número de bins por histograma.

    Ejemplo
    -------
    >>> plot_distributions(df, columns=["edad", "ingreso"])
    """
    cols = _numeric_columns(df, columns)
    if not cols:
        raise ValueError("No hay columnas numéricas para graficar.")

    n_cols = min(3, len(cols))
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        series = df[col].dropna()
        ax.hist(series, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(series.mean(), color="crimson", linestyle="--", linewidth=1.2, label="media")
        ax.axvline(series.median(), color="navy", linestyle=":", linewidth=1.2, label="mediana")
        ax.set_title(col, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Distribución de variables", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_correlation(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "pearson",
    cmap: str = "RdYlGn",
) -> None:
    """Heatmap de la matriz de correlación entre columnas numéricas.

    Parameters
    ----------
    df : DataFrame
        Datos a analizar.
    columns : list, optional
        Columnas a incluir. Por defecto, todas las numéricas.
    method : str, default="pearson"
        "pearson", "spearman" o "kendall".

    Ejemplo
    -------
    >>> plot_correlation(df, method="spearman")
    """
    cols = _numeric_columns(df, columns)
    if len(cols) < 2:
        raise ValueError("Se necesitan al menos 2 columnas numéricas.")

    corr = df[cols].corr(method=method)

    fig, ax = plt.subplots(figsize=(0.8 * len(cols) + 3, 0.7 * len(cols) + 2.5))
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(image, label="correlación")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.6 else "black",
            )

    ax.set_title(f"Matriz de correlación ({method})")
    plt.tight_layout()
    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    by: Optional[str] = None,
) -> None:
    """Boxplots de columnas numéricas (útil para detectar outliers).

    Parameters
    ----------
    df : DataFrame
        Datos a explorar.
    columns : list, optional
        Columnas a graficar. Por defecto, todas las numéricas.
    by : str, optional
        Columna categórica para segmentar los boxplots
        (por ejemplo ``by="species"``).

    Ejemplo
    -------
    >>> plot_boxplots(df, columns=["precio"], by="ciudad")
    """
    cols = _numeric_columns(df, columns)
    if by is not None and by in cols:
        cols.remove(by)
    if not cols:
        raise ValueError("No hay columnas numéricas para graficar.")

    if by is None:
        fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 3, 4.5))
        data = [df[col].dropna() for col in cols]
        ax.boxplot(data, tick_labels=cols)
        ax.set_title("Boxplots")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
    else:
        groups = df[by].dropna().unique()
        n_cols = min(3, len(cols))
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cols):
            data = [df.loc[df[by] == g, col].dropna() for g in groups]
            ax.boxplot(data, tick_labels=[str(g) for g in groups])
            ax.set_title(f"{col} por {by}", fontsize=11)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(alpha=0.25)
        for ax in axes[len(cols):]:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    target: Optional[str] = None,
    max_cols: int = 6,
) -> None:
    """Matriz de dispersión entre pares de columnas numéricas.

    Parameters
    ----------
    df : DataFrame
        Datos a explorar.
    columns : list, optional
        Columnas a incluir (máximo ``max_cols``).
    target : str, optional
        Columna categórica para colorear los puntos.
    max_cols : int, default=6
        Límite de columnas para mantener el gráfico legible.

    Ejemplo
    -------
    >>> plot_scatter_matrix(df, target="species")
    """
    cols = _numeric_columns(df, columns)[:max_cols]
    if len(cols) < 2:
        raise ValueError("Se necesitan al menos 2 columnas numéricas.")

    # Filtrar nulos antes de graficar para que los colores coincidan
    # con las filas que pandas efectivamente dibuja.
    subset = cols + ([target] if target is not None and target not in cols else [])
    data = df[subset].dropna()

    colors = None
    if target is not None:
        categories = pd.Categorical(data[target])
        palette = plt.cm.tab10(np.linspace(0, 1, 10))
        colors = palette[categories.codes % 10]

    axes = pd.plotting.scatter_matrix(
        data[cols], figsize=(2.2 * len(cols) + 2, 2.2 * len(cols) + 2),
        diagonal="hist", alpha=0.7, c=colors,
    )
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    plt.suptitle(
        "Matriz de dispersión" + (f" (color: {target})" if target else ""),
        fontsize=13,
    )
    plt.show()


def plot_missing(df: pd.DataFrame) -> None:
    """Gráfico de barras con el porcentaje de valores faltantes por columna.

    Ejemplo
    -------
    >>> plot_missing(df)
    """
    missing = (df.isnull().mean() * 100).sort_values(ascending=True)
    missing = missing[missing > 0]

    if missing.empty:
        print("No hay valores faltantes en el DataFrame.")
        return

    plt.figure(figsize=(7, max(2.5, 0.4 * len(missing))))
    bars = plt.barh(missing.index, missing.values, color="#e07a5f")
    for bar, value in zip(bars, missing.values):
        plt.text(
            value + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=9,
        )
    plt.xlabel("% de valores faltantes")
    plt.title("Valores faltantes por columna")
    plt.grid(alpha=0.25, axis="x")
    plt.tight_layout()
    plt.show()
