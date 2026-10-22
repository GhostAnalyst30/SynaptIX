"""
Renderers de alto nivel de ViewX para resultados de SynaptIX.

Usan el mismo contrato que statslibx: serializan con ``to_report_data()``
y delegan en ``viewx.from_report_payload()``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from ._availability import require_viewx
from .adapters import to_report_data
from .export import HTMLTheme, PresentationTheme

ReportTarget = Literal["html", "report", "presentation"]


def render_html(
    result: Any,
    filename: str = "report.html",
    theme: HTMLTheme = "dark_enterprise",
    include_figures: bool = True,
    data: Optional[Any] = None,
    show: bool = False,
    **kwargs: Any,
) -> str:
    """
    Renderiza un resultado de synaptix a un dashboard HTML con ViewX.

    Parameters
    ----------
    result : resultado de synaptix
        Cualquier objeto soportado por ``to_report_data()``.
    filename : str
        Ruta del HTML de salida.
    theme : str
        Tema del dashboard de ViewX.
    include_figures : bool
        Incluir figuras Plotly en el payload.
    data : DataFrame, optional
        Datos fuente para figuras adicionales.
    show : bool
        Abrir el navegador al terminar.
    **kwargs
        Se reenvían a ``from_report_payload``.

    Returns
    -------
    str
        Ruta del archivo generado.

    Raises
    ------
    ImportError
        Si ViewX no está instalado.
    """
    require_viewx("render_html")
    from viewx.shared.stats_payload import from_report_payload

    payload = to_report_data(result, include_figures=include_figures, data=data)
    return from_report_payload(
        payload,
        target="html",
        filename=filename,
        theme=theme,
        show=show,
        **kwargs,
    )


def render_report(
    result: Any,
    filename: str = "ml_report",
    outdir: str = "output",
    include_figures: bool = False,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """
    Renderiza un resultado de synaptix a un reporte PDF con ViewX.

    Parameters
    ----------
    result : resultado de synaptix
        Cualquier objeto soportado por ``to_report_data()``.
    filename : str
        Nombre base del archivo sin extensión.
    outdir : str
        Directorio de salida.
    include_figures : bool
        Incrustar imágenes de las figuras cuando sea posible.
    data : DataFrame, optional
        Datos fuente para figuras opcionales.
    **kwargs
        Se reenvían a ``from_report_payload``.

    Returns
    -------
    str
        Ruta del PDF generado.

    Raises
    ------
    ImportError
        Si ViewX no está instalado.
    """
    require_viewx("render_report")
    from viewx.shared.stats_payload import from_report_payload

    payload = to_report_data(result, include_figures=include_figures, data=data)
    return from_report_payload(
        payload,
        target="report",
        filename=filename,
        outdir=outdir,
        **kwargs,
    )


def render_presentation(
    result: Any,
    filename: str = "presentation.html",
    theme: PresentationTheme = "dark",
    include_figures: bool = True,
    data: Optional[Any] = None,
    open_browser: bool = False,
    **kwargs: Any,
) -> str:
    """
    Renderiza un resultado de synaptix a una presentación HTML con ViewX.

    Parameters
    ----------
    result : resultado de synaptix
        Cualquier objeto soportado por ``to_report_data()``.
    filename : str
        Ruta HTML de salida.
    theme : str
        Tema de la presentación.
    include_figures : bool
        Incluir slides con gráficos cuando haya figuras disponibles.
    data : DataFrame, optional
        Datos fuente para figuras adicionales.
    open_browser : bool
        Abrir el navegador al exportar.
    **kwargs
        Se reenvían a ``from_report_payload``.

    Returns
    -------
    str
        Ruta de la presentación generada.

    Raises
    ------
    ImportError
        Si ViewX no está instalado.
    """
    require_viewx("render_presentation")
    from viewx.shared.stats_payload import from_report_payload

    payload = to_report_data(result, include_figures=include_figures, data=data)
    return from_report_payload(
        payload,
        target="presentation",
        filename=filename,
        theme=theme,
        open_browser=open_browser,
        **kwargs,
    )


def render(
    result: Any,
    target: ReportTarget = "html",
    **kwargs: Any,
) -> str:
    """
    Despacha el renderer según el engine de destino.

    Parameters
    ----------
    result : resultado de synaptix
        Objeto a serializar y renderizar.
    target : {'html', 'report', 'presentation'}
        Engine de ViewX a usar.
    **kwargs
        Se reenvían al renderer específico.

    Returns
    -------
    str
        Ruta del artefacto generado.
    """
    if target == "html":
        return render_html(result, **kwargs)
    if target == "report":
        return render_report(result, **kwargs)
    return render_presentation(result, **kwargs)
