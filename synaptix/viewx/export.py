"""
Mixin de exportación a ViewX para los objetos de resultado de SynaptIX.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from ._availability import require_viewx

HTMLTheme = Literal[
    "corporate_blue",
    "dark_enterprise",
    "modern_green",
    "void_indigo",
    "glass_ocean",
    "cyberpunk_neon",
]

PresentationTheme = Literal["dark", "light", "neon", "ocean", "sunset", "corporate"]


class ViewXExportMixin:
    """
    Mixin que agrega métodos de exportación a ViewX a los resultados de SynaptIX.

    Todos los métodos de exportación requieren ``viewx``
    (``pip install synaptix[viewx]``). El análisis ML de synaptix
    funciona sin ViewX instalado.
    """

    def to_report_data(
        self,
        include_figures: bool = False,
        data: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Serializa este resultado en un payload compatible con ViewX.

        No requiere que ViewX esté instalado.

        Parameters
        ----------
        include_figures : bool
            Si es True, adjunta figuras Plotly al payload (requiere plotly,
            ``pip install synaptix[viz]``).
        data : DataFrame, optional
            Datos fuente para figuras adicionales.
        **kwargs
            Se reenvían al serializador del adapter.

        Returns
        -------
        dict
            Payload con claves ``title``, ``sections``, ``tables`` y
            opcionales ``valueboxes`` y ``figures``.
        """
        from .adapters import to_report_data

        return to_report_data(
            self,
            include_figures=include_figures,
            data=data,
            **kwargs,
        )

    def to_html(
        self,
        filename: str = "report.html",
        theme: HTMLTheme = "dark_enterprise",
        include_figures: bool = True,
        data: Optional[Any] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Exporta este resultado a un dashboard HTML interactivo vía ViewX.

        Parameters
        ----------
        filename : str
            Ruta del HTML de salida.
        theme : str
            Tema soportado por el ThemeManager de ViewX.
        include_figures : bool
            Incluir figuras Plotly cuando estén disponibles.
        data : DataFrame, optional
            Datos fuente para figuras adicionales.
        show : bool
            Abrir el archivo generado en el navegador.
        **kwargs
            Opciones extra reenviadas al renderer.

        Returns
        -------
        str
            Ruta del archivo HTML generado.

        Raises
        ------
        ImportError
            Si ViewX no está instalado.
        """
        require_viewx("exportación HTML")
        from .renderers import render_html

        return render_html(
            self,
            filename=filename,
            theme=theme,
            include_figures=include_figures,
            data=data,
            show=show,
            **kwargs,
        )

    # Alias para resultados donde ``to_html`` colisiona con pandas
    # (por ejemplo ``ModelComparisonResult``, que hereda de DataFrame).
    def to_dashboard(self, *args: Any, **kwargs: Any) -> str:
        """Alias de :meth:`to_html` (dashboard HTML vía ViewX)."""
        return ViewXExportMixin.to_html(self, *args, **kwargs)

    def to_report(
        self,
        filename: str = "ml_report",
        outdir: str = "output",
        include_figures: bool = False,
        data: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Exporta este resultado a un reporte PDF vía el engine Report de ViewX.

        Parameters
        ----------
        filename : str
            Nombre base del archivo sin extensión.
        outdir : str
            Directorio de salida para el PDF y sus assets.
        include_figures : bool
            Incrustar figuras como imágenes cuando sea posible.
        data : DataFrame, optional
            Datos fuente para figuras opcionales.
        **kwargs
            Opciones extra del renderer.

        Returns
        -------
        str
            Ruta del PDF generado.

        Raises
        ------
        ImportError
            Si ViewX no está instalado.
        """
        require_viewx("exportación PDF")
        from .renderers import render_report

        return render_report(
            self,
            filename=filename,
            outdir=outdir,
            include_figures=include_figures,
            data=data,
            **kwargs,
        )

    def to_presentation(
        self,
        filename: str = "presentation.html",
        theme: PresentationTheme = "dark",
        include_figures: bool = True,
        data: Optional[Any] = None,
        open_browser: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Exporta este resultado a una presentación HTML vía ViewX Presentation.

        Parameters
        ----------
        filename : str
            Ruta HTML de salida para la presentación.
        theme : str
            Tema de la presentación.
        include_figures : bool
            Incluir slides con gráficos Plotly cuando estén disponibles.
        data : DataFrame, optional
            Datos fuente para figuras adicionales.
        open_browser : bool
            Abrir la presentación al terminar.
        **kwargs
            Opciones extra del renderer.

        Returns
        -------
        str
            Ruta del archivo de presentación generado.

        Raises
        ------
        ImportError
            Si ViewX no está instalado.
        """
        require_viewx("exportación de presentación")
        from .renderers import render_presentation

        return render_presentation(
            self,
            filename=filename,
            theme=theme,
            include_figures=include_figures,
            data=data,
            open_browser=open_browser,
            **kwargs,
        )
