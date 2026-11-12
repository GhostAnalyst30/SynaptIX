"""
Reportes de AutoPipeline: consola, diccionario, JSON y HTML autocontenido.

La validación cruzada (en el conjunto de entrenamiento, usada para elegir)
y la evaluación final (en el conjunto de prueba, evaluado una sola vez) se
presentan siempre en secciones separadas.

Matplotlib solo se importa si se piden gráficos.
"""

from __future__ import annotations

import base64
import html
import io
import json
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .experiments import to_jsonable
from .explain import CAUSAL_NOTE

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import AutoPipeline

SELECTION_NOTE = (
    "La puntuación de validación del modelo elegido es ligeramente optimista: "
    "se eligió el máximo entre varios candidatos. La estimación imparcial es la "
    "del conjunto de prueba."
)


def build_report(pipe: "AutoPipeline") -> Dict[str, Any]:
    """Contenido del reporte en 12 secciones."""
    profile = pipe.profile_
    spec = pipe.metric_specs_[0]
    statuses = pipe.ranking_["estado"].value_counts().to_dict()
    tuning = {
        name: {"estrategia": r.strategy, "combinaciones": r.n_evaluated,
               "fallidas": r.n_failed, "mejor_cv": float(spec.to_display(r.best_score)),
               "mejores_parametros": r.best_params, "segundos": round(r.seconds, 3),
               "nota": r.note}
        for name, r in pipe.tuning_results_.items()
    }
    importance = None
    if pipe.feature_importances_ is not None:
        importance = pipe.feature_importances_.head(20)
    report = {
        "1_dataset": {
            "filas": profile.n_rows, "columnas": profile.n_columns,
            "memoria_mb": profile.memory_mb, "duplicados": profile.n_duplicates,
            "entrada": pipe._input_kind, "engine": pipe._engine_used,
            "perfilado_s": round(profile.seconds, 4),
        },
        "2_tarea": {
            "tarea": pipe.task_, "target": pipe.target_,
            "deteccion": None if pipe.task_detection_ is None else {
                "confianza": pipe.task_detection_.confidence,
                "criterios": pipe.task_detection_.reasons},
            "target_resumen": profile.target,
        },
        "3_variables": {k: v for k, v in pipe.feature_types_.items() if v},
        "4_calidad": profile.warnings_frame(),
        "5_modelos": {"evaluados": int(len(pipe.ranking_)), "estados": statuses},
        "6_ranking": pipe.ranking_,
        "7_validacion": {
            "estrategia": pipe.cv_strategy_, "metrica": spec.display,
            "modelo": pipe.best_name_, "media": pipe.cv_score_.get("mean"),
            "desviacion": pipe.cv_score_.get("std"), "origen": pipe.cv_score_.get("source"),
            "nota": SELECTION_NOTE,
        },
        "8_prueba": {
            "holdout": pipe.holdout_, "n_train": pipe.n_train_, "n_test": pipe.n_test_,
            "metrica_principal": None if pipe.test_score_ is None else {
                spec.display: pipe.test_score_},
            "metricas": dict(pipe.results_ or {}),
        },
        "9_hiperparametros": {"modelo_final": pipe.best_params_, "busquedas": tuning,
                              "ensemble": pipe.ensemble_result_},
        "10_importancia": {
            "permutacion_test": importance,
            "modelo": None if pipe.model_importances_ is None
            else pipe.model_importances_.head(20),
            "nota": CAUSAL_NOTE,
        },
        "11_tiempos": {"etapas_s": pipe.timings_,
                       "total_s": round(sum(pipe.timings_.values()), 3),
                       "pico_rss_mb": None if pipe.profiler_ is None
                       else pipe.profiler_.peak_rss_mb,
                       "n_jobs": pipe.n_jobs},
        "12_advertencias": {"advertencias": pipe.warnings_, "notas": pipe.notes_},
    }
    return report


# ---------------------------------------------------------------------------
# Consola
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    d = report["1_dataset"]
    add("\n================ AutoPipeline ================")
    add(f"1. Dataset      : {d['filas']} filas x {d['columnas']} columnas, "
        f"{d['memoria_mb']} MB, {d['duplicados']} duplicados "
        f"(entrada {d['entrada']}, engine {d['engine']})")
    t = report["2_tarea"]
    det = t["deteccion"] or {}
    add(f"2. Tarea        : {t['tarea']} (target '{t['target']}'; "
        f"detección {det.get('confianza', '-')}: {'; '.join(det.get('criterios', []))})")
    add("3. Variables    : " + ", ".join(f"{k}={len(v)}" for k, v in report["3_variables"].items()))
    quality = report["4_calidad"]
    add(f"4. Calidad      : {len(quality)} observaciones")
    for _, row in quality.iterrows():
        col = f"{row['column']}: " if row["column"] else ""
        add(f"     [{row['severity']}] {row['code']} {col}{row['message']}")
    m = report["5_modelos"]
    add(f"5. Modelos      : {m['evaluados']} evaluados {m['estados']}")
    add("6. Ranking (validación cruzada en train):")
    table = report["6_ranking"].drop(columns=["parametros"], errors="ignore")
    add(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    v = report["7_validacion"]
    add(f"7. Validación   : {v['modelo']} -> {v['metrica']} = {_fmt(v['media'])} "
        f"± {_fmt(v['desviacion'])} ({v['estrategia']}, {v['origen']})")
    p = report["8_prueba"]
    add(f"8. Prueba       : holdout {p['holdout']} ({p['n_train']} train / {p['n_test']} test), "
        "evaluado una sola vez")
    for key, value in p["metricas"].items():
        add(f"     {key:<18}: {_fmt(value)}")
    h = report["9_hiperparametros"]
    add(f"9. Parámetros   : {h['modelo_final']}")
    for name, info in h["busquedas"].items():
        add(f"     {name}: {info['estrategia']}, {info['combinaciones']} combinaciones, "
            f"mejor CV {_fmt(info['mejor_cv'])} {('- ' + info['nota']) if info['nota'] else ''}")
    if h["ensemble"]:
        add(f"     ensemble: {h['ensemble']}")
    imp = report["10_importancia"]
    if imp["permutacion_test"] is not None:
        add("10. Importancia (permutación en prueba, top 10):")
        for _, row in imp["permutacion_test"].head(10).iterrows():
            add(f"     {row['feature']:<24} {row['importance_mean']:.4f} "
                f"± {row['importance_std']:.4f}")
    elif imp["modelo"] is not None:
        add("10. Importancia (del modelo, top 10):")
        for _, row in imp["modelo"].head(10).iterrows():
            add(f"     {row['feature']:<24} {row['importance']:.4f}")
    add(f"     Nota: {imp['nota']}")
    tm = report["11_tiempos"]
    add(f"11. Tiempos     : total {tm['total_s']} s; " + ", ".join(
        f"{k}={v:.2f}s" for k, v in tm["etapas_s"].items())
        + (f"; pico RSS {tm['pico_rss_mb']:.0f} MB" if tm["pico_rss_mb"] else ""))
    w = report["12_advertencias"]
    add("12. Advertencias metodológicas:")
    for item in w["advertencias"] or ["(ninguna)"]:
        add(f"     - {item}")
    for item in w["notas"]:
        add(f"     * {item}")
    add(f"     * {v['nota']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def build_figures(pipe: "AutoPipeline") -> List[Tuple[str, Any]]:
    """Figuras de matplotlib del modelo final (vacío sin datos de prueba)."""
    import matplotlib.pyplot as plt
    from sklearn import metrics as skm

    figures: List[Tuple[str, Any]] = []
    ev = pipe._eval
    if ev:
        y_true, y_pred, proba = ev["y_true"], ev["y_pred"], ev["proba"]
        if pipe.task_ == "classification":
            cm = pipe.diagnostics_["confusion_matrix"] if pipe.diagnostics_ else None
            if cm is not None:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm.to_numpy(), cmap="Blues")
                ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(cm.index)), cm.index)
                for (i, j), value in np.ndenumerate(cm.to_numpy()):
                    ax.text(j, i, str(value), ha="center", va="center")
                ax.set_xlabel("predicho")
                ax.set_ylabel("real")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                figures.append(("Matriz de confusión (prueba)", fig))
            if proba is not None:
                classes = list(pipe.classes_)
                fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))
                targets = [classes[-1]] if len(classes) == 2 else classes
                for label in targets:
                    col = classes.index(label)
                    positive = (y_true == label)
                    fpr, tpr, _ = skm.roc_curve(positive, proba[:, col])
                    prec, rec, _ = skm.precision_recall_curve(positive, proba[:, col])
                    ax_roc.plot(fpr, tpr, label=f"{label} (AUC={skm.auc(fpr, tpr):.3f})")
                    ax_pr.plot(rec, prec, label=f"{label}")
                ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC (prueba)")
                ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-Recall (prueba)")
                ax_roc.legend(fontsize=8)
                ax_pr.legend(fontsize=8)
                fig.tight_layout()
                figures.append(("Curvas ROC y Precision-Recall", fig))
        else:
            residuals = y_true - y_pred
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].scatter(y_true, y_pred, s=10, alpha=0.6)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            axes[0].plot(lims, lims, "k--", lw=0.8)
            axes[0].set(xlabel="real", ylabel="predicho", title="Real vs predicho (prueba)")
            axes[1].scatter(y_pred, residuals, s=10, alpha=0.6)
            axes[1].axhline(0, color="k", lw=0.8)
            axes[1].set(xlabel="predicho", ylabel="residuo", title="Residuos vs predicho")
            axes[2].hist(residuals, bins=30)
            axes[2].set(xlabel="residuo", title="Distribución de residuos")
            fig.tight_layout()
            figures.append(("Diagnóstico de residuos", fig))
    importance = pipe.feature_importances_
    column, title = "importance_mean", "Importancia por permutación (prueba)"
    if importance is None and pipe.model_importances_ is not None:
        importance, column, title = pipe.model_importances_, "importance", "Importancia del modelo"
    if importance is not None and len(importance):
        top = importance.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(top) + 1.2))
        ax.barh(top["feature"].astype(str), top[column])
        ax.set_title(title)
        fig.tight_layout()
        figures.append((title, fig))
    return figures


def _figure_to_base64(fig: Any) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML / JSON
# ---------------------------------------------------------------------------
_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;color:#1d2433;max-width:1100px}
h1{font-size:1.6rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #dde;padding-bottom:.3rem}
table{border-collapse:collapse;font-size:.85rem;margin:.5rem 0}td,th{border:1px solid #dde;padding:.25rem .5rem;text-align:left}
th{background:#f3f5fa}.note{color:#555;font-size:.9rem}.warn{color:#9a5b00}img{max-width:100%}
"""


def _html_value(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return "<p class='note'>(sin datos)</p>"
        return value.to_html(index=False, float_format=lambda v: f"{v:.4f}", escape=True)
    if isinstance(value, dict):
        rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{_html_value(v)}</td></tr>"
                       for k, v in value.items())
        return f"<table>{rows}</table>"
    if isinstance(value, list):
        return "<ul>" + "".join(f"<li>{_html_value(v)}</li>" for v in value) + "</ul>"
    if isinstance(value, (float, np.floating)):
        return _fmt(value)
    return html.escape(str(value))


def render_html(report: Dict[str, Any], figures: List[Tuple[str, str]], experiment_id: str) -> str:
    titles = {
        "1_dataset": "1. Resumen del dataset", "2_tarea": "2. Tipo de problema",
        "3_variables": "3. Variables detectadas", "4_calidad": "4. Calidad de datos",
        "5_modelos": "5. Modelos evaluados", "6_ranking": "6. Ranking (validación cruzada en train)",
        "7_validacion": "7. Métricas de validación",
        "8_prueba": "8. Métricas de prueba (evaluado una sola vez)",
        "9_hiperparametros": "9. Hiperparámetros", "10_importancia": "10. Importancia de variables",
        "11_tiempos": "11. Tiempo y recursos", "12_advertencias": "12. Advertencias metodológicas",
    }
    parts = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'>"
             f"<title>SynaptIX AutoML {html.escape(experiment_id)}</title><style>{_CSS}</style>"
             f"</head><body><h1>SynaptIX AutoML</h1><p class='note'>Experimento "
             f"{html.escape(experiment_id)}</p>"]
    for key, title in titles.items():
        value = report[key]
        if isinstance(value, pd.DataFrame) and "parametros" in value.columns:
            value = value.assign(parametros=value["parametros"].map(
                lambda p: json.dumps(to_jsonable(p), ensure_ascii=False)))
        parts.append(f"<h2>{title}</h2>{_html_value(value)}")
    if figures:
        parts.append("<h2>Gráficos</h2>")
        for title, encoded in figures:
            parts.append(f"<h3>{html.escape(title)}</h3>"
                         f"<img alt='{html.escape(title)}' src='data:image/png;base64,{encoded}'>")
    parts.append("</body></html>")
    return "".join(parts)


def render_report(pipe: "AutoPipeline", plot: bool = True, save: Optional[str] = None) -> None:
    """Imprime el reporte; con ``save`` lo exporta a ``.html`` o ``.json``."""
    report = build_report(pipe)
    print(render_text(report))

    figures: List[Tuple[str, Any]] = []
    if plot:
        try:
            figures = build_figures(pipe)
        except ImportError:
            warnings.warn("matplotlib no está instalado: se omiten los gráficos.", stacklevel=3)

    if save is not None:
        lower = save.lower()
        if lower.endswith(".json"):
            payload = {"report": to_jsonable(report),
                       "experiment": pipe.experiment_.to_dict()}
            with open(save, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
        elif lower.endswith((".html", ".htm")):
            encoded = [(title, _figure_to_base64(fig)) for title, fig in figures]
            with open(save, "w", encoding="utf-8") as file:
                file.write(render_html(report, encoded, pipe.experiment_.experiment_id))
        else:
            raise ValueError("save debe terminar en .html o .json")

    if figures:
        import matplotlib.pyplot as plt

        if save is None:
            plt.show()
        for _, fig in figures:
            plt.close(fig)
