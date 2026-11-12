"""Genera ``benchmarks/results/README.md`` a partir de los JSON de resultados.

Todas las cifras del README salen de los archivos JSON; este script no
ejecuta ningún benchmark.

Uso::

    python benchmarks/make_report.py
"""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _load(name):
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _env_table(env):
    pk = env["packages"]
    rows = [
        ("Fecha", env["date"]), ("Python", env["python"]), ("Sistema", env["platform"]),
        ("CPU", f"{env['processor']} ({env['logical_cpus']} hilos lógicos)"),
        ("RAM", f"{env['ram_gb']} GB"),
    ] + [(name, version or "no instalado") for name, version in pk.items()]
    lines = ["| Elemento | Valor |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def engines_section(data, crossover):
    out = ["## 1. pandas vs polars por etapa", "",
           f"Mediana de {data['repeats']} ejecuciones, cada una en un proceso nuevo. "
           "`pico MB` = RSS máximo durante la tarea menos el RSS al empezarla "
           "(librerías ya importadas).", ""]
    rows = list(data["results"]) + (crossover["results"] if crossover else [])
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    titles = {
        "carga": "Carga (archivo → DataFrame de pandas listo para scikit-learn)",
        "perfilado": "Perfilado de datos (`profile_data`)",
        "preprocesamiento": "Preprocesamiento (`ColumnTransformer`, igual en ambos motores)",
        "flujo_completo": "Flujo completo (`AutoPipeline.fit`, HistGradientBoosting, cv=3)",
    }
    for stage, title in titles.items():
        if stage not in by_stage:
            continue
        out += [f"### {title}", "", "| filas | variante | formato | segundos | pico MB |",
                "|---:|---|---|---:|---:|"]
        for r in sorted(by_stage[stage], key=lambda r: (r["rows"], r["format"], r["variant"])):
            out.append(f"| {r['rows']:,} | {r['variant']} | {r['format']} | "
                       f"{_fmt(r['seconds'])} | {_fmt(r['peak_mb'], 1)} |")
        out.append("")
    return out


def automl_section(data):
    out = ["## 2. SynaptIX frente a scikit-learn y AutoML externos", "",
           "Mismo split 80/20 para todas las herramientas; evaluación única en test. "
           "Media ± desviación estándar entre semillas (estabilidad) del score de test; "
           "tiempos y memoria: mediana entre semillas.", "",
           "Disponibilidad de herramientas externas en este entorno:", ""]
    for tool, status in data["availability"].items():
        out.append(f"- **{tool}**: {status}")
    out.append("")
    groups = defaultdict(list)
    for r in data["results"]:
        groups[(r["dataset"], r["rows"])].append(r)
    for (dataset, rows), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        metric = "test_accuracy" if "test_accuracy" in items[0] else "test_r2"
        second = "test_balanced_accuracy" if metric == "test_accuracy" else "test_rmse"
        out += [f"### {dataset} — {rows:,} filas", "",
                f"| herramienta | {metric[5:]} | {second[5:]} | entrenamiento total s | "
                "preprocesamiento s | ajuste final s | inferencia ms/1k filas | pico MB | "
                "modelo elegido (por semilla) | semillas |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
        by_tool = defaultdict(list)
        for r in items:
            by_tool[r["tool"]].append(r)
        for tool, runs in by_tool.items():
            def ms(key):
                vals = [r[key] for r in runs if r.get(key) is not None]
                if not vals:
                    return "—"
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                return f"{mean:.4f} ± {std:.4f}"

            def med(key, digits=2):
                vals = [r[key] for r in runs if r.get(key) is not None]
                return _fmt(statistics.median(vals), digits) if vals else "—"

            chosen = ", ".join(str(r["selected"]) for r in runs)
            out.append(f"| {tool} | {ms(metric)} | {ms(second)} | {med('train_total_s')} | "
                       f"{med('preprocess_s', 3)} | {med('final_fit_s', 3)} | "
                       f"{med('inference_ms_per_1k')} | {med('peak_mb', 0)} | {chosen} | "
                       f"{len(runs)} |")
        out.append("")
    return out


def main():
    engines = _load("engines")
    crossover = _load("engines_profile_crossover")
    automl = _load("automl")
    env = (automl or engines)["environment"]
    lines = ["# Resultados de los benchmarks", "",
             "Generado por `benchmarks/make_report.py` a partir de los JSON de esta carpeta. "
             "Reproducir con `python benchmarks/bench_engines.py`, "
             "`python benchmarks/bench_automl.py` y luego este script.", "",
             "## Entorno", "", _env_table(env), ""]
    if engines:
        lines += engines_section(engines, crossover)
    if automl:
        lines += automl_section(automl)
    path = os.path.join(RESULTS, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
