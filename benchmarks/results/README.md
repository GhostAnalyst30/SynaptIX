# Resultados de los benchmarks

Generado por `benchmarks/make_report.py` a partir de los JSON de esta carpeta. Reproducir con `python benchmarks/bench_engines.py`, `python benchmarks/bench_automl.py` y luego este script.

## Entorno

| Elemento | Valor |
|---|---|
| Fecha | 2026-11-11 21:15:31 |
| Python | 3.15.1 |
| Sistema | Windows-10-10.0.19045-SP0 |
| CPU | Intel64 Family 6 Model 142 Stepping 9, GenuineIntel (4 hilos lógicos) |
| RAM | 12.8 GB |
| synaptix | 0.3.0 (C:\Users\Usuario\Documents\Emmanuel Ascendra\GitHub_GhostAnalyst\Librerias_Python\Libreria_MachineLearning\synaptix) |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| polars | 2.1.0 |
| pyarrow | 26.0.1 |
| optuna | 5.0.0 |
| flaml | 2.8.0 |
| pycaret | no instalado |
| autogluon.tabular | no instalado |
| psutil | 7.2.2 |

## 1. pandas vs polars por etapa

Mediana de 3 ejecuciones, cada una en un proceso nuevo. `pico MB` = RSS máximo durante la tarea menos el RSS al empezarla (librerías ya importadas).

### Carga (archivo → DataFrame de pandas listo para scikit-learn)

| filas | variante | formato | segundos | pico MB |
|---:|---|---|---:|---:|
| 10,000 | pandas | csv | 0.024 | 7.2 |
| 10,000 | polars | csv | 0.037 | 17.0 |
| 10,000 | pandas | parquet | 0.042 | 11.2 |
| 10,000 | polars | parquet | 0.035 | 13.6 |
| 100,000 | pandas | csv | 0.067 | 63.0 |
| 100,000 | polars | csv | 0.155 | 85.7 |
| 100,000 | pandas | parquet | 0.068 | 45.4 |
| 100,000 | polars | parquet | 0.044 | 57.1 |
| 1,000,000 | pandas | csv | 0.563 | 504.0 |
| 1,000,000 | polars | csv | 0.707 | 563.0 |
| 1,000,000 | pandas | parquet | 0.281 | 328.1 |
| 1,000,000 | polars | parquet | 0.293 | 400.9 |

### Perfilado de datos (`profile_data`)

| filas | variante | formato | segundos | pico MB |
|---:|---|---|---:|---:|
| 10,000 | pandas | memoria | 0.072 | 15.6 |
| 10,000 | polars | memoria | 0.058 | 37.9 |
| 10,000 | polars_desde_pandas | memoria | 0.075 | 33.4 |
| 20,000 | pandas | memoria | 0.103 | 19.9 |
| 20,000 | polars | memoria | 0.064 | 53.6 |
| 20,000 | polars_desde_pandas | memoria | 0.081 | 47.2 |
| 50,000 | pandas | memoria | 0.278 | 44.4 |
| 50,000 | polars | memoria | 0.113 | 102.1 |
| 50,000 | polars_desde_pandas | memoria | 0.140 | 93.3 |
| 100,000 | pandas | memoria | 0.577 | 78.5 |
| 100,000 | polars | memoria | 0.186 | 157.4 |
| 100,000 | polars_desde_pandas | memoria | 0.198 | 134.7 |
| 1,000,000 | pandas | memoria | 9.531 | 439.3 |
| 1,000,000 | polars | memoria | 2.547 | 998.2 |
| 1,000,000 | polars_desde_pandas | memoria | 2.463 | 1006.6 |

### Preprocesamiento (`ColumnTransformer`, igual en ambos motores)

| filas | variante | formato | segundos | pico MB |
|---:|---|---|---:|---:|
| 10,000 | sklearn | memoria | 0.212 | 19.1 |
| 100,000 | sklearn | memoria | 0.927 | 172.9 |
| 1,000,000 | sklearn | memoria | 8.969 | 1294.2 |

### Flujo completo (`AutoPipeline.fit`, HistGradientBoosting, cv=3)

| filas | variante | formato | segundos | pico MB |
|---:|---|---|---:|---:|
| 10,000 | pandas | csv | 6.253 | 29.5 |
| 10,000 | polars | csv | 6.053 | 56.9 |
| 100,000 | pandas | csv | 19.776 | 173.4 |
| 100,000 | polars | csv | 19.476 | 277.9 |

## 2. SynaptIX frente a scikit-learn y AutoML externos

Mismo split 80/20 para todas las herramientas; evaluación única en test. Media ± desviación estándar entre semillas (estabilidad) del score de test; tiempos y memoria: mediana entre semillas.

Disponibilidad de herramientas externas en este entorno:

- **flaml**: disponible
- **pycaret**: no disponible: ModuleNotFoundError: No module named 'pycaret'
- **autogluon**: no disponible: ModuleNotFoundError: No module named 'autogluon'

### classification — 5,000 filas

| herramienta | accuracy | balanced_accuracy | entrenamiento total s | preprocesamiento s | ajuste final s | inferencia ms/1k filas | pico MB | modelo elegido (por semilla) | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sklearn_baseline | 0.8317 ± 0.0132 | 0.8315 ± 0.0130 | 1.08 | 0.036 | 1.054 | 18.14 | 17 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| sklearn_random_search | 0.8350 ± 0.0155 | 0.8346 ± 0.0153 | 35.97 | 0.031 | 0.759 | 16.08 | 23 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| synaptix_fast | 0.8380 ± 0.0070 | 0.8379 ± 0.0066 | 18.74 | 0.027 | 0.730 | 13.74 | 19 | hist_gradient_boosting, hist_gradient_boosting, hist_gradient_boosting | 3 |
| synaptix_balanced | 0.8353 ± 0.0072 | 0.8351 ± 0.0073 | 67.66 | 0.027 | 1.705 | 13.31 | 40 | gradient_boosting, gradient_boosting, gradient_boosting | 3 |
| flaml | 0.8290 ± 0.0087 | 0.8289 ± 0.0082 | 68.87 | — | — | 37.61 | 42 | lgbm, xgboost, lgbm | 3 |

### classification — 50,000 filas

| herramienta | accuracy | balanced_accuracy | entrenamiento total s | preprocesamiento s | ajuste final s | inferencia ms/1k filas | pico MB | modelo elegido (por semilla) | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sklearn_baseline | 0.8422 ± 0.0022 | 0.8421 ± 0.0022 | 2.79 | 0.148 | 2.648 | 8.36 | 75 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| sklearn_random_search | 0.8425 ± 0.0030 | 0.8424 ± 0.0030 | 72.30 | 0.141 | 2.739 | 9.79 | 80 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| synaptix_fast | 0.8433 ± 0.0029 | 0.8432 ± 0.0029 | 60.78 | 0.181 | 2.303 | 7.64 | 114 | hist_gradient_boosting, hist_gradient_boosting, hist_gradient_boosting | 3 |
| flaml | 0.8411 ± 0.0018 | 0.8409 ± 0.0018 | 62.39 | — | — | 7.16 | 114 | xgboost, lgbm, xgb_limitdepth | 3 |

### penguins — 344 filas

| herramienta | accuracy | balanced_accuracy | entrenamiento total s | preprocesamiento s | ajuste final s | inferencia ms/1k filas | pico MB | modelo elegido (por semilla) | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sklearn_baseline | 0.9758 ± 0.0084 | 0.9765 ± 0.0131 | 0.36 | 0.017 | 0.341 | 220.00 | 6 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| sklearn_random_search | 0.9855 ± 0.0145 | 0.9839 ± 0.0191 | 13.02 | 0.022 | 0.418 | 387.00 | 8 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| synaptix_fast | 0.9855 ± 0.0000 | 0.9881 ± 0.0013 | 4.28 | 0.020 | 0.033 | 218.73 | 10 | logistic, hist_gradient_boosting, logistic | 3 |
| synaptix_balanced | 0.9903 ± 0.0084 | 0.9919 ± 0.0071 | 17.22 | 0.017 | 0.080 | 173.56 | 12 | extra_trees, logistic, extra_trees | 3 |
| flaml | 0.9903 ± 0.0084 | 0.9926 ± 0.0064 | 17.64 | — | — | 704.20 | 26 | extra_tree, lgbm, extra_tree | 3 |

### regression — 5,000 filas

| herramienta | r2 | rmse | entrenamiento total s | preprocesamiento s | ajuste final s | inferencia ms/1k filas | pico MB | modelo elegido (por semilla) | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sklearn_baseline | 0.9234 ± 0.0073 | 1.6907 ± 0.0578 | 0.92 | 0.100 | 0.828 | 19.70 | 18 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| sklearn_random_search | 0.9243 ± 0.0060 | 1.6805 ± 0.0579 | 30.92 | 0.105 | 0.877 | 16.44 | 26 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| synaptix_fast | 0.9253 ± 0.0082 | 1.6683 ± 0.0791 | 23.10 | 0.029 | 1.313 | 20.61 | 37 | hist_gradient_boosting, hist_gradient_boosting, hist_gradient_boosting | 3 |
| synaptix_balanced | 0.9301 ± 0.0081 | 1.6143 ± 0.0831 | 63.47 | 0.028 | 1.658 | 10.79 | 62 | gradient_boosting, gradient_boosting, gradient_boosting | 3 |
| flaml | 0.9233 ± 0.0199 | 1.6841 ± 0.2211 | 63.91 | — | — | 37.90 | 44 | xgboost, lgbm, xgb_limitdepth | 3 |

### regression — 50,000 filas

| herramienta | r2 | rmse | entrenamiento total s | preprocesamiento s | ajuste final s | inferencia ms/1k filas | pico MB | modelo elegido (por semilla) | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sklearn_baseline | 0.9389 ± 0.0029 | 1.4858 ± 0.0381 | 2.70 | 0.235 | 2.463 | 12.43 | 72 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| sklearn_random_search | 0.9390 ± 0.0028 | 1.4851 ± 0.0367 | 63.62 | 0.215 | 2.373 | 9.74 | 77 | HistGradientBoosting, HistGradientBoosting, HistGradientBoosting | 3 |
| synaptix_fast | 0.9397 ± 0.0029 | 1.4761 ± 0.0388 | 136.20 | 0.194 | 3.181 | 11.87 | 291 | hist_gradient_boosting, hist_gradient_boosting, hist_gradient_boosting | 3 |
| flaml | 0.9405 ± 0.0026 | 1.4663 ± 0.0328 | 137.53 | — | — | 8.29 | 163 | lgbm, lgbm, xgb_limitdepth | 3 |

