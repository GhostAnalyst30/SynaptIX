# Registro de cambios

## 0.3.0 — AutoML Engine 2.0

Guía completa: [`docs/AUTOML.md`](docs/AUTOML.md). Benchmarks:
[`benchmarks/results/README.md`](benchmarks/results/README.md).

### Corregido (metodología)

- **Fuga de datos en `AutoPipeline`**: `DataCleaner` se ajustaba con el
  dataset completo antes del split (el escalador veía las 344 filas de
  penguins en vez de las 275 de train). Ahora el test se separa sobre los
  datos crudos y el preprocesamiento vive dentro de un `Pipeline` de
  scikit-learn que se reajusta en cada fold y en cada combinación de la
  búsqueda.
- La misma fuga en el `Pipeline` declarativo (`synaptix.pipeline.Pipeline`).
- `drop_duplicates` antes del split ocultaba duplicados entre train y test;
  ahora se reportan (`duplicates`, `train_test_overlap`) y no se borran.
- Un modelo fallido podía quedar primero en el ranking (score NaN). Ahora
  queda `failed` con su error y, si todos fallan, se lanza
  `AllModelsFailedError` con el detalle.
- Una columna 100 % nula producía un error críptico; ahora se imputa y se
  avisa (`all_missing`).
- `detect_task` dependía de `dtype == object` (frágil con el dtype `str` de
  pandas 3) y de un solo criterio.
- `tests/test_smoke.py`: el decorador `@check` capturaba las excepciones y
  pytest marcaba todo como aprobado.
- `save()` ya no guarda los datos de test dentro del pickle.

### Añadido

- Subpaquete `synaptix.automl` (data, features, evaluation, selection,
  tuning, ensemble, explain, parallel, profiling, experiments, reporting).
- `AutoPipeline`: `engine`, `mode` (`fast`/`balanced`/`exhaustive`),
  `exclude_models`, `cv_strategy`, `cv_repeats`, `scoring`, `metrics`,
  `tuning` (`grid`, `random`, `halving`, `optuna`/`bayesian`), `n_iter`,
  `tune_top_k`, `time_budget`, `param_distributions`, `test_size=None`,
  `impute_strategy="knn"`, `max_onehot_categories`, `drop_constant`,
  `feature_selection`, `ensemble` (`voting`, `stacking`, `blending`),
  `explain`, `ignore_columns`, `n_jobs`, `pre_dispatch`,
  `memory_optimization`, `cache`; `fit(..., groups=, time_column=)`.
- Entradas polars `DataFrame`/`LazyFrame`, NumPy y rutas `.csv`/`.parquet`.
- `DataProfile` con advertencias (constantes, IDs, nulos, cardinalidad,
  desbalance, duplicados, sospechas de fuga) sin eliminar columnas.
- Métricas: PR-AUC, MCC, log loss, MedAE y MAPE seguro con objetivos ≈ 0.
- Ranking con media, desviación, mejor y peor fold, tiempos de ajuste y
  predicción, parámetros, estado y error.
- Explicabilidad: permutación sobre test, importancias nativas,
  coeficientes, SHAP opcional y diagnósticos de errores, con nota de no
  causalidad.
- `ExperimentRecord` con id, versiones, semilla, datos, ranking, métricas,
  tiempos y advertencias; exportación JSON/CSV.
- `report(plot=True, save="reporte.html" | "reporte.json")` con validación
  y test en secciones separadas.
- Wrappers `ExtraTreesClassifier/Regressor`,
  `HistGradientBoostingClassifier/Regressor` y `ElasticNetRegression`.
- Extras `synaptix[automl]` (polars, pyarrow, optuna, psutil) y
  `synaptix[explain]` (shap).
- `benchmarks/`: `bench_engines.py` (pandas vs polars por etapa),
  `bench_automl.py` (vs scikit-learn y FLAML) y `make_report.py`, con
  resultados en `benchmarks/results/` (JSON + README generado).
- Sitio web: página "AutoML Engine", página de pipelines y extras de
  instalación actualizados; los 5 wrappers nuevos en la página de modelos
  supervisados.
- Notebook `how_use_synaptix.ipynb`: sección 13 que compara los 27
  modelos envueltos con scikit-learn (predicciones con los mismos
  parámetros, defaults, métricas y tiempos), AutoPipeline frente a un flujo
  manual y los resultados del benchmark.
- CI: job `automl-extras` que ejecuta `tests/automl` con las dependencias
  opcionales instaladas.
- 122 pruebas en `tests/automl/` (fuga de datos, reproducibilidad,
  errores de modelos, entradas, validación temporal y por grupos, tuning,
  ensembles, SHAP...).

### Cambiado

- `engine="auto"` calibrado con benchmarks: los archivos y DataFrames de
  pandas se leen con pandas; el perfilado usa polars desde 1 millón de
  celdas (2–4× más rápido, el doble de memoria pico; se desactiva con
  `memory_optimization=True`).
- `n_jobs` por defecto es 1 y se aplica en un solo nivel (antes
  `GridSearch` usaba `n_jobs=-1` siempre).
- `synaptix.pipeline.auto` y `synaptix.pipeline.grids` son fachadas de
  `synaptix.automl`; los imports anteriores siguen funcionando.

### Notas de compatibilidad

- `AutoPipeline().fit(df, target)`, `report()`, `predict`,
  `predict_proba`, `save`/`load` y los atributos `task_`, `ranking_`,
  `best_model_`, `best_name_`, `best_params_`, `results_` y `cleaner_` se
  conservan. `report()` sigue devolviendo `results_`.
- Las columnas de `ranking_` cambian (ahora incluyen estado, error, mejor
  y peor fold y tiempos).
- Al corregir la fuga, las métricas pueden ser ligeramente distintas (más
  honestas) que en 0.2.0.
- El ejemplo final del notebook ahora divide train/test antes de ajustar
  `DataCleaner` (antes limpiaba con todo el dataset).
