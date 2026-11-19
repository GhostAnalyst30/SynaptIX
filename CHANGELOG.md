# Registro de cambios

## Sin publicar — AutoPipeline: feature engineering, calidad y explicabilidad

### Corregido (metodología)

- **Calidad de datos estimada con el test**: `warning_strategy` analizaba e
  imputaba con el dataset completo antes del holdout, así que la mediana de
  imputación (y los cuantiles de outliers) contenían filas de prueba. Ahora
  solo las acciones que eliminan filas (duplicados) ocurren antes del split;
  la detección y las correcciones que aprenden parámetros se ajustan con las
  filas de entrenamiento y se aplican después a train y test.
- **`fe_handle_div_zero` no hacía nada**: el manejador dependía de un
  `except ZeroDivisionError` que NumPy nunca lanza (devuelve inf/nan).
  `"zero"` y `"error"` no tenían efecto. La política se aplica ahora en el
  evaluador AST: `nan`, `inf`, `zero` y `error` funcionan igual en `fit` y en
  `predict`.
- Una expresión encadenada que dividía por otra variable derivada se validaba
  con ceros de relleno y podía fallar en `fit` aunque los datos fueran válidos.

### Añadido

- `AutoPipeline.explain(df=None)`: importancias (permutación en test, nativas
  y coeficientes), diagnósticos de error, calibración (Brier + curva de
  fiabilidad en binario) y SHAP local opcional. No reentrena nada.
- `selected_features_`, `removed_features_`, `feature_selection_info_` y
  `feature_importance_` / `feature_importance_source_` para auditar la
  selección de variables del modelo final.
- `explain_shap()` sigue disponible; `model_importances_` y `coefficients_` se
  calculan aunque `explain=False`.

### Cambios de API

- El parámetro de constructor `explain` (bool) se guarda en
  `_explain_enabled` y se expone con `explain_enabled`; el nombre `explain`
  queda para el método `explain()`. `get_params()["explain"]` sigue devolviendo
  el booleano.

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
