# Auditoría de `AutoPipeline` (SynaptIX)

Estado auditado: `synaptix/automl/pipeline.py` (1130 líneas) y sus módulos.
Sustituye a la auditoría anterior, que describía el estado previo a la
refactorización (antes de existir `feature_engineering.py` y
`data_quality.py`) y ya no coincidía con el código.

## 1. Tabla de auditoría

| Componente | Estado actual | Problema | Prioridad | Solución |
|---|---|---|---|---|
| `fit()` | ✅ Implementado | Orden: carga → FE → limpieza de filas → perfilado → holdout → calidad (train) → selección → tuning → ensemble → entrenamiento final → test | — | Mantener |
| `predict()` / `predict_proba()` | ✅ Implementado | Aplica `FeatureEngineeringTransformer` y valida columnas | — | Mantener |
| `report()` / `report_dict()` | ✅ Implementado | 12 secciones + `13_calidad_datos` si `warning_report=True` | — | Mantener |
| `associations` | ✅ Implementado (antes: ❌) | `FeatureEngineeringTransformer` (AST sin `eval()`) se ajusta en `fit` y se reaplica en `predict`; se serializa con `save()` | — | Mantener |
| `fe_handle_div_zero` | ✅ Implementado (antes: ❌ silencioso) | Dependía de `except ZeroDivisionError`, que NumPy nunca lanza: `"zero"` y `"error"` no tenían efecto | CRÍTICA | Política aplicada en el evaluador AST (`nan`/`inf`/`zero`/`error`) |
| `warning_strategy` | ✅ Implementado | Detectaba y estimaba parámetros con **todos** los datos, antes del holdout (la mediana de imputación veía el test) | CRÍTICA | Detección y correcciones estimadas solo con train; solo el borrado de duplicados es pre-split |
| `DataQualityAnalyzer` / `WarningSolver` / `DataQualityReport` | ✅ Implementado | 14 chequeos, estrategias `none`/`safe`/`aggressive` con confirmación para acciones destructivas | — | Mantener |
| `warning_report` | ✅ Implementado | Añade la sección 13 al reporte | — | Mantener |
| `extended`, `extended_max_features` | ❌ No existen en la API actual | La auditoría anterior los citaba; hoy no son parámetros públicos, así que no hay promesa incumplida | — | No reintroducirlos sin implementación real |
| `data_train` / `data_val` / `data_test` / `test_val` | ❌ No existen en la API actual | Igual que arriba: `fit()` solo acepta `data`, `target`, `groups`, `time_column` | — | Idem |
| `solve_warning` | ❌ No existe | Se sustituyó por `warning_strategy` (tri-estado) en lugar de un booleano ambiguo | — | Documentado; no se añade alias |
| `selected_features_` / `removed_features_` / `feature_selection_info_` | ✅ Añadido | Antes la selección era opaca | MEDIA | Atributos públicos con justificación |
| `feature_importance_` / `feature_importance_source_` | ✅ Añadido | Solo existía `feature_importances_` (permutación, requiere `explain=True` y holdout) | MEDIA | Importancia preferida + origen |
| `explain()` | ✅ Añadido | No existía API unificada (solo `explain_shap`) | MEDIA | Método `explain(df=None)` |
| `save()` / `load()` | ✅ Implementado | Pickle sin `_eval` ni `profiler_`; conserva FE, calidad, métricas y configuración | — | Validación de versiones: pendiente (ver §4) |
| Validación cruzada | ✅ Implementado | 8 estrategias + splitter propio; el preprocesamiento se reajusta por fold (test de espía) | — | Mantener |
| Selección de modelos | ✅ Implementado | Registro por modo, modelos que fallan quedan fuera con su error | — | Mantener |
| Optimización | ✅ Implementado | grid/random/halving/optuna/bayesian con `time_budget`, `n_iter` y `random_state` | — | Mantener |
| Ensembles | ✅ Implementado | voting/stacking/blending sobre los mismos splits; stacking se omite en temporal | — | Mantener |
| Explainability | ✅ Implementado | Permutación en test, nativas, coeficientes, SHAP opcional, diagnósticos, calibración | — | Curvas de aprendizaje: no implementadas |
| `cache` | ⚠️ Parcial | `joblib.Memory` en directorio temporal que se borra al final: acelera la búsqueda dentro de una ejecución, no persiste entre ejecuciones | BAJA | Invalidación por hash (`joblib` lo hace); persistencia no implementada |

## 2. Problemas priorizados y estado

1. **CRÍTICO — Fuga en calidad de datos.** Resuelto: el solver recibe
   `reference` (filas de entrenamiento) y estima medianas, cuantiles y
   frecuencias con ella. Verificado con test dedicado.
2. **CRÍTICO — `fe_handle_div_zero` sin efecto.** Resuelto en el evaluador AST.
3. **ALTO — Selección de features no consultable.** Resuelto con
   `selected_features_`, `removed_features_`, `feature_selection_info_`.
4. **MEDIO — Sin `explain()` unificado.** Resuelto.
5. **MEDIO — Chained expressions validadas con ceros de relleno.** Resuelto
   usando los valores calculados en `fit`.
6. **BAJO — `cache` no persiste entre ejecuciones.** Documentado, no cambiado.

## 3. Parámetros públicos: clasificación

| Parámetro | Estado | Efecto real |
|---|---|---|
| `task`, `models`, `exclude_models`, `mode`, `engine` | ✅ | Detecta/usa tarea, candidatos, preset, motor de lectura |
| `cv`, `cv_strategy`, `cv_repeats` | ✅ | Splitter, estrategia y repeticiones |
| `scoring`, `metrics` | ✅ | Métrica principal y secundarias |
| `tune`, `tuning`, `n_iter`, `tune_top_k`, `time_budget`, `param_grids`, `param_distributions` | ✅ | Búsqueda y su presupuesto |
| `test_size` | ✅ | Holdout (fracción/filas) o `None` |
| `impute_strategy`, `encode_method`, `scale_method`, `max_onehot_categories`, `drop_constant` | ✅ | Preprocesamiento por fold |
| `feature_selection`, `feature_selection_k` | ✅ | Selector por fold + atributos de consulta |
| `ensemble`, `ensemble_top_k` | ✅ | Ensemble evaluado con la misma CV |
| `explain` | ✅ | Cálculo automático de importancias/diagnósticos en `fit` |
| `ignore_columns`, `n_jobs`, `pre_dispatch`, `memory_optimization`, `cache` | ✅ | Columnas excluidas, paralelismo y memoria |
| `associations`, `fe_handle_div_zero`, `fe_handle_invalid` | ✅ | Feature engineering persistente |
| `warning_strategy`, `warning_report` | ✅ | Diagnóstico/corrección de calidad |
| `random_state`, `verbose` | ✅ | Reproducibilidad y trazas |

`fit()` expone únicamente `data`, `target`, `groups`, `time_column`.
No hay parámetros declarados sin efecto.

## 4. Limitaciones pendientes

- **Validación de versiones al cargar**: `load()` no compara las versiones de
  scikit-learn/pandas del pickle con las instaladas. Un pickle de otra versión
  puede fallar de forma opaca.
- **Curvas de aprendizaje y SHAP global**: no se calculan (coste alto); SHAP
  local es opcional vía `shap`.
- **`cache` persistente**: el directorio temporal se elimina al terminar
  `fit`; no hay caché entre ejecuciones.
- **Benchmarks**: no se han medido tiempos antes/después en esta ronda; no se
  afirman mejoras de rendimiento.
- **Polars**: se usa para lectura y perfilado; el entrenamiento siempre pasa
  por pandas/NumPy (scikit-learn).

## 5. Verificación

- `python -m pytest tests` — resultados reales en la entrega.
- Tests nuevos en `tests/automl/test_quality_and_explain.py`: políticas de
  división por cero, expresiones encadenadas, atributos de selección,
  `explain()` (global, calibración, permutación), estrategia `none` sin
  modificaciones, estadísticos de calidad desde train y comprobación de que el
  solver nunca recibe las filas del holdout.
