# AutoPipeline Extended Improvements Plan

## Goal
Add constructor options `extended=True` and `solve_warning=True`, support split datasets in `fit`, and support association/feature-engineering dictionaries with `.data` access.

## Scope
- File: `synaptix/automl/pipeline.py` (`AutoPipeline`)
- Tests: `tests/automl/test_pipeline.py`
- Keep backward compatibility with existing `fit(data, target, ...)`.

## Design Decisions
1. `extended`: when True, train AutoPipeline on feature subsets of sizes 1..`extended_max_features` (default 3, configurable), rank by primary metric, store `extended_results_`, set `best_model_`/`pipeline_` to best subset, and expose `feature_subset_`.
2. `solve_warning`: when True, profile data and automatically apply fixes:
   - drop uninformative/constant/all-missing columns
   - switch scoring to balanced metric for class imbalance
   - record actions in `warnings_`/`notes_`
3. `fit` signature:
   `fit(data=None, target=None, groups=None, time_column=None, test_val=None, data_train=None, data_val=None, data_test=None, associations=None, extended_max_features=None)`
   - If `data` provided: existing behavior, using `test_size` and optional `test_val`.
   - If `data_train` provided: use explicit splits; if `data_val` is None, train/test only.
   - `test_val` only used in single-data mode to create train/val/test.
4. Associations: accept dict `{"new_column": "column1 * column2"}`, evaluate safely with pandas `eval`, apply to all provided data before split, store resulting frame in `self.data`.
5. Add `best_model` property aliasing `best_model_`.

## Implementation Steps
1. Update `__init__` params and `_PUBLIC_PARAMS`.
2. Add `extended`, `solve_warning`, `associations` handling.
3. Refactor `fit` to normalize inputs into train/val/test.
4. Add feature-subset exploration loop for extended mode.
5. Add warning solver pass before model selection.
6. Apply associations and store `.data`.
7. Add tests for extended, solve_warning, split fit, associations.

## Risks
- Extended mode combinatorial cost; default `extended_max_features=3`.
- Safe eval of associations; restrict to arithmetic/column references.
- Associations, data loading, and split handling must work for both pandas and polars engines.
- Convert polars inputs to pandas before association evaluation and preserve original engine behavior for profiling."""


## Validation
- Run existing pipeline tests.
- Add tests for new features.
- Lint/typecheck if available.
