import Link from "next/link";
import { Sparkles } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";
import { SYNAPTIX_VERSION } from "@/lib/version";

const presets = [
  { mode: "fast", cv: "3", nIter: "10", topK: "1", models: "Classification: logistic, decision_tree, random_forest, hist_gradient_boosting, naive_bayes. Regression: linear, ridge, decision_tree, random_forest, hist_gradient_boosting" },
  { mode: "balanced", cv: "5", nIter: "20", topK: "1", models: "fast + extra_trees, gradient_boosting, knn; plus svm (classification) or lasso, elastic_net, bayesian_ridge (regression)" },
  { mode: "exhaustive", cv: "5", nIter: "60", topK: "3", models: "balanced + gaussian_process (and svm for regression)" },
];

const guarantees = [
  { rule: "The test set and the CV folds are disjoint; the test set is split from the raw data before any fitting.", test: "test_holdout_and_cv_folds_are_disjoint" },
  { rule: "Preprocessing is refit inside every CV fold and search candidate and never sees test rows.", test: "test_preprocessing_never_sees_test_rows" },
  { rule: "Halving, Optuna, stacking and blending never see test rows.", test: "test_advanced_strategies_never_see_test_rows" },
  { rule: "Groups never cross train/validation; time series use the last rows as test and TimeSeriesSplit.", test: "test_group_validation / test_time_series_validation" },
  { rule: "Ensembles are compared with the same CV splits as the single models; blending members never see the meta holdout.", test: "test_ensembles_are_evaluated_with_same_cv" },
  { rule: "Failed models are reported with their error in the ranking and never win.", test: "test_failing_model_never_wins" },
  { rule: "The declarative Pipeline also splits first and cleans only with training data.", test: "test_declarative_pipeline_has_no_leakage" },
];

const engineRows = [
  { stage: "Load CSV (1M rows)", pandas: "0.56 s", polars: "0.71 s (polars + to_pandas)", choice: "pandas" },
  { stage: "Load Parquet (1M rows)", pandas: "0.28 s", polars: "0.29 s", choice: "pandas" },
  { stage: "Profile (50k rows)", pandas: "0.28 s", polars: "0.14 s", choice: "polars from 1M cells" },
  { stage: "Profile (1M rows)", pandas: "9.5 s · 439 MB", polars: "2.5 s · 1007 MB", choice: "polars (unless memory_optimization)" },
  { stage: "Full AutoML (100k rows)", pandas: "19.8 s", polars: "19.5 s", choice: "no material difference" },
];

const comparisonRows = [
  { data: "penguins · accuracy · 344", base: "0.9758 ± 0.0084 · 0.4 s", rs: "0.9855 ± 0.0145 · 13.0 s", fast: "0.9855 ± 0.0000 · 4.3 s", balanced: "0.9903 ± 0.0084 · 17.2 s", flaml: "0.9903 ± 0.0084 · 17.6 s" },
  { data: "classification · accuracy · 5k", base: "0.8317 ± 0.0132 · 1.1 s", rs: "0.8350 ± 0.0155 · 36.0 s", fast: "0.8380 ± 0.0070 · 18.7 s", balanced: "0.8353 ± 0.0072 · 67.7 s", flaml: "0.8290 ± 0.0087 · 68.9 s" },
  { data: "classification · accuracy · 50k", base: "0.8422 ± 0.0022 · 2.8 s", rs: "0.8425 ± 0.0030 · 72.3 s", fast: "0.8433 ± 0.0029 · 60.8 s", balanced: "—", flaml: "0.8411 ± 0.0018 · 62.4 s" },
  { data: "regression · R² · 5k", base: "0.9234 ± 0.0073 · 0.9 s", rs: "0.9243 ± 0.0060 · 30.9 s", fast: "0.9253 ± 0.0082 · 23.1 s", balanced: "0.9301 ± 0.0081 · 63.5 s", flaml: "0.9233 ± 0.0199 · 63.9 s" },
  { data: "regression · R² · 50k", base: "0.9389 ± 0.0029 · 2.7 s", rs: "0.9390 ± 0.0028 · 63.6 s", fast: "0.9397 ± 0.0029 · 136.2 s", balanced: "—", flaml: "0.9405 ± 0.0026 · 137.5 s" },
];

export default function AutoMLDocs() {
  return (
    <>
      <DocHeader
        title="AutoML Engine"
        description="synaptix.automl is the modular engine behind AutoPipeline: data adapters (pandas, polars, NumPy, files), data intelligence, leakage-free preprocessing, model selection, cross-validation, hyperparameter search, ensembles, explainability, profiling, experiment tracking and reports."
        icon={<Sparkles className="w-6 h-6" />}
        version={SYNAPTIX_VERSION}
      />

      <section className="mb-12">
        <h2 className="section-title">Installation</h2>
        <CodeBlock
          title="bash"
          code={`pip install synaptix              # core: pandas + scikit-learn only
pip install "synaptix[automl]"    # + polars, pyarrow, optuna, psutil
pip install "synaptix[explain]"   # + shap`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          Every extra is optional. If the dependency is missing,{" "}
          <code className="code-inline">tuning=&quot;optuna&quot;</code>,{" "}
          <code className="code-inline">engine=&quot;polars&quot;</code> and{" "}
          <code className="code-inline">explain_shap()</code> raise a clear{" "}
          <code className="code-inline">MissingDependencyError</code> naming the extra to install;{" "}
          <code className="code-inline">engine=&quot;auto&quot;</code> falls back to pandas without polars, and
          memory is not measured without psutil.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Usage by Level</h2>
        <CodeBlock
          title="basic.py"
          code={`import synaptix as sx

pipe = sx.AutoPipeline()
pipe.fit(df, target="species")
pipe.report()
pipe.predict(new_df)`}
        />
        <div className="mt-4" />
        <CodeBlock
          title="intermediate.py"
          code={`pipe = sx.AutoPipeline(
    mode="balanced",
    scoring="f1_weighted",
    metrics=["balanced_accuracy", "roc_auc_ovr"],
    exclude_models=["svm"],
    tuning="random", n_iter=30,
    feature_selection="kbest", feature_selection_k=0.5,
)
pipe.fit("data.parquet", target="churn", groups="customer_id")
print(pipe.profile_.warnings)     # nulls, imbalance, leakage suspects...
print(pipe.ranking_)              # mean ± std per metric, fit time, status
pipe.report(save="automl_report.html")`}
        />
        <div className="mt-4" />
        <CodeBlock
          title="advanced.py"
          code={`import polars as pl

pipe = sx.AutoPipeline(
    engine="polars",                         # lazy scan with column projection
    models=["hgb", "extra_trees", ("logistic", {"C": 0.5})],
    tuning="optuna", n_iter=60, tune_top_k=2, time_budget=300,
    param_distributions={"hist_gradient_boosting": {
        "model__learning_rate": ("float", 0.01, 0.3, True),
        "model__max_leaf_nodes": ("int", 15, 127, False),
    }},
    ensemble="blending", ensemble_top_k=3,
    n_jobs=-1, memory_optimization=True, cache=True,
)
pipe.fit(pl.scan_parquet("big.parquet"), target="y", time_column="date")
pipe.experiment_.to_json("runs/exp_001.json")   # params, seeds, versions, metrics, timings
shap = pipe.explain_shap(new_df)              # requires synaptix[explain]`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Architecture</h2>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Module</th>
                <th className="px-4 py-2 font-medium">Responsibility</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              {[
                ["data/adapters", "pandas / polars DataFrame / LazyFrame / NumPy / CSV / Parquet input; engine policy"],
                ["data/detection, data/intelligence", "task detection with reasons; profile: types, nulls, cardinality, imbalance, duplicates, leakage suspects"],
                ["features", "ColumnTransformer per feature type (numeric, categorical, boolean, datetime, text), fitted per fold"],
                ["evaluation", "metrics per task and CV strategies (k-fold, stratified, repeated, group, time series)"],
                ["selection", "model registry with aliases and capabilities, candidate evaluation and ranking"],
                ["tuning", "grid, random, successive halving and Optuna TPE with pruning, under a time budget"],
                ["ensemble", "voting, stacking and blending (meta-model on an internal holdout)"],
                ["explain", "permutation / native importances, coefficients, error diagnostics, SHAP"],
                ["profiling, experiments, reporting", "time and memory per stage, reproducible JSON/CSV records, HTML/JSON reports"],
              ].map(([module, desc]) => (
                <tr key={module} className="border-t border-border">
                  <td className="px-4 py-2 font-mono text-xs text-accent">{module}</td>
                  <td className="px-4 py-2">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Methodological Guarantees</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Each rule is enforced by an automated test in <code className="code-inline">tests/automl/</code>.
          The ranking reports validation scores (training data only); <code className="code-inline">results_</code>{" "}
          reports the test set, which is never used to choose models or hyperparameters.
        </p>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Guarantee</th>
                <th className="px-4 py-2 font-medium">Test</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              {guarantees.map((g) => (
                <tr key={g.test} className="border-t border-border">
                  <td className="px-4 py-2">{g.rule}</td>
                  <td className="px-4 py-2 font-mono text-xs">{g.test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Presets</h2>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">mode</th>
                <th className="px-4 py-2 font-medium">cv</th>
                <th className="px-4 py-2 font-medium">n_iter</th>
                <th className="px-4 py-2 font-medium">tune_top_k</th>
                <th className="px-4 py-2 font-medium">Candidates</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              {presets.map((p) => (
                <tr key={p.mode} className="border-t border-border">
                  <td className="px-4 py-2 font-mono text-xs text-accent">{p.mode}</td>
                  <td className="px-4 py-2">{p.cv}</td>
                  <td className="px-4 py-2">{p.nIter}</td>
                  <td className="px-4 py-2">{p.topK}</td>
                  <td className="px-4 py-2">{p.models}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted leading-relaxed mt-4">
          Explicit parameters (<code className="code-inline">cv</code>, <code className="code-inline">n_iter</code>,{" "}
          <code className="code-inline">models</code>...) always override the preset. Slow models
          are skipped from the default candidates above a row limit (Gaussian process 2,000, SVM 20,000,
          KNN 100,000) with a note in <code className="code-inline">notes_</code>; request them in{" "}
          <code className="code-inline">models=</code> to include them anyway.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">pandas vs polars Engine</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Median of 3 runs, each in a fresh process (Intel i5-7200U, 4 threads, Windows, Python 3.15,
          pandas 3.1, polars 2.1). Models always train with scikit-learn on pandas/NumPy, so polars only
          helps where the data is read or summarised. With <code className="code-inline">engine=&quot;auto&quot;</code>,
          polars is used only where it measured faster. Reproduce with{" "}
          <code className="code-inline">python benchmarks/bench_engines.py</code>.
        </p>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Stage</th>
                <th className="px-4 py-2 font-medium">pandas</th>
                <th className="px-4 py-2 font-medium">polars</th>
                <th className="px-4 py-2 font-medium">engine=&quot;auto&quot;</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              {engineRows.map((r) => (
                <tr key={r.stage} className="border-t border-border">
                  <td className="px-4 py-2">{r.stage}</td>
                  <td className="px-4 py-2 font-mono text-xs">{r.pandas}</td>
                  <td className="px-4 py-2 font-mono text-xs">{r.polars}</td>
                  <td className="px-4 py-2">{r.choice}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">AutoPipeline vs scikit-learn vs FLAML</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Same 80/20 split for every tool, 3 seeds, each run in a fresh process. Test metric as mean ± std
          across seeds; training time as the median. Baseline: ColumnTransformer + HistGradientBoosting with
          defaults; random search: the same plus 10 RandomizedSearchCV candidates (3 folds); FLAML gets the
          same time budget SynaptIX used. PyCaret and AutoGluon could not be installed on Python 3.15 and are
          reported as unavailable instead of estimated. Reproduce with{" "}
          <code className="code-inline">python benchmarks/bench_automl.py</code>.
        </p>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-3 py-2 font-medium">Dataset</th>
                <th className="px-3 py-2 font-medium">sklearn baseline</th>
                <th className="px-3 py-2 font-medium">sklearn random search</th>
                <th className="px-3 py-2 font-medium">synaptix fast</th>
                <th className="px-3 py-2 font-medium">synaptix balanced</th>
                <th className="px-3 py-2 font-medium">FLAML</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              {comparisonRows.map((r) => (
                <tr key={r.data} className="border-t border-border">
                  <td className="px-3 py-2">{r.data}</td>
                  <td className="px-3 py-2 font-mono text-xs">{r.base}</td>
                  <td className="px-3 py-2 font-mono text-xs">{r.rs}</td>
                  <td className="px-3 py-2 font-mono text-xs">{r.fast}</td>
                  <td className="px-3 py-2 font-mono text-xs">{r.balanced}</td>
                  <td className="px-3 py-2 font-mono text-xs">{r.flaml}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted leading-relaxed mt-4">
          SynaptIX has the best mean in 3 of 5 scenarios, ties with FLAML on penguins and FLAML wins regression
          at 50k rows — but the gap between the best and second tool (0.000–0.005) is of the same order as the
          seed-to-seed spread, so with 3 seeds the quality ranking is not conclusive. Comparing several models
          costs 12–50× the time of a single default model and more peak memory (up to 291 MB vs 77 MB on
          regression at 50k rows). AutoPipeline pays off when you don&apos;t know which model to use, when data
          problems must be detected, or when you need a traceable ranking, experiment record and report.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="AutoPipeline (full parameter list)"
            signature="sx.AutoPipeline(...) · sx.automl.AutoPipeline(...)"
            description="Same class in both namespaces. Basic usage and attributes are documented on the Pipelines page."
            parameters={[
              { name: "exclude_models", type: "list[str] | None", description: "Models removed from the candidates.", default: "None" },
              { name: "engine", type: "'auto' | 'pandas' | 'polars'", description: "Reading and profiling engine (see the benchmark table).", default: "'auto'" },
              { name: "cv_repeats", type: "int", description: "Repetitions for the repeated* strategies.", default: "3" },
              { name: "metrics", type: "list[str] | None", description: "Secondary metrics shown in the ranking.", default: "None" },
              { name: "n_iter / tune_top_k", type: "int | None", description: "Search iterations and number of top models tuned (preset by default).", default: "None" },
              { name: "time_budget", type: "float | None", description: "Seconds for selection and search; remaining candidates end as 'skipped'.", default: "None" },
              { name: "max_onehot_categories", type: "int", description: "Categories per feature before grouping the rest as 'infrequent'.", default: "20" },
              { name: "drop_constant / feature_selection / feature_selection_k", type: "bool / str / int | float", description: "Fold-wise constant removal and selection ('variance', 'kbest', 'model').", default: "False / None / None" },
              { name: "ensemble / ensemble_top_k", type: "str | None / int", description: "'voting', 'stacking' or 'blending' with the top models; kept only if CV improves. Stacking is skipped for time series (blending keeps chronological order).", default: "None / 3" },
              { name: "explain", type: "bool", description: "Permutation importances on the test set, native importances, coefficients and error diagnostics.", default: "True" },
              { name: "ignore_columns", type: "list[str] | None", description: "Columns excluded from the features (not read with polars lazy).", default: "None" },
              { name: "n_jobs / pre_dispatch", type: "int / str", description: "Single level of parallelism for CV and search; inner models use 1 thread.", default: "1 / '2*n_jobs'" },
              { name: "memory_optimization / cache", type: "bool", description: "Sparse one-hot for models that accept it (and pandas profiling); joblib.Memory cache of the preprocessing during search.", default: "False / False" },
            ]}
          />

          <MethodCard
            name="profile_data / detect_task_details"
            signature="sx.automl.profile_data(df, target, task) -> DataProfile · sx.automl.detect_task_details(y)"
            description="Standalone data intelligence: column types, nulls, cardinality, constant and ID-like columns, duplicates, missing targets, class imbalance and rare classes. During fit, AutoPipeline also adds leakage suspects computed on the training data only (suspicious names, near-perfect correlation or purity with the target). Task detection returns the task, a confidence and the reasons."
            example={`profile = sx.automl.profile_data(df, "species", "classification")
for w in profile.warnings:
    print(w.code, w.message)`}
          />

          <MethodCard
            name="ExperimentRecord"
            signature="pipe.experiment_.to_json(path) · to_csv(path) · ExperimentRecord.from_json(path)"
            description="Reproducible record of a run: configuration, seed, environment and package versions, data shape and feature types, CV strategy, ranking, tuning, ensemble, selected model, CV score, test metrics, timings per stage and warnings. <code className='code-inline'>to_csv</code> writes the ranking with the experiment id on each row."
          />

          <MethodCard
            name="list_models"
            signature="sx.automl.list_models(task=None) -> dict"
            description="Canonical model names per task. Aliases such as <code className='code-inline'>rf</code>, <code className='code-inline'>hgb</code>, <code className='code-inline'>gbm</code>, <code className='code-inline'>svc</code>/<code className='code-inline'>svr</code>, <code className='code-inline'>logreg</code> and <code className='code-inline'>ols</code> are also accepted in <code className='code-inline'>models=</code>."
          />
        </div>
      </section>

      <p className="text-sm text-muted leading-relaxed">
        The full guide (Spanish) is in{" "}
        <code className="code-inline">docs/AUTOML.md</code> of the repository; the basic API is on the{" "}
        <Link href="/docs/pipeline" className="text-accent hover:underline">Pipelines</Link> page.
      </p>
    </>
  );
}
