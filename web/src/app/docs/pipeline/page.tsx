import Link from "next/link";
import { Workflow } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";
import { SYNAPTIX_VERSION } from "@/lib/version";

export default function PipelineDocs() {
  return (
    <>
      <DocHeader
        title="Pipelines"
        description="Automatic machine learning end-to-end: AutoPipeline detects the task, compares models with leakage-free cross-validation, tunes the best ones and evaluates them on a held-out test set. Pipeline offers the same steps in declarative form."
        icon={<Workflow className="w-6 h-6" />}
        version={SYNAPTIX_VERSION}
      />

      <section className="mb-12">
        <h2 className="section-title">AutoPipeline — ML in One Line</h2>
        <CodeBlock
          title="auto.py"
          code={`import synaptix as sx

df = sx.load_dataset("penguins")

auto = sx.AutoPipeline()          # task detected, mode="balanced"
auto.fit(df, target="species")    # split -> profile -> compare (CV) -> tune -> evaluate

print("Best model:", auto.best_name_, auto.best_params_)
print(auto.ranking_)              # cross-validation ranking (training data only)
print(auto.results_)              # metrics on the held-out test set
auto.report(plot=True)            # summary + evaluation plots
auto.report(save="report.html")   # self-contained HTML report

# Predict on raw data — the fitted preprocessing is applied automatically
auto.predict(df.head(5))
auto.predict_proba(df.head(5))

auto.save("model.pkl")
auto2 = sx.AutoPipeline.load("model.pkl")`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          Since v0.3.0 AutoPipeline runs on the modular AutoML engine (
          <code className="code-inline">synaptix.automl</code>). The test set is split from the raw data
          first, and all preprocessing (imputation, encoding, scaling, feature selection) lives inside a
          scikit-learn <code className="code-inline">Pipeline</code> that is refit in every cross-validation
          fold and every search candidate, so the test rows never influence cleaning, model selection or
          tuning. See the <Link href="/docs/automl" className="text-accent hover:underline">AutoML Engine</Link>{" "}
          page for validation strategies, tuning, ensembles, polars support and benchmarks.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Configuring AutoPipeline</h2>
        <CodeBlock
          title="auto_config.py"
          code={`auto = sx.AutoPipeline(
    mode="fast",                       # "fast" | "balanced" | "exhaustive"
    models=["random_forest", "hgb", ("svm", {"C": 10})],
    scoring="f1_weighted",
    tuning="optuna",                   # "auto" | "grid" | "random" | "halving" | "optuna" | "bayesian"
    n_iter=30,
    time_budget=120,                   # seconds for selection + search
    ensemble="stacking",               # "voting" | "stacking" | "blending" (kept only if CV improves)
    n_jobs=-1,
)
auto.fit(df, target="species")

# Groups never cross train/validation; time series use the last rows as test
auto.fit(df, target="y", groups="patient_id")
auto.fit(df, target="sales", time_column="date")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Declarative Pipeline</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          List the steps in order; each step is a string (sensible defaults) or a tuple with its own
          parameters. The test set is split before any step, cleaning is fitted on the training data (and
          inside each fold during selection and tuning), and outliers are removed only from the training rows.
        </p>
        <CodeBlock
          title="declarative.py"
          code={`import synaptix as sx

df = sx.load_dataset("titanic").drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

pipe = sx.Pipeline([
    ("clean", {"impute_strategy": "mean"}),
    ("outliers", {"columns": ["Age", "Fare"], "method": "iqr", "threshold": 1.5}),
    ("model", "random_forest", {"n_estimators": 300}),
    "tune",                                   # default grid for the model
    ("evaluate", {"plot": True}),
], test_size=0.2, cv=5)

pipe.fit(df, target="Survived")
print(pipe.model_)      # trained SynaptixModel (receives cleaned data)
print(pipe.results_)    # metrics from the evaluate step
pipe.predict(df.head()) # raw data: cleaning + model via pipe.pipeline_`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="AutoPipeline"
            signature="AutoPipeline(task=None, models=None, exclude_models=None, mode='balanced', engine='auto', cv=None, cv_strategy='auto', scoring=None, metrics=None, tune=True, tuning='auto', n_iter=None, tune_top_k=None, time_budget=None, param_grids=None, param_distributions=None, test_size=0.2, impute_strategy='median', encode_method='onehot', scale_method='standard', ensemble=None, explain=True, n_jobs=1, random_state=42, verbose=True, ...)"
            description="End-to-end AutoML. Detects the task, profiles the data, compares the candidates with leakage-free cross-validation, tunes the top models, optionally builds an ensemble and evaluates the final pipeline on the held-out test set. The full parameter list is on the AutoML Engine page."
            parameters={[
              { name: "task", type: "'classification' | 'regression' | None", description: "Task type; detected automatically (with reasons in task_detection_) when None.", default: "None" },
              { name: "models", type: "list | None", description: "Names or aliases ('random_forest', 'rf', 'hgb'...), (name, params) tuples, scikit-learn estimators or SynaptIX models. Defaults to the candidates of the mode.", default: "None" },
              { name: "mode", type: "'fast' | 'balanced' | 'exhaustive'", description: "Preset of candidates, folds (3/5/5), search iterations (10/20/60) and number of tuned models (1/1/3).", default: "'balanced'" },
              { name: "cv / cv_strategy", type: "int | splitter / str", description: "Folds and strategy: 'auto', 'kfold', 'stratified', 'repeated', 'group', 'stratified_group', 'timeseries'.", default: "None / 'auto'" },
              { name: "scoring", type: "str | callable", description: "Main metric (accuracy / r2 by default, with a warning on imbalanced data).", default: "None" },
              { name: "tuning", type: "str | None", description: "'auto', 'grid', 'random', 'halving', 'optuna' or 'bayesian' (Optuna TPE).", default: "'auto'" },
              { name: "test_size", type: "float | int | None", description: "Held-out fraction (or rows) for the final evaluation; None for CV only.", default: "0.2" },
              { name: "impute_strategy / encode_method / scale_method", type: "str", description: "Preprocessing applied inside every fold.", default: "'median' / 'onehot' / 'standard'" },
              { name: "param_grids / param_distributions", type: "dict | None", description: "Custom search spaces per model name; override the defaults.", default: "None" },
            ]}
            example={`auto = sx.AutoPipeline(task="regression", models=["ridge", "random_forest"], tune=False)
auto.fit(df, target="price")`}
          />

          <MethodCard
            name="AutoPipeline methods & attributes"
            signature="fit(data, target, groups=None, time_column=None) · predict(df) · predict_proba(df) · report(plot=True, save=None) · report_dict() · explain_shap(df) · save(path) · AutoPipeline.load(path)"
            description="After <code className='code-inline'>fit</code>: <code className='code-inline'>task_</code>, <code className='code-inline'>profile_</code> (data warnings), <code className='code-inline'>ranking_</code> (CV comparison on train), <code className='code-inline'>best_name_</code>, <code className='code-inline'>best_params_</code>, <code className='code-inline'>cv_score_</code>, <code className='code-inline'>pipeline_</code> (preprocessing + model, accepts raw data), <code className='code-inline'>best_model_</code>, <code className='code-inline'>cleaner_</code>, <code className='code-inline'>results_</code> (test metrics), <code className='code-inline'>feature_importances_</code>, <code className='code-inline'>diagnostics_</code>, <code className='code-inline'>timings_</code>, <code className='code-inline'>experiment_</code> and <code className='code-inline'>warnings_</code>. <code className='code-inline'>data</code> may be a pandas or polars DataFrame, a polars LazyFrame, a NumPy array or a CSV/Parquet path."
            returns="report() -> dict"
          />

          <MethodCard
            name="Pipeline"
            signature="Pipeline(steps, task=None, cv=5, test_size=0.2, random_state=42, verbose=True)"
            description="Declarative pipeline. Valid steps: <code className='code-inline'>clean</code>, <code className='code-inline'>outliers</code> (training rows only), <code className='code-inline'>model</code> (automatic selection, a registry name, and/or params), <code className='code-inline'>tune</code> (optional <code className='code-inline'>{'{'}&quot;param_grid&quot;: ...{'}'}</code>) and <code className='code-inline'>evaluate</code>. Methods: <code className='code-inline'>fit(df, target)</code>, <code className='code-inline'>predict(df)</code>. Attributes: <code className='code-inline'>task_</code>, <code className='code-inline'>cleaner_</code>, <code className='code-inline'>model_</code>, <code className='code-inline'>pipeline_</code>, <code className='code-inline'>model_name_</code>, <code className='code-inline'>best_params_</code> (after tune), <code className='code-inline'>ranking_</code> (only with automatic model step), <code className='code-inline'>results_</code>."
            example={`pipe = sx.Pipeline(["clean", "model", "evaluate"])   # model chosen automatically
pipe.fit(df, target="species")
print(pipe.ranking_)`}
          />

          <MethodCard
            name="detect_task"
            signature="detect_task(y) -> str"
            description="Infer the task type from the target column: classification for categorical, boolean or few distinct integer values; regression for continuous ones. <code className='code-inline'>sx.automl.detect_task_details(y)</code> also returns the confidence and the reasons."
            returns="str"
            example={`task = sx.pipeline.detect_task(df["price"])   # 'regression'`}
          />

          <MethodCard
            name="MODEL_REGISTRY / DEFAULT_CANDIDATES / DEFAULT_GRIDS"
            signature="sx.pipeline.MODEL_REGISTRY[task][name]"
            description="Registry of model names usable in <code className='code-inline'>models=</code> and the <code className='code-inline'>model</code> step. Classification: logistic, decision_tree, random_forest, extra_trees, gradient_boosting, hist_gradient_boosting, svm, knn, naive_bayes, gaussian_process. Regression: linear, ridge, lasso, elastic_net, bayesian_ridge, decision_tree, random_forest, extra_trees, gradient_boosting, hist_gradient_boosting, svm, knn, gaussian_process. <code className='code-inline'>DEFAULT_CANDIDATES</code> are the candidates of the balanced mode."
            example={`from synaptix.pipeline import DEFAULT_CANDIDATES, DEFAULT_GRIDS
print(DEFAULT_CANDIDATES["classification"])
print(DEFAULT_GRIDS["random_forest"])`}
          />
        </div>
      </section>
    </>
  );
}
