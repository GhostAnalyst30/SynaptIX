import { Workflow } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function PipelineDocs() {
  return (
    <>
      <DocHeader
        title="Pipelines"
        description="Automatic machine learning end-to-end: AutoPipeline detects the task, cleans the data, compares models, tunes the best one and evaluates it. Pipeline offers the same steps in declarative form."
        icon={<Workflow className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">AutoPipeline — ML in One Line</h2>
        <CodeBlock
          title="auto.py"
          code={`import synaptix as sx

df = sx.load_dataset("penguins")

auto = sx.AutoPipeline(task="classification", test_size=0.2)
auto.fit(df, target="species")   # detect -> clean -> compare -> tune -> evaluate

print("Best model:", auto.best_name_, auto.best_params_)
print(auto.ranking_)             # cross-validation ranking of every candidate
print(auto.results_)             # metrics on the held-out test set
auto.report(plot=True)           # summary + evaluation plots

# Predict on raw data — the fitted cleaning is applied automatically
auto.predict(df.head(5))
auto.predict_proba(df.head(5))

auto.save("model.pkl")
auto2 = sx.AutoPipeline.load("model.pkl")`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          If <code className="code-inline">task</code> is omitted, it is detected automatically from the
          target column with <code className="code-inline">detect_task</code>.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Declarative Pipeline</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          List the steps in order; each step is a string (sensible defaults) or a tuple with its own parameters:
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
print(pipe.model_)      # trained SynaptixModel
print(pipe.results_)    # metrics from the evaluate step
pipe.predict(df.head())`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="AutoPipeline"
            signature="AutoPipeline(task=None, models=None, cv=5, tune=True, test_size=0.2, impute_strategy='median', encode_method='onehot', scale_method='standard', param_grids=None, random_state=42, verbose=True)"
            description="End-to-end automatic ML. Cleans with DataCleaner, compares the candidate models with cross-validation, tunes the best one with GridSearch and evaluates it on a held-out test set."
            parameters={[
              { name: "task", type: "'classification' | 'regression' | None", description: "Task type; detected automatically when None.", default: "None" },
              { name: "models", type: "list[str] | None", description: "Candidate names from MODEL_REGISTRY (e.g. 'random_forest', 'svm'); defaults to DEFAULT_CANDIDATES.", default: "None" },
              { name: "cv", type: "int", description: "Cross-validation folds for ranking and tuning.", default: "5" },
              { name: "tune", type: "bool", description: "Tune the best model with its parameter grid.", default: "True" },
              { name: "test_size", type: "float", description: "Fraction held out for the final evaluation.", default: "0.2" },
              { name: "impute_strategy / encode_method / scale_method", type: "str", description: "Passed to DataCleaner.", default: "'median' / 'onehot' / 'standard'" },
              { name: "param_grids", type: "dict | None", description: "Custom grids per model name; overrides DEFAULT_GRIDS.", default: "None" },
            ]}
            example={`auto = sx.AutoPipeline(task="regression", models=["ridge", "random_forest"], tune=False)
auto.fit(df, target="price")`}
          />

          <MethodCard
            name="AutoPipeline methods & attributes"
            signature="fit(df, target) · predict(df) · predict_proba(df) · report(plot=True) · save(path) · AutoPipeline.load(path)"
            description="After <code className='code-inline'>fit</code>: <code className='code-inline'>task_</code>, <code className='code-inline'>target_</code>, <code className='code-inline'>cleaner_</code>, <code className='code-inline'>ranking_</code> (comparison table), <code className='code-inline'>best_model_</code>, <code className='code-inline'>best_name_</code>, <code className='code-inline'>best_params_</code> and <code className='code-inline'>results_</code> (test metrics). <code className='code-inline'>report()</code> prints the summary and returns the test metrics."
            returns="report() -> dict"
          />

          <MethodCard
            name="Pipeline"
            signature="Pipeline(steps, task=None, cv=5, test_size=0.2, random_state=42, verbose=True)"
            description="Declarative pipeline. Valid steps: <code className='code-inline'>clean</code>, <code className='code-inline'>outliers</code>, <code className='code-inline'>model</code> (automatic selection, a registry name, and/or params), <code className='code-inline'>tune</code> (optional <code className='code-inline'>{'{'}&quot;param_grid&quot;: ...{'}'}</code>) and <code className='code-inline'>evaluate</code>. Methods: <code className='code-inline'>fit(df, target)</code>, <code className='code-inline'>predict(df)</code>. Attributes: <code className='code-inline'>task_</code>, <code className='code-inline'>cleaner_</code>, <code className='code-inline'>model_</code>, <code className='code-inline'>model_name_</code>, <code className='code-inline'>best_params_</code> (after tune), <code className='code-inline'>ranking_</code> (only with automatic model step), <code className='code-inline'>results_</code>. Without <code className='code-inline'>columns</code>, the outliers step checks every numeric column — including one-hot dummies created by clean — so pass the columns explicitly when cleaning categoricals."
            example={`pipe = sx.Pipeline(["clean", "model", "evaluate"])   # model chosen automatically
pipe.fit(df, target="species")
print(pipe.ranking_)`}
          />

          <MethodCard
            name="detect_task"
            signature="detect_task(y) -> str"
            description="Infer the task type from the target column: classification for categorical/object or few distinct integer values, regression for continuous ones."
            returns="str"
            example={`task = sx.pipeline.detect_task(df["price"])   # 'regression'`}
          />

          <MethodCard
            name="MODEL_REGISTRY / DEFAULT_CANDIDATES / DEFAULT_GRIDS"
            signature="sx.pipeline.MODEL_REGISTRY[task][name]"
            description="Registry of model names usable in <code className='code-inline'>models=</code> and the <code className='code-inline'>model</code> step. Classification: logistic, decision_tree, random_forest, gradient_boosting, svm, knn, naive_bayes, gaussian_process. Regression: linear, ridge, lasso, bayesian_ridge, decision_tree, random_forest, gradient_boosting, svm, knn, gaussian_process."
            example={`from synaptix.pipeline import DEFAULT_CANDIDATES, DEFAULT_GRIDS
print(DEFAULT_CANDIDATES["classification"])
print(DEFAULT_GRIDS["random_forest"])`}
          />
        </div>
      </section>
    </>
  );
}
