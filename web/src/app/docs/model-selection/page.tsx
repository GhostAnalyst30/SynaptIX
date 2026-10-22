import { Crosshair } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function ModelSelectionDocs() {
  return (
    <>
      <DocHeader
        title="Model Selection"
        description="Cross-validation, hyperparameter tuning and model comparison: k-fold cross_validate, GridSearch with best params, and compare_models — an AutoML-lite sweep over 7 classifiers or 7 regressors."
        icon={<Crosshair className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Compare Models</h2>
        <CodeBlock
          title="compare.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]

# Cross-validated comparison of 7 classifiers
tabla = sx.model_selection.compare_models(X, y, task="classification", cv=5)
print(tabla)   # DataFrame sorted by score: score, std, fit time

# The table exports to a ViewX dashboard (synaptix[viewx])
tabla.to_dashboard("comparacion.html")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Cross-Validation &amp; Tuning</h2>
        <CodeBlock
          title="tune.py"
          code={`import synaptix as sx

model = sx.supervised.RandomForestClassifier(random_state=42)

# k-fold cross-validation
cv = sx.model_selection.cross_validate(model, X, y, cv=10)
print("Score: %.3f ± %.3f" % (cv.mean_score, cv.std_score))

# GridSearch: retrained best model
search = sx.model_selection.GridSearch(
    model,
    {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    cv=5,
)
search.fit(X, y)
best = search.best_estimator_
print("Best score:", search.best_score_)
print("Best params:", search.best_params_)`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="cross_validate"
            signature="cross_validate(model, X, y, cv=5) -> CrossValidationResult"
            description="K-fold cross-validation of any SynaptixModel. The result is dict-compatible and exports to ViewX."
            returns="CrossValidationResult"
            example={`cv = sx.model_selection.cross_validate(model, X, y, cv=10)
cv.to_html("cv.html")`}
          />

          <MethodCard
            name="GridSearch"
            signature="GridSearch(model, param_grid, cv=5)"
            description="Grid search wrapper over scikit-learn's GridSearchCV; returns the retrained best model with best_params_ and best_score_."
            parameters={[
              { name: "model", type: "SynaptixModel", description: "Model to tune." },
              { name: "param_grid", type: "dict", description: "Hyperparameter grid." },
              { name: "cv", type: "int", description: "CV folds.", default: "5" },
            ]}
            example={`search = sx.model_selection.GridSearch(model, {"C": [0.1, 1, 10]})
search.fit(X, y)`}
          />

          <MethodCard
            name="compare_models"
            signature="compare_models(X, y, task='classification', cv=5) -> ModelComparisonResult"
            description="AutoML-lite: trains and cross-validates the default candidate set for the task and returns a sorted comparison table (score, std, time); individual model failures are tolerated."
            returns="ModelComparisonResult (DataFrame-compatible)"
            example={`tabla = sx.model_selection.compare_models(X, y, task="regression")
print(tabla.head())`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Default Grids</h2>
        <p className="text-sm text-muted leading-relaxed">
          <code className="code-inline">synaptix.pipeline.grids.DEFAULT_GRIDS</code> ships sensible
          grids per model — <code className="code-inline">AutoPipeline</code> uses them for its tuning
          step, and you can reuse them in your own <code className="code-inline">GridSearch</code> calls.
        </p>
      </section>
    </>
  );
}
