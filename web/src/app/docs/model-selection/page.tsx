import { Crosshair } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function ModelSelectionDocs() {
  return (
    <>
      <DocHeader
        title="Model Selection"
        description="Cross-validation, hyperparameter tuning and model comparison: k-fold cross_validate, GridSearch with best params, and compare_models — an AutoML-lite sweep over classifiers, regressors or clustering models with train and test metrics side by side."
        icon={<Crosshair className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Compare Models</h2>
        <CodeBlock
          title="compare.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]

# Cross-validated comparison of 7 classifiers (task inferred from y)
tabla = sx.model_selection.compare_models(X, y, cv=5)
print(tabla)   # sorted by test accuracy: train/test metrics, std, overfitting, fit time

# Choose the test size, the models and the metrics
tabla = sx.model_selection.compare_models(
    X, y,
    test_size=0.3,                       # 70/30 holdout, same split for every model
    models=["RandomForestClassifier", "LogisticRegression", "KNNClassifier"],
    metrics=["accuracy", "f1", "mcc"],
)
print(tabla[["modelo", "accuracy_train", "accuracy", "sobreajuste"]])

# Which names can I pass to models=?
print(sx.model_selection.list_models("classification"))

# The table exports to a ViewX dashboard (synaptix[viewx])
tabla.to_dashboard("comparacion.html")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Train vs. Test Metrics</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          For every metric the table has three columns:{" "}
          <code className="code-inline">&lt;metric&gt;_train</code> (training set),{" "}
          <code className="code-inline">&lt;metric&gt;</code> (test set) and{" "}
          <code className="code-inline">&lt;metric&gt;_std</code> (spread across splits). The{" "}
          <code className="code-inline">sobreajuste</code> column shows how much better the model does
          on train than on test for the sort metric — positive values mean overfitting. Without{" "}
          <code className="code-inline">test_size</code>, <code className="code-inline">cv</code> is the
          number of k-fold folds (default 5); with it, <code className="code-inline">cv</code> is the number of
          random train/test splits (default 1).
        </p>
        <CodeBlock
          title="regression_and_clustering.py"
          code={`# Regression: sort by test RMSE (error metrics sort ascending automatically)
sx.model_selection.compare_models(
    X_reg, y_reg, task="regression", test_size=0.25, cv=3,
    metrics=["rmse", "r2", "mape"], sort_by="rmse",
)

# Clustering: no y needed; pass y to also get ARI / NMI against real labels
sx.model_selection.compare_models(X, task="clustering", test_size=0.2, n_clusters=3)`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Cross-Validation &amp; Tuning</h2>
        <CodeBlock
          title="tune.py"
          code={`import synaptix as sx

model = sx.supervised.RandomForestClassifier(random_state=42)

# k-fold cross-validation
cv = sx.model_selection.cross_validate(model, X, y, cv=10, scoring="f1_weighted")
print("Score: %.3f ± %.3f" % (cv["media"], cv["desviacion"]))
print(cv["scores"])   # one score per fold

# GridSearch: fit() returns the best model, retrained on all data
search = sx.model_selection.GridSearch(
    model,
    {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    cv=5,
)
best = search.fit(X, y)
best.fit_evaluate(X, y, test_size=0.2)
print("Best score:", search.best_score_)
print("Best params:", search.best_params_)`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="cross_validate"
            signature="cross_validate(model, X, y, cv=5, scoring=None, verbose=True) -> CrossValidationResult"
            description="K-fold cross-validation of any SynaptixModel. The result is a dict with <code className='code-inline'>scores</code> (per fold), <code className='code-inline'>media</code> and <code className='code-inline'>desviacion</code>, and exports to ViewX. <code className='code-inline'>scoring</code> accepts any scikit-learn scorer name; defaults to accuracy / r2."
            returns="CrossValidationResult"
            example={`cv = sx.model_selection.cross_validate(model, X, y, cv=10)
cv.to_html("cv.html")`}
          />

          <MethodCard
            name="GridSearch"
            signature="GridSearch(model, param_grid, cv=5, scoring=None) · fit(X, y, verbose=True) -> SynaptixModel"
            description="Grid search over scikit-learn's GridSearchCV. <code className='code-inline'>fit()</code> returns the best model (a SynaptixModel retrained on all data) and sets <code className='code-inline'>best_params_</code> and <code className='code-inline'>best_score_</code>."
            parameters={[
              { name: "model", type: "SynaptixModel", description: "Model to tune." },
              { name: "param_grid", type: "dict", description: "Hyperparameter grid." },
              { name: "cv", type: "int", description: "CV folds.", default: "5" },
              { name: "scoring", type: "str | None", description: "scikit-learn scorer name.", default: "None" },
            ]}
            example={`search = sx.model_selection.GridSearch(model, {"C": [0.1, 1, 10]}, scoring="f1_macro")
best = search.fit(X, y)
print(search.best_params_, search.best_score_)`}
          />

          <MethodCard
            name="compare_models"
            signature="compare_models(X, y=None, task=None, cv=None, test_size=None, models=None, metrics=None, sort_by=None, ascending=None, return_train_score=True, n_clusters=3, random_state=42) -> ModelComparisonResult"
            description="AutoML-lite: trains the candidate set for the task (or only the models you pick) on identical splits and returns a sorted table with train and test metrics, overfitting gap and fit time. Individual model failures are tolerated."
            parameters={[
              { name: "y", type: "array-like", description: "Target. Omit it to compare clustering models.", default: "None" },
              { name: "task", type: "str", description: "'classification', 'regression' or 'clustering'. Inferred from y when omitted.", default: "None" },
              { name: "test_size", type: "float | int", description: "Test fraction in (0, 1) or number of test samples. Stratified in classification.", default: "None (k-fold)" },
              { name: "cv", type: "int", description: "k-fold folds without test_size; number of random train/test splits with it.", default: "5 / 1" },
              { name: "models", type: "list[str | model]", description: "Catalog names (see list_models) and/or configured model instances.", default: "None (scenario defaults)" },
              { name: "metrics", type: "list[str]", description: "Classification: accuracy, f1, precision, recall, roc_auc, balanced_accuracy, mcc, log_loss, *_macro. Regression: r2, mae, mse, rmse, medae, mape, explained_variance. Clustering: silhouette, calinski_harabasz, davies_bouldin, ari, nmi, homogeneity, completeness, v_measure.", default: "None (task defaults)" },
              { name: "sort_by", type: "str", description: "Test metric used to sort the table.", default: "first metric" },
              { name: "return_train_score", type: "bool", description: "Add <metric>_train columns and the sobreajuste (overfitting) column.", default: "True" },
              { name: "n_clusters", type: "int", description: "Clusters for the default clustering candidates.", default: "3" },
            ]}
            returns="ModelComparisonResult (DataFrame-compatible)"
            example={`tabla = sx.model_selection.compare_models(
    X, y, test_size=0.2, models=["SVR", "RandomForestRegressor"], sort_by="rmse"
)
print(tabla.head())`}
          />

          <MethodCard
            name="list_models"
            signature="list_models(task=None) -> list[str] | dict"
            description="Model names accepted by compare_models(models=[...]): the scenario defaults plus extras such as SVR, BayesianRidgeRegression, GaussianProcessClassifier or the Naive Bayes variants."
            returns="list[str] or dict[task, list[str]]"
            example={`sx.model_selection.list_models("regression")`}
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
