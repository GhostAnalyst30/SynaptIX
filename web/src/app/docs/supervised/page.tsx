import { Target } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

const CLASSIFIERS = [
  "LogisticRegression",
  "DecisionTreeClassifier",
  "RandomForestClassifier",
  "GradientBoostingClassifier",
  "SVMClassifier",
  "KNNClassifier",
  "NaiveBayes",
];

const REGRESSORS = [
  "LinearRegression",
  "RidgeRegression",
  "LassoRegression",
  "DecisionTreeRegressor",
  "RandomForestRegressor",
  "GradientBoostingRegressor",
  "SVR",
  "KNNRegressor",
];

export default function SupervisedDocs() {
  return (
    <>
      <DocHeader
        title="Supervised Models"
        description="8 regressors and 7 classifiers behind one unified API: fit, predict, predict_proba, score, evaluate, summary, feature_importances, save/load and diagnostics — all shared through SynaptixModel."
        icon={<Target className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Classification</h2>
        <CodeBlock
          title="classify.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]
X_tr, X_te, y_tr, y_te = sx.preprocessing.train_test_split(X, y, random_state=42)

model = sx.supervised.RandomForestClassifier(n_estimators=100)
model.fit(X_tr, y_tr)

print(model.evaluate(X_te, y_te))          # metrics + report
print(model.predict_proba(X_te.head(3)))    # class probabilities
print(model.feature_importances())          # per-column importance`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Regression</h2>
        <CodeBlock
          title="regress.py"
          code={`import synaptix as sx
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "x1": rng.uniform(0, 10, 300),
    "x2": rng.uniform(0, 10, 300),
})
df["y"] = 2.5 * df["x1"] - 1.2 * df["x2"] + rng.normal(0, 1, 300)

model = sx.supervised.GradientBoostingRegressor()
model.fit(df[["x1", "x2"]], df["y"])
result = model.evaluate(df[["x1", "x2"]], df["y"])
print(result)   # MAE, MSE, RMSE, R2, MAPE`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Available Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h3 className="font-syne text-sm font-semibold text-white mb-2">Classifiers (7)</h3>
            <ul className="space-y-1">
              {CLASSIFIERS.map((m) => (
                <li key={m} className="font-mono text-xs text-accent">{m}</li>
              ))}
            </ul>
          </div>
          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h3 className="font-syne text-sm font-semibold text-white mb-2">Regressors (8)</h3>
            <ul className="space-y-1">
              {REGRESSORS.map((m) => (
                <li key={m} className="font-mono text-xs text-accent">{m}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">The Unified Model API</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Every model — supervised, unsupervised or bayesian — shares the same interface:
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Method</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Purpose</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                ["fit(X, y)", "Train the model (DataFrames, arrays, lists, polars)"],
                ["predict(X)", "Predict on new data"],
                ["predict_proba(X)", "Class probabilities (classifiers)"],
                ["score(X, y)", "Default metric score"],
                ["evaluate(X, y, plot=False)", "Full metrics (optionally with plots)"],
                ["summary()", "Model configuration summary"],
                ["feature_importances()", "Per-column importance (when available)"],
                ["diagnostics(X, y)", "Statistical tests via StatsLibX (extra stats)"],
                ["save(path) / load(path)", "Pickle persistence"],
              ].map(([m, p]) => (
                <tr key={m} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 pr-4 font-mono text-xs text-accent">{m}</td>
                  <td className="py-2.5 text-muted">{p}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted leading-relaxed mt-4">
          Results from <code className="code-inline">evaluate()</code>,{" "}
          <code className="code-inline">cross_validate()</code> and{" "}
          <code className="code-inline">compare_models()</code> are plain dicts/DataFrames that also
          export to ViewX (<code className="code-inline">.to_html()</code>,{" "}
          <code className="code-inline">.to_report()</code>).
        </p>
      </section>
    </>
  );
}
