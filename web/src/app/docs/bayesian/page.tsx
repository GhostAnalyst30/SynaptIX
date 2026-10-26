import { Sigma } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function BayesianDocs() {
  return (
    <>
      <DocHeader
        title="Bayesian Models"
        description="Uncertainty-aware models: Bayesian ridge, ARD, Gaussian processes, Naive Bayes variants and full PyMC models with prediction intervals — new in v0.1.7."
        icon={<Sigma className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Models with Prediction Intervals</h2>
        <CodeBlock
          title="intervals.py"
          code={`import numpy as np
import pandas as pd
import synaptix as sx

rng = np.random.default_rng(1)
df = pd.DataFrame({"x": np.arange(60, dtype=float)})
df["y"] = 0.8 * df["x"] + 5 + rng.normal(0, 4, 60)

model = sx.supervised.BayesianRidgeRegression()
model.fit_evaluate(df[["x"]], df["y"], test_size=0.25)   # train/test metrics

mean, low, high = model.predict_interval(df[["x"]], std=2.0)   # ~95% band
for m, lo, hi in zip(mean[:3], low[:3], high[:3]):
    print(f"  {m:.2f}  [{lo:.2f}, {hi:.2f}]")`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          The bayesian regressors (<code className="code-inline">BayesianRidgeRegression</code>,{" "}
          <code className="code-inline">ARDRegression</code>,{" "}
          <code className="code-inline">GaussianProcessRegressor</code>) expose{" "}
          <code className="code-inline">predict_interval(X, std=2.0)</code>, returning{" "}
          <code className="code-inline">(mean, mean - std·σ, mean + std·σ)</code>. They also share the full
          supervised API: <code className="code-inline">fit</code>, <code className="code-inline">predict</code>,{" "}
          <code className="code-inline">evaluate</code>, <code className="code-inline">fit_evaluate</code>,{" "}
          <code className="code-inline">diagnostics</code>, <code className="code-inline">save</code>/<code className="code-inline">load</code>.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Available Models (9)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Model</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Signature / notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                ["BayesianRidgeRegression", "(max_iter=300, **kwargs) — predict_interval(X, std=2.0)"],
                ["ARDRegression", "(max_iter=300, **kwargs) — sparse coefficients; predict_interval(X, std=2.0)"],
                ["GaussianProcessRegressor", "(kernel=None, normalize_y=True, random_state=42) — predict_interval(X, std=2.0)"],
                ["GaussianProcessClassifier", "(kernel=None, random_state=42) — probabilistic classification"],
                ["MultinomialNB", "(alpha=1.0) — counts / text features"],
                ["BernoulliNB", "(alpha=1.0, binarize=0.0) — binary features"],
                ["ComplementNB", "(alpha=1.0) — imbalanced text data"],
                ["PyMCLinearRegression", "(draws=1000, tune=1000, chains=2, random_seed=42) — requires synaptix[bayes]"],
                ["PyMCLogisticRegression", "(draws=1000, tune=1000, chains=2, random_seed=42) — binary target, requires synaptix[bayes]"],
              ].map(([m, n]) => (
                <tr key={m} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 pr-4 font-mono text-xs text-accent">{m}</td>
                  <td className="py-2.5 text-muted">{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">PyMC Models</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The PyMC models run full MCMC sampling and require the bayes extra:
        </p>
        <CodeBlock
          title="bash + python"
          code={`pip install 'synaptix[bayes]'

import synaptix as sx

model = sx.supervised.PyMCLinearRegression(draws=1000, tune=1000, chains=2)
model.fit(X_train, y_train, progressbar=False)
print(model.summary())                                # ArviZ posterior summary (DataFrame)
model.plot_posterior()
mean, low, high = model.predict_interval(X_test, hdi=0.94)
model.evaluate(X_test, y_test, metrics=["MAE", "R2"])

clf = sx.supervised.PyMCLogisticRegression(draws=500)
clf.fit_evaluate(X, y_binary, test_size=0.2)
proba = clf.predict_proba(X_test)`}
        />
        <div className="method-list mt-6">
          <MethodCard
            name="PyMC methods"
            signature="fit(X, y, progressbar=False) · predict(X) · predict_interval(X, hdi=0.94) · predict_proba(X) · summary() · plot_posterior() · evaluate(X, y, verbose=True, metrics=None) · fit_evaluate(X, y, test_size=0.2, metrics=None, ...)"
            description="<code className='code-inline'>predict_interval</code> (linear model only) returns the posterior mean and the HDI bounds; <code className='code-inline'>predict_proba</code> is available on the logistic model."
          />
        </div>
      </section>
    </>
  );
}
