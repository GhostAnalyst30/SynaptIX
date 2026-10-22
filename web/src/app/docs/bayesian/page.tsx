import { Sigma } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

export default function BayesianDocs() {
  return (
    <>
      <DocHeader
        title="Bayesian Models"
        description="Uncertainty-aware models: Bayesian ridge, ARD, Gaussian processes, Naive Bayes variants and full PyMC models with prediction intervals — new in v0.1.7."
        icon={<Sigma className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Models with Prediction Intervals</h2>
        <CodeBlock
          title="intervals.py"
          code={`import synaptix as sx
import pandas as pd

# Data with noise
df = pd.DataFrame({"x": range(60)}, dtype=float)
df["y"] = 0.8 * df["x"] + 5 + pd.Series(
    __import__("numpy").random.default_rng(1).normal(0, 4, 60)
)

model = sx.supervised.BayesianRidgeRegression()
model.fit(df[["x"]], df["y"])

mean, low, high = model.predict_interval(df[["x"]], alpha=0.05)
print("First 3 predictions with 95% intervals:")
for m, lo, hi in zip(mean[:3], low[:3], high[:3]):
    print(f"  {m:.2f}  [{lo:.2f}, {hi:.2f}]")`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          The bayesian regressors (<code className="code-inline">BayesianRidgeRegression</code>,{" "}
          <code className="code-inline">ARDRegression</code>,{" "}
          <code className="code-inline">GaussianProcessRegressor</code>) expose{" "}
          <code className="code-inline">predict_interval(mean, low, high)</code>.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Available Models (9)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Model</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                ["BayesianRidgeRegression", "Regularized linear regression with posterior estimates; predict_interval available"],
                ["ARDRegression", "Automatic Relevance Determination — sparse coefficients"],
                ["GaussianProcessRegressor", "Non-parametric with calibrated uncertainty"],
                ["GaussianProcessClassifier", "Probabilistic classification"],
                ["MultinomialNB / BernoulliNB / ComplementNB", "Naive Bayes family for counts, binary and imbalanced text data"],
                ["PyMCLinearRegression", "Full Bayesian inference via MCMC (requires synaptix[bayes])"],
                ["PyMCLogisticRegression", "Bayesian logistic regression via PyMC (requires synaptix[bayes])"],
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

model = sx.supervised.PyMCLinearRegression()
model.fit(X_train, y_train, draws=1000)
summary = model.summary()          # MCMC diagnostics
mean, low, high = model.predict_interval(X_test)`}
        />
      </section>
    </>
  );
}
