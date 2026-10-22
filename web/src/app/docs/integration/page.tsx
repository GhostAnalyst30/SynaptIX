import { Boxes } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

export default function IntegrationDocs() {
  return (
    <>
      <DocHeader
        title="Ecosystem — ViewX & StatsLibX"
        description="SynaptIX is the ML engine of the GhostAnalyst data ecosystem. Export results to ViewX dashboards and reports, run statistical diagnostics with StatsLibX — each library also works standalone."
        icon={<Boxes className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">The Three Libraries</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Library</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Role</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Install</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">statslibx</td>
                <td className="py-3 pr-4 text-muted">Statistics, data handling, simulations</td>
                <td className="py-3 font-mono text-xs text-muted">pip install statslibx</td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">synaptix</td>
                <td className="py-3 pr-4 text-muted">ML models, evaluation and tuning</td>
                <td className="py-3 font-mono text-xs text-muted">pip install synaptix</td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">viewx</td>
                <td className="py-3 pr-4 text-muted">Dashboards, EDA, PDF, slides</td>
                <td className="py-3 font-mono text-xs text-muted">pip install viewx</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Export to ViewX (pip install synaptix[viewx])</h2>
        <CodeBlock
          title="export.py"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]
model = sx.supervised.RandomForestClassifier().fit(X, y)
result = model.evaluate(X, y)   # plain dict of metrics, as always

result.to_html("evaluacion.html", theme="dark_enterprise")   # dashboard
result.to_report("evaluacion")                               # PDF
result.to_presentation("evaluacion_slides.html")             # slides

# compare_models table -> dashboard
tabla = sx.model_selection.compare_models(X, y, task="classification")
tabla.to_dashboard("comparacion.html")

# Or the manual pipeline with the shared payload contract
payload = result.to_report_data(include_figures=True)
from viewx import from_report_payload
from_report_payload(payload, target="html", filename="evaluacion.html")`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          Without ViewX installed everything works normally — export methods raise a clear error with
          the install instruction.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Statistical Diagnostics (pip install synaptix[stats])</h2>
        <CodeBlock
          title="diagnostics.py"
          code={`diagnostico = model.diagnostics(X_test, y_test)

# Regression models: residual normality (shapiro, ks, anderson,
# jarque-bera) + t-test of mean-zero residuals
print(diagnostico["normalidad_residuos"].interpretation)

# Classification models: chi-square of real vs predicted`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Data In from StatsLibX</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Models accept DataFrames, arrays, lists and polars DataFrames — StatsLibX output plugs in
          without manual conversion:
        </p>
        <CodeBlock
          title="dataflow.py"
          code={`df_pl = sx.load_dataset("iris", backend="polars")   # requires synaptix[polars]
model.fit(df_pl.drop("species"), df_pl["species"])

X, y = sx.load_dataset("iris", return_X_y=(
    ["sepal_length", "sepal_width", "petal_length", "petal_width"], "species"))`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Full Ecosystem Flow</h2>
        <CodeBlock
          title="ecosystem.py"
          code={`# pip install statslibx[viewx] synaptix[viewx,stats] viewx
from statslibx import DescriptiveStats, load_iris
from viewx import from_report_payload
import synaptix as sx

df = load_iris()

# 1. Statistics and data quality (StatsLibX)
DescriptiveStats(df).summary()

# 2. Model, evaluation and diagnostics (SynaptIX + StatsLibX)
X, y = df.drop(columns="species"), df["species"]
model = sx.supervised.RandomForestClassifier().fit(X, y)
evaluation = model.evaluate(X, y)
model.diagnostics(X, y)

# 3. Everything into one dashboard (ViewX)
payload = evaluation.to_report_data(include_figures=True)
from_report_payload(payload, target="html", filename="analisis.html")`}
        />
      </section>
    </>
  );
}
