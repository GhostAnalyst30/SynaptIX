import { Shell, Package, Download, Heart, Beaker, FileJson, Cpu, Boxes, Sigma, Database, MessageSquare } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

const requirements = [
  { package: "Python", version: ">= 3.9" },
  { package: "numpy", version: ">= 1.21.0" },
  { package: "pandas", version: ">= 1.3.0" },
  { package: "scipy", version: ">= 1.7.0" },
  { package: "matplotlib", version: ">= 3.4.0" },
  { package: "scikit-learn", version: ">= 1.2.0" },
  { package: "requests", version: ">= 2.25.0" },
  { package: "tensorflow", version: ">= 2.12.0 (optional extra: dl)" },
  { package: "pymc", version: ">= 5.0.0 (optional extra: bayes)" },
  { package: "viewx", version: ">= 0.3.0 (optional extra: viewx)" },
  { package: "statslibx", version: ">= 0.4.0 (optional extra: stats)" },
];

const troubleshooting = [
  {
    problem: "Command not found: pip",
    solution:
      "Ensure Python and pip are installed and added to your PATH. Try using python -m pip install synaptix instead.",
  },
  {
    problem: "ImportError: No module named 'synaptix'",
    solution:
      "The package may not be installed correctly. Re-run pip install synaptix and verify with pip list.",
  },
  {
    problem: "ImportError about tensorflow when importing synaptix.neural",
    solution:
      "Neural networks (MLP, CNN, LSTMNet) require Keras/TensorFlow. Install the extra: pip install 'synaptix[dl]'. Every other module works without it.",
  },
  {
    problem: "Model diagnostics() raises a statslibx error",
    solution:
      "diagnostics() uses StatsLibX for inferential tests on residuals. Install the extra: pip install 'synaptix[stats]'.",
  },
  {
    problem: "to_html()/to_report() raise 'viewx not installed'",
    solution:
      "Result export to dashboards/PDF/slides needs ViewX. Install the extra: pip install 'synaptix[viewx]'.",
  },
];

export default function InstallationPage() {
  return (
    <>
      <DocHeader
        title="Installation"
        description="Get started with SynaptIX in minutes"
        icon={<Shell className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Package className="w-5 h-5 text-accent" />
          <h2 className="section-title">Quick Install</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Install SynaptIX from PyPI using pip — the core (with scikit-learn) has no heavy requirements:
        </p>
        <CodeBlock title="bash" code="pip install synaptix" />
        <p className="text-sm text-muted leading-relaxed mt-6 mb-4">
          Verify the installation:
        </p>
        <CodeBlock
          title="python"
          code={`import synaptix as sx
print(sx.__version__)
print(sx.list_datasets())`}
        />
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Download className="w-5 h-5 text-accent" />
          <h2 className="section-title">Requirements</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Core dependencies ship with every install; the rest are extras:
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne font-semibold text-white py-3 px-4">Package</th>
                <th className="text-left font-syne font-semibold text-white py-3 px-4">Minimum Version</th>
              </tr>
            </thead>
            <tbody>
              {requirements.map((req) => (
                <tr key={req.package} className="border-b border-border/50 hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-4 font-mono text-text">{req.package}</td>
                  <td className="py-3 px-4 font-mono text-accent">{req.version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Beaker className="w-5 h-5 text-accent" />
          <h2 className="section-title">Optional Extras</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">dl</span>
            </div>
            <p className="text-xs text-muted">TensorFlow for neural networks and DQN.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Sigma className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">bayes</span>
            </div>
            <p className="text-xs text-muted">PyMC for full Bayesian inference models.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Boxes className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">viewx</span>
            </div>
            <p className="text-xs text-muted">Export results to HTML dashboards, PDF and presentations.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Database className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">stats</span>
            </div>
            <p className="text-xs text-muted">StatsLibX for statistical model diagnostics.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <MessageSquare className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">nlp</span>
            </div>
            <p className="text-xs text-muted">NLTK for the legacy NLP API.</p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Heart className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">all</span>
            </div>
            <p className="text-xs text-muted">Everything above, plus polars and plotly.</p>
          </div>
        </div>
        <div className="space-y-3">
          <CodeBlock title="bash" code="pip install 'synaptix[dl]'" />
          <CodeBlock title="bash" code="pip install 'synaptix[viewx,stats]'" />
          <CodeBlock title="bash" code="pip install 'synaptix[all]'" />
        </div>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <FileJson className="w-5 h-5 text-accent" />
          <h2 className="section-title">Development Install</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Clone the repository and install in editable mode:
        </p>
        <CodeBlock
          title="bash"
          code={`git clone https://github.com/GhostAnalyst30/SynaptIX.git
cd SynaptIX
pip install -e .`}
        />
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Package className="w-5 h-5 text-accent" />
          <h2 className="section-title">Quick Start</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Train and evaluate your first model in a few lines:
        </p>
        <CodeBlock
          title="python"
          code={`import synaptix as sx

df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]
X_train, X_test, y_train, y_test = sx.preprocessing.train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = sx.supervised.RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
model.evaluate(X_test, y_test, plot=True)`}
        />
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Heart className="w-5 h-5 text-accent" />
          <h2 className="section-title">First Steps</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">1</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Load Data</h3>
              <p className="text-xs text-muted leading-relaxed">
                Use <code className="code-inline">sx.load_dataset()</code> or any pandas/polars DataFrame.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">2</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Preprocess</h3>
              <p className="text-xs text-muted leading-relaxed">
                <code className="code-inline">DataCleaner</code>, <code className="code-inline">Scaler</code>,{" "}
                <code className="code-inline">Encoder</code> and <code className="code-inline">Imputer</code> keep column names.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">3</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Train &amp; Evaluate</h3>
              <p className="text-xs text-muted leading-relaxed">
                <code className="code-inline">fit()</code> then <code className="code-inline">evaluate()</code> — or let{" "}
                <code className="code-inline">AutoPipeline</code> do everything.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">4</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Improve &amp; Export</h3>
              <p className="text-xs text-muted leading-relaxed">
                Tune with <code className="code-inline">GridSearch</code>, diagnose with{" "}
                <code className="code-inline">diagnostics()</code> (statslibx) and export with{" "}
                <code className="code-inline">to_html()</code> (viewx).
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <FileJson className="w-5 h-5 text-accent" />
          <h2 className="section-title">Troubleshooting</h2>
        </div>
        <div className="space-y-3">
          {troubleshooting.map((item) => (
            <details
              key={item.problem}
              className="group rounded-lg border border-border bg-white/[0.02] overflow-hidden"
            >
              <summary className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-white/[0.03] transition-colors">
                <span className="text-sm font-mono text-text">{item.problem}</span>
                <span className="text-muted group-open:rotate-180 transition-transform">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </span>
              </summary>
              <div className="px-4 pb-3 border-t border-border/50 pt-3">
                <p className="text-xs text-muted leading-relaxed">{item.solution}</p>
              </div>
            </details>
          ))}
        </div>
      </section>
    </>
  );
}
