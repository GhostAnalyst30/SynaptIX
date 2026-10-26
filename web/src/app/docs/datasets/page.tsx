import { Database } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

const DATASETS = [
  { name: "iris", desc: "Fisher's Iris flower measurements (3 species)", rows: "150", cols: "5" },
  { name: "penguins", desc: "Palmer penguins morphometrics (3 species)", rows: "344", cols: "7" },
  { name: "titanic", desc: "Titanic passenger manifest", rows: "418", cols: "12" },
  { name: "sp500_companies", desc: "S&P 500 companies with financial metrics", rows: "502", cols: "16" },
  { name: "course_completion", desc: "Student course completion records", rows: "100,000", cols: "40" },
];

export default function DatasetsDocs() {
  return (
    <>
      <DocHeader
        title="Datasets"
        description="Bundled datasets ready for training, with the same loader signature as StatsLibX and ViewX — including polars backend and (X, y) array output."
        icon={<Database className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Bundled Datasets</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Dataset</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Description</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Rows</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Cols</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {DATASETS.map((ds) => (
                <tr key={ds.name} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 pr-4 font-mono text-xs text-accent">{ds.name}</td>
                  <td className="py-3 pr-4 text-muted">{ds.desc}</td>
                  <td className="py-3 pr-4 font-mono text-xs text-muted">{ds.rows}</td>
                  <td className="py-3 font-mono text-xs text-muted">{ds.cols}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="load_dataset"
            signature="load_dataset(name, backend='pandas', return_X_y=None) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            description="Load a bundled dataset as a pandas DataFrame, a polars DataFrame (requires synaptix[polars]) or as (X, y) numpy arrays. The signature matches statslibx and viewx."
            parameters={[
              { name: "name", type: "str", description: "Dataset name (e.g. 'iris') — see list_datasets()." },
              { name: "backend", type: "str", description: "'pandas' or 'polars'.", default: "'pandas'" },
              { name: "return_X_y", type: "bool | str | None", description: "True returns (X, y) numpy arrays using the dataset's default target column; a string is used as the target column name.", default: "None" },
            ]}
            returns="pd.DataFrame | pl.DataFrame | tuple[np.ndarray, np.ndarray]"
            example={`import synaptix as sx

df = sx.load_dataset("iris")

# polars DataFrame — feeds models directly, no conversion needed
df_pl = sx.load_dataset("iris", backend="polars")

# Arrays for sklearn-style workflows
X, y = sx.load_dataset("iris", return_X_y=True)             # default target: species
X, y = sx.load_dataset("titanic", return_X_y="Survived")    # custom target column`}
          />

          <MethodCard
            name="list_datasets"
            signature="list_datasets() -> list[str]"
            description="List the bundled dataset names."
            returns="list[str]"
            example={`print(sx.list_datasets())   # ['iris', 'penguins', ...]`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Ecosystem Compatibility</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The same <code className="code-inline">load_dataset</code> call works across the ecosystem, and
          polars output from StatsLibX feeds SynaptIX models directly:
        </p>
        <CodeBlock
          title="cross.py"
          code={`from statslibx import load_dataset as slx_load
import synaptix as sx

df_pl = slx_load("iris.csv", backend="polars")   # statslibx polars backend
model = sx.supervised.RandomForestClassifier()
model.fit(df_pl.drop("species"), df_pl["species"])   # converted internally`}
        />
      </section>
    </>
  );
}
