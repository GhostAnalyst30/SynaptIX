import type { Metadata } from "next";
import Link from "next/link";
import CodeBlock from "@/components/CodeBlock";
import { MODULES } from "@/lib/content";

export const metadata: Metadata = {
  title: "Documentación — SynaptIX",
  description: "Instalación y primeros pasos con SynaptIX.",
};

export default function DocsIndex() {
  return (
    <article>
      <p className="font-mono text-sm text-volt">// documentación</p>
      <h1 className="mt-2 font-display text-4xl font-bold tracking-tight">
        Instalación y primeros pasos
      </h1>
      <p className="mt-4 max-w-2xl leading-relaxed text-mist">
        SynaptIX requiere Python 3.9+. La instalación base incluye
        scikit-learn; las redes neuronales y el agente DQN requieren el extra{" "}
        <code className="font-mono text-sm text-teal">[dl]</code> (tensorflow).
      </p>

      <h2 className="mt-10 font-display text-2xl font-bold">Instalación</h2>
      <CodeBlock
        title="terminal"
        code={`pip install synaptix            # núcleo: sklearn, pandas, matplotlib
pip install synaptix[dl]        # + tensorflow (neural, DQNAgent)
pip install synaptix[all]       # todo`}
      />

      <h2 className="mt-10 font-display text-2xl font-bold">
        Tu primer modelo en 60 segundos
      </h2>
      <CodeBlock
        title="quickstart.py"
        code={`import synaptix as sx

# 1. Datos: cargar un dataset incluido
df = sx.load_dataset("iris")
X, y = df.drop(columns="species"), df["species"]

# 2. División train/test
X_train, X_test, y_train, y_test = sx.preprocessing.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar
model = sx.supervised.RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluar (métricas + matriz de confusión)
model.evaluate(X_test, y_test, plot=True)

# 5. Guardar para producción
model.save("modelo_iris.pkl")`}
      />

      <h2 className="mt-10 font-display text-2xl font-bold">
        ¿No sabes qué modelo usar?
      </h2>
      <p className="mt-3 max-w-2xl leading-relaxed text-mist">
        Deja que SynaptIX los pruebe todos con validación cruzada:
      </p>
      <CodeBlock
        title="automl.py"
        code={`from synaptix.model_selection import compare_models

tabla = compare_models(X, y, task="classification", cv=5)
# devuelve un ranking de 7 modelos con score, desviación y tiempo`}
      />

      <h2 className="mt-10 font-display text-2xl font-bold">Módulos</h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {MODULES.map((mod) => (
          <Link
            key={mod.slug}
            href={`/docs/${mod.slug}`}
            className="group rounded-lg border border-line bg-ink-2 p-4 transition-colors hover:border-volt/60"
          >
            <p className="font-mono text-sm font-bold text-teal group-hover:text-volt">
              {mod.name}
            </p>
            <p className="mt-1 text-sm text-mist">{mod.title}</p>
          </Link>
        ))}
      </div>
    </article>
  );
}
