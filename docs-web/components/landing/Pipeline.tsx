"use client";

import { motion } from "framer-motion";

const STAGES = [
  {
    icon: "▤",
    title: "Datos",
    module: "sx.datasets",
    description:
      "Carga datasets incluidos (iris, titanic, penguins...) o tus propios DataFrames de pandas.",
    code: 'df = sx.load_dataset("titanic")',
  },
  {
    icon: "⚙",
    title: "Preprocesamiento",
    module: "sx.preprocessing",
    description:
      "DataCleaner analiza nulos, outliers y tipos; imputa, codifica y escala en una sola llamada.",
    code: 'df = DataCleaner().clean(df, target="survived")',
  },
  {
    icon: "◉",
    title: "Modelo",
    module: "sx.supervised",
    description:
      "15 modelos con la misma API. O deja que compare_models() elija el mejor por ti.",
    code: "model.fit(X_train, y_train)",
  },
  {
    icon: "✓",
    title: "Evaluación",
    module: "sx.metrics",
    description:
      "Métricas y gráficos automáticos: matriz de confusión, ROC, curvas de aprendizaje.",
    code: "model.evaluate(X_test, y_test, plot=True)",
  },
];

export default function Pipeline() {
  return (
    <section className="border-t border-line py-24">
      <div className="mx-auto max-w-7xl px-5">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
        >
          <p className="font-mono text-sm text-volt">// el pipeline completo</p>
          <h2 className="mt-3 font-display text-4xl font-bold tracking-tight sm:text-5xl">
            De CSV a modelo evaluado
            <br />
            en cuatro pasos.
          </h2>
        </motion.div>

        <div className="relative mt-16 grid gap-6 lg:grid-cols-4">
          {/* línea conectora */}
          <div className="absolute top-10 right-8 left-8 hidden h-px bg-line lg:block" />

          {STAGES.map((stage, index) => (
            <motion.div
              key={stage.title}
              initial={{ opacity: 0, y: 32 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.55, delay: index * 0.14 }}
              className="group relative rounded-lg border border-line bg-ink-2 p-6 transition-colors hover:border-volt/50"
            >
              <div className="relative z-10 grid h-9 w-9 place-items-center rounded border border-line bg-ink font-mono text-volt transition-colors group-hover:border-volt">
                {stage.icon}
              </div>
              <p className="mt-4 font-mono text-xs text-dim">
                paso {index + 1} — {stage.module}
              </p>
              <h3 className="mt-1 font-display text-xl font-bold">
                {stage.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {stage.description}
              </p>
              <code className="mt-4 block overflow-x-auto rounded bg-ink px-3 py-2 font-mono text-[11px] whitespace-nowrap text-teal">
                {stage.code}
              </code>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
