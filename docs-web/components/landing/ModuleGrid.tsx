"use client";

import { motion } from "framer-motion";
import Link from "next/link";

const MODULES = [
  {
    slug: "preprocessing",
    name: "preprocessing",
    tagline: "Limpieza y transformación",
    items: ["Scaler", "Encoder", "Imputer", "DataCleaner", "outliers"],
    accent: "text-teal",
  },
  {
    slug: "supervised",
    name: "supervised",
    tagline: "Regresión y clasificación",
    items: ["RandomForest", "GradientBoosting", "SVM", "KNN", "+11 más"],
    accent: "text-volt",
  },
  {
    slug: "unsupervised",
    name: "unsupervised",
    tagline: "Clustering y reducción",
    items: ["KMeans", "DBSCAN", "GaussianMixture", "PCA", "TSNE"],
    accent: "text-amber",
  },
  {
    slug: "reinforcement",
    name: "reinforcement",
    tagline: "Aprendizaje por refuerzo",
    items: ["QLearningAgent", "SARSAAgent", "DQNAgent", "GridWorld"],
    accent: "text-rose",
  },
  {
    slug: "neural",
    name: "neural",
    tagline: "Redes neuronales",
    items: ["MLP", "CNN", "LSTMNet", "early stopping", "curvas"],
    accent: "text-volt",
  },
  {
    slug: "metrics",
    name: "metrics",
    tagline: "Evaluación de modelos",
    items: ["regresión", "clasificación", "clustering", "reportes"],
    accent: "text-teal",
  },
  {
    slug: "model-selection",
    name: "model_selection",
    tagline: "Validación y AutoML-lite",
    items: ["cross_validate", "GridSearch", "compare_models"],
    accent: "text-amber",
  },
  {
    slug: "datasets",
    name: "datasets",
    tagline: "Datos de ejemplo",
    items: ["iris", "titanic", "penguins", "sp500", "cursos"],
    accent: "text-rose",
  },
];

export default function ModuleGrid() {
  return (
    <section className="border-t border-line py-24">
      <div className="mx-auto max-w-7xl px-5">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="flex flex-wrap items-end justify-between gap-4"
        >
          <div>
            <p className="font-mono text-sm text-volt">// ocho submódulos</p>
            <h2 className="mt-3 font-display text-4xl font-bold tracking-tight sm:text-5xl">
              Una librería, todo el stack.
            </h2>
          </div>
          <Link
            href="/docs"
            className="link-slide font-mono text-sm text-mist hover:text-fog"
          >
            ver toda la API →
          </Link>
        </motion.div>

        <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {MODULES.map((mod, index) => (
            <motion.div
              key={mod.slug}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.5, delay: (index % 4) * 0.08 }}
            >
              <Link
                href={`/docs/${mod.slug}`}
                className="group block h-full rounded-lg border border-line bg-ink-2 p-5 transition-all hover:-translate-y-1 hover:border-volt/60 hover:shadow-[0_8px_40px_rgba(163,255,69,0.07)]"
              >
                <p className={`font-mono text-sm font-bold ${mod.accent}`}>
                  synaptix.{mod.name}
                </p>
                <p className="mt-1 text-sm text-mist">{mod.tagline}</p>
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {mod.items.map((item) => (
                    <span
                      key={item}
                      className="rounded border border-line px-1.5 py-0.5 font-mono text-[10px] text-dim group-hover:text-mist"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
