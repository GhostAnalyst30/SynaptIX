"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Link from "next/link";
import {
  Workflow,
  Filter,
  Target,
  Boxes,
  Brain,
  Gamepad2,
  Gauge,
  Crosshair,
  Database,
  Sigma,
  Layers,
  Hash,
  Package,
  GitBranch,
  ArrowRight,
  Terminal,
  Copy,
  Check,
  BookOpen,
  ExternalLink,
  Boxes as EcosystemIcon,
  Sparkles,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { ClassCard } from "@/components/ClassCard";
import { CodeBlock } from "@/components/CodeBlock";
import Image from "next/image";
import { SYNAPTIX_VERSION } from "@/lib/version";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

const stagger = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const modules = [
  {
    icon: <Workflow className="w-6 h-6" />,
    title: "Pipelines",
    description:
      "AutoPipeline: automatic ML end-to-end — detects the task, cleans, compares models, tunes and evaluates. Plus declarative step-based Pipelines.",
    tags: ["autopipeline", "automl", "tune"],
    href: "/docs/pipeline",
  },
  {
    icon: <Target className="w-6 h-6" />,
    title: "Supervised Models",
    description:
      "8 regressors and 7 classifiers (random forest, boosting, SVM, KNN...) with one unified fit/predict/evaluate API.",
    tags: ["classification", "regression", "sklearn"],
    href: "/docs/supervised",
  },
  {
    icon: <Boxes className="w-6 h-6" />,
    title: "Unsupervised",
    description:
      "KMeans (elbow + silhouette), DBSCAN, hierarchical, Gaussian mixtures, PCA and TSNE for clustering and dimensionality reduction.",
    tags: ["clustering", "pca", "dbscan"],
    href: "/docs/unsupervised",
  },
  {
    icon: <Sigma className="w-6 h-6" />,
    title: "Bayesian Models",
    description:
      "Bayesian ridge, ARD, Gaussian processes, Naive Bayes variants and full PyMC models with prediction intervals.",
    tags: ["bayesian", "pymc", "uncertainty"],
    href: "/docs/bayesian",
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: "Neural Networks",
    description:
      "MLP, CNN and LSTM networks over Keras with training history plots and forecasting (requires synaptix[dl]).",
    tags: ["mlp", "cnn", "lstm"],
    href: "/docs/neural",
  },
  {
    icon: <Gamepad2 className="w-6 h-6" />,
    title: "Reinforcement",
    description:
      "Q-Learning, SARSA and DQN agents over the GridWorld environment — tabular and deep RL in one API.",
    tags: ["qlearning", "sarsa", "dqn"],
    href: "/docs/reinforcement",
  },
  {
    icon: <Gauge className="w-6 h-6" />,
    title: "Metrics",
    description:
      "Regression (MAE/RMSE/R2/MAPE), classification (accuracy/precision/recall/F1) and clustering metrics with full reports.",
    tags: ["evaluate", "f1", "r2"],
    href: "/docs/metrics",
  },
  {
    icon: <Crosshair className="w-6 h-6" />,
    title: "Model Selection",
    description:
      "k-fold cross-validation, GridSearch tuning and compare_models — AutoML-lite across 7 classifiers or 7 regressors.",
    tags: ["cv", "gridsearch", "compare"],
    href: "/docs/model-selection",
  },
  {
    icon: <Filter className="w-6 h-6" />,
    title: "Preprocessing",
    description:
      "DataCleaner, Scaler, Encoder, Imputer, outlier detection/removal and train_test_split — transformers keep column names.",
    tags: ["clean", "scale", "encode"],
    href: "/docs/preprocessing",
  },
  {
    icon: <Database className="w-6 h-6" />,
    title: "Datasets",
    description:
      "Bundled datasets (iris, penguins, titanic, S&P 500, course completion) with the same loader signature as StatsLibX and ViewX.",
    tags: ["load", "polars", "csv"],
    href: "/docs/datasets",
  },
  {
    icon: <EcosystemIcon className="w-6 h-6" />,
    title: "Ecosystem",
    description:
      "Export results to ViewX dashboards and run statistical diagnostics of models with StatsLibX — each library also works standalone.",
    tags: ["viewx", "statslibx", "payload"],
    href: "/docs/integration",
  },
];

const stats = [
  { icon: Layers, value: "12", label: "Modules", color: "text-accent" },
  { icon: Hash, value: "24+", label: "Models", color: "text-accent2" },
  { icon: Database, value: "5", label: "Datasets", color: "text-accent3" },
  { icon: Package, value: `v${SYNAPTIX_VERSION}`, label: "Version", color: "text-accent" },
  { icon: Zap, value: "AutoML", label: "AutoPipeline", color: "text-accent2" },
];

export default function HomePage() {
  const [copiedPip, setCopiedPip] = useState(false);

  const handleCopyPip = async () => {
    await navigator.clipboard.writeText("pip install synaptix");
    setCopiedPip(true);
    toast.success("Copied to clipboard", {
      description: "pip install synaptix",
    });
    setTimeout(() => setCopiedPip(false), 2000);
  };

  return (
    <div>
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div
          className="absolute inset-0 opacity-[0.15]"
          style={{
            backgroundImage:
              "radial-gradient(rgba(255,255,255,0.3) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="hero-mesh" aria-hidden />

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-accent/10 blur-[120px] pointer-events-none" />

        <motion.div
          className="container-main relative z-10 text-center"
          variants={stagger}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={fadeUp} className="flex justify-center mb-8">
            <div className="relative">
              <div className="absolute inset-0 bg-accent/20 blur-2xl rounded-full scale-150 pointer-events-none" />
              <Image
                src="/icons/favicon.svg"
                alt="SynaptIX"
                width={96}
                height={96}
                className="relative drop-shadow-[0_0_24px_rgba(124,106,247,0.35)]"
                priority
              />
            </div>
          </motion.div>

          <motion.div variants={fadeUp} className="flex justify-center mb-4 gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs font-mono text-accent bg-accent/10 border border-accent/20 px-3 py-1 rounded-full">
              <Package className="w-3 h-3" />
              v{SYNAPTIX_VERSION}
            </span>
            <span className="inline-flex items-center gap-1.5 text-xs font-mono text-accent2 bg-accent2/10 border border-accent2/20 px-3 py-1 rounded-full">
              <Zap className="w-3 h-3" />
              New: Bayesian models + AutoPipeline
            </span>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-syne text-5xl md:text-7xl lg:text-8xl font-extrabold tracking-tight leading-none"
          >
            <span className="text-white">Train. Evaluate.</span>
            <br />
            <span className="text-gradient">Improve.</span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="mt-6 text-lg md:text-xl text-muted max-w-2xl mx-auto leading-relaxed"
          >
            An integral machine learning library: supervised, unsupervised,
            reinforcement and neural models behind one unified API, with
            automatic pipelines, evaluation and tuning.
            <span className="text-text font-medium"> The real library runs in your browser</span> on the playground: no install required.
          </motion.p>

          <motion.div variants={fadeUp} className="mt-8 flex justify-center">
            <div className="inline-flex items-center gap-3 bg-surface border border-border rounded-lg px-5 py-3 font-mono text-sm">
              <Terminal className="w-4 h-4 text-accent2" />
              <span className="text-muted">$</span>
              <span className="text-text">pip install synaptix</span>
              <button
                onClick={handleCopyPip}
                className="ml-2 p-1.5 rounded-md hover:bg-white/5 transition-colors cursor-pointer"
              >
                {copiedPip ? (
                  <Check className="w-4 h-4 text-green-400" />
                ) : (
                  <Copy className="w-4 h-4 text-muted hover:text-white" />
                )}
              </button>
            </div>
          </motion.div>

          <motion.div
            variants={fadeUp}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <Link
              href="/playground"
              className="inline-flex items-center gap-2 bg-accent hover:bg-accent/90 text-white font-semibold px-6 py-3 rounded-lg transition-all no-underline glow-accent"
            >
              <Sparkles className="w-4 h-4" />
              Try the Playground <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/installation"
              className="inline-flex items-center gap-2 border border-border hover:border-accent/40 hover:text-white text-text font-semibold px-6 py-3 rounded-lg transition-all no-underline"
            >
              Get Started <ArrowRight className="w-4 h-4" />
            </Link>
            <a
              href="https://github.com/GhostAnalyst30/SynaptIX"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 border border-border hover:border-white/20 text-text font-semibold px-6 py-3 rounded-lg transition-all no-underline"
            >
              <GitBranch className="w-4 h-4" />
              GitHub <ExternalLink className="w-3.5 h-3.5 text-muted" />
            </a>
          </motion.div>

          <motion.div
            className="absolute bottom-2 left-1/2 -translate-x-1/2"
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="w-5 h-8 rounded-full border border-border flex items-start justify-center p-1.5">
              <div className="w-1 h-2 rounded-full bg-accent" />
            </div>
          </motion.div>
        </motion.div>
      </section>

      <section className="border-y border-border bg-surface">
        <div className="container-main py-6">
          <motion.div
            className="flex flex-wrap items-center justify-center gap-8 md:gap-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            {stats.map((stat) => {
              const Icon = stat.icon;
              return (
                <div key={stat.label} className="flex items-center gap-2.5">
                  <Icon className={`w-4 h-4 ${stat.color}`} />
                  <span className="font-syne font-bold text-lg text-white">
                    {stat.value}
                  </span>
                  <span className="text-sm text-muted">{stat.label}</span>
                </div>
              );
            })}
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container-main">
          <motion.div
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Built-in Modules
            </h2>
            <p className="mt-3 text-muted max-w-xl mx-auto">
              Twelve modules covering the full ML workflow — from raw data to a
              tuned, evaluated and exported model.
            </p>
          </motion.div>

          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
          >
            {modules.map((mod) => (
              <motion.div key={mod.title} variants={fadeUp}>
                <ClassCard
                  icon={mod.icon}
                  title={mod.title}
                  description={mod.description}
                  tags={mod.tags}
                  href={mod.href}
                />
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28 bg-surface/50">
        <div className="container-main">
          <motion.div
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Quick Start
            </h2>
            <p className="mt-3 text-muted max-w-xl mx-auto">
              One line from raw DataFrame to a trained, evaluated model.
            </p>
          </motion.div>

          <motion.div
            className="max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <CodeBlock
              title="example.py"
              code={`import synaptix as sx

df = sx.load_dataset("iris")

# Automatic ML: detect task -> clean -> compare -> tune -> evaluate
auto = sx.AutoPipeline(task="classification")
auto.fit(df, target="species")
print(auto.best_name_, auto.results_)

# Or manual: split, train and get train vs. test metrics in one call
X, y = df.drop(columns="species"), df["species"]
model = sx.supervised.RandomForestClassifier(n_estimators=100)
result = model.fit_evaluate(X, y, test_size=0.2, metrics=["accuracy", "f1"])
print(result.to_frame())
model.diagnostics(X, y)   # via statslibx`}
            />
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container-main">
          <motion.div
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Built-in Datasets
            </h2>
            <p className="mt-3 text-muted max-w-xl mx-auto">
              Ready-to-use datasets for immediate training and evaluation.
            </p>
          </motion.div>

          <motion.div
            className="flex flex-wrap justify-center gap-3"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
          >
            {["iris.csv", "penguins.csv", "titanic.csv", "sp500_companies.csv", "course_completion.csv"].map((name) => (
              <motion.div key={name} variants={fadeUp}>
                <span className="inline-flex items-center gap-2 bg-card border border-border rounded-full px-4 py-2 text-sm font-mono text-muted hover:border-accent/30 hover:text-text transition-colors cursor-default">
                  {name}
                </span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28 bg-surface/50 border-t border-border">
        <div className="container-main text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Ready to Train?
            </h2>
            <p className="mt-3 text-muted max-w-lg mx-auto">
              The playground loads the real synaptix (with scikit-learn) in
              your browser — train models and evaluate them, zero installation.
            </p>
            <Link
              href="/playground"
              className="mt-8 inline-flex items-center gap-2 bg-accent hover:bg-accent/90 text-white font-semibold px-8 py-3.5 rounded-lg transition-all no-underline"
            >
              <BookOpen className="w-4 h-4" />
              Open Playground <ArrowRight className="w-4 h-4" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
