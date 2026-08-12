"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { useEffect, useState } from "react";

const CODE_LINES = [
  "import synaptix as sx",
  "",
  'df = sx.load_dataset("iris")',
  'X, y = df.drop(columns="species"), df["species"]',
  "",
  "model = sx.supervised.RandomForestClassifier()",
  "model.fit(X_train, y_train)",
  "model.evaluate(X_test, y_test, plot=True)",
  "",
  "# accuracy: 0.9556 · f1: 0.9552 ✓",
];

function TypingCode() {
  const [text, setText] = useState("");

  useEffect(() => {
    const full = CODE_LINES.join("\n");
    let index = 0;
    const interval = setInterval(() => {
      index += 2;
      setText(full.slice(0, index));
      if (index >= full.length) clearInterval(interval);
    }, 24);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-ink-2 shadow-[0_0_80px_rgba(163,255,69,0.06)]">
      <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-rose/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-amber/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-volt/70" />
        <span className="ml-2 font-mono text-xs text-dim">quickstart.py</span>
      </div>
      <pre className="min-h-[240px] p-5 font-mono text-[13px] leading-relaxed text-fog">
        <code>
          {text.split("\n").map((line, index) => (
            <div key={index}>
              {line.startsWith("#") ? (
                <span className="text-volt">{line}</span>
              ) : (
                colorize(line)
              )}
            </div>
          ))}
        </code>
        <span className="typing-cursor" />
      </pre>
    </div>
  );
}

function colorize(line: string) {
  const parts = line.split(
    /(\bimport\b|\bfrom\b|\bas\b|"[^"]*"|\bmodel\b|\bsx\b)/g,
  );
  return parts.map((part, index) => {
    if (["import", "from", "as"].includes(part))
      return (
        <span key={index} className="tok-kw">
          {part}
        </span>
      );
    if (part.startsWith('"'))
      return (
        <span key={index} className="tok-str">
          {part}
        </span>
      );
    if (part === "sx" || part === "model")
      return (
        <span key={index} className="tok-fn">
          {part}
        </span>
      );
    return <span key={index}>{part}</span>;
  });
}

function PipInstall() {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText("pip install synaptix");
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      onClick={copy}
      className="group flex items-center gap-3 rounded-lg border border-line bg-ink-2 px-5 py-3 font-mono text-sm transition-colors hover:border-volt"
    >
      <span className="text-dim">$</span>
      <span className="text-fog">pip install synaptix</span>
      <span className="text-xs text-dim transition-colors group-hover:text-volt">
        {copied ? "✓" : "⧉"}
      </span>
    </button>
  );
}

export default function Hero() {
  return (
    <section className="bg-blueprint relative overflow-hidden pt-32 pb-24">
      {/* halo */}
      <div className="pointer-events-none absolute -top-40 left-1/2 h-[560px] w-[900px] -translate-x-1/2 rounded-full bg-volt/6 blur-3xl" />

      <div className="mx-auto grid max-w-7xl items-center gap-14 px-5 lg:grid-cols-2">
        <div>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-5 inline-block rounded-full border border-line px-3 py-1 font-mono text-xs text-volt"
          >
            v0.1.7 — supervisado · bayesiano · refuerzo · redes neuronales · AutoML
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-display text-5xl leading-[1.02] font-extrabold tracking-tight sm:text-6xl lg:text-7xl"
          >
            Machine Learning
            <br />
            <span className="text-volt">sin fricción,</span>
            <br />
            en español.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.22 }}
            className="mt-6 max-w-lg text-lg leading-relaxed text-mist"
          >
            SynaptIX unifica todo el ciclo de ML — preprocesamiento, 15 modelos
            supervisados, clustering, agentes de refuerzo y redes neuronales —
            bajo una sola API estilo{" "}
            <code className="font-mono text-sm text-teal">
              fit / predict / evaluate
            </code>
            .
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.34 }}
            className="mt-9 flex flex-wrap items-center gap-4"
          >
            <PipInstall />
            <Link
              href="/docs"
              className="rounded-lg bg-volt px-6 py-3 font-mono text-sm font-bold text-ink transition-transform hover:scale-[1.03]"
            >
              leer la documentación →
            </Link>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.3 }}
        >
          <TypingCode />
        </motion.div>
      </div>
    </section>
  );
}
