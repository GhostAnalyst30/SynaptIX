"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import GridWorldViz from "../viz/GridWorldViz";
import KMeansViz from "../viz/KMeansViz";

export default function LiveDemos() {
  return (
    <section className="bg-blueprint border-t border-line py-24">
      <div className="mx-auto max-w-7xl px-5">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
        >
          <p className="font-mono text-sm text-volt">
            // los algoritmos, en vivo
          </p>
          <h2 className="mt-3 max-w-2xl font-display text-4xl font-bold tracking-tight sm:text-5xl">
            Mira cómo aprenden los modelos.
          </h2>
          <p className="mt-4 max-w-xl text-mist">
            Las mismas ideas que implementa SynaptIX, animadas en tu navegador.
            Presiona <span className="font-mono text-volt">entrenar</span> y
            observa la convergencia.
          </p>
        </motion.div>

        <div className="mt-14 grid gap-6 lg:grid-cols-2">
          <motion.div
            initial={{ opacity: 0, x: -32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-3 flex items-baseline justify-between">
              <h3 className="font-display text-lg font-bold">
                K-Means · <span className="text-amber">no supervisado</span>
              </h3>
              <Link
                href="/docs/unsupervised"
                className="link-slide font-mono text-xs text-dim"
              >
                docs →
              </Link>
            </div>
            <KMeansViz />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <div className="mb-3 flex items-baseline justify-between">
              <h3 className="font-display text-lg font-bold">
                Q-Learning · <span className="text-rose">refuerzo</span>
              </h3>
              <Link
                href="/docs/reinforcement"
                className="link-slide font-mono text-xs text-dim"
              >
                docs →
              </Link>
            </div>
            <GridWorldViz />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
