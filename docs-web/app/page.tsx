import Link from "next/link";
import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";
import Hero from "@/components/landing/Hero";
import LiveDemos from "@/components/landing/LiveDemos";
import ModuleGrid from "@/components/landing/ModuleGrid";
import Pipeline from "@/components/landing/Pipeline";
import Reveal from "@/components/Reveal";

const MARQUEE_ITEMS = [
  "RandomForestClassifier",
  "KMeans",
  "QLearningAgent",
  "DataCleaner",
  "compare_models",
  "MLP",
  "PCA",
  "GridSearch",
  "LSTMNet",
  "DBSCAN",
  "classification_report",
  "GradientBoosting",
];

function Marquee() {
  const items = [...MARQUEE_ITEMS, ...MARQUEE_ITEMS];
  return (
    <div className="overflow-hidden border-y border-line bg-ink-2 py-3">
      <div className="animate-marquee flex w-max gap-8">
        {items.map((item, index) => (
          <span
            key={index}
            className="font-mono text-sm whitespace-nowrap text-dim"
          >
            <span className="text-volt">·</span> {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function FinalCta() {
  return (
    <section className="border-t border-line py-28 text-center">
      <div className="mx-auto max-w-3xl px-5">
        <Reveal>
          <h2 className="font-display text-4xl font-bold tracking-tight sm:text-6xl">
            Tu próximo modelo,
            <br />
            en <span className="text-volt">tres líneas</span>.
          </h2>
        </Reveal>
        <Reveal delay={0.15}>
          <div className="mt-10 inline-block rounded-lg border border-line bg-ink-2 px-6 py-4 text-left font-mono text-sm">
            <p>
              <span className="text-dim">$</span> pip install synaptix
            </p>
          </div>
        </Reveal>
        <Reveal delay={0.25}>
          <div className="mt-8">
            <Link
              href="/docs"
              className="inline-block rounded-lg bg-volt px-8 py-4 font-mono text-sm font-bold text-ink transition-transform hover:scale-[1.03]"
            >
              empezar ahora →
            </Link>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Marquee />
        <Pipeline />
        <LiveDemos />
        <ModuleGrid />
        <FinalCta />
      </main>
      <Footer />
    </>
  );
}
