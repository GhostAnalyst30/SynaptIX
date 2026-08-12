import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import CodeBlock from "@/components/CodeBlock";
import Reveal from "@/components/Reveal";
import GradientDescentViz from "@/components/viz/GradientDescentViz";
import GridWorldViz from "@/components/viz/GridWorldViz";
import KMeansViz from "@/components/viz/KMeansViz";
import { getModule, MODULES } from "@/lib/content";

export function generateStaticParams() {
  return MODULES.map((mod) => ({ slug: mod.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const mod = getModule(slug);
  if (!mod) return {};
  return {
    title: `${mod.name} — SynaptIX`,
    description: mod.intro,
  };
}

const VIZ = {
  kmeans: KMeansViz,
  gradient: GradientDescentViz,
  gridworld: GridWorldViz,
};

export default async function ModulePage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const mod = getModule(slug);
  if (!mod) notFound();

  const index = MODULES.findIndex((m) => m.slug === slug);
  const prev = index > 0 ? MODULES[index - 1] : null;
  const next = index < MODULES.length - 1 ? MODULES[index + 1] : null;

  return (
    <article>
      <p className="font-mono text-sm text-volt">// {mod.name}</p>
      <h1 className="mt-2 font-display text-4xl font-bold tracking-tight">
        {mod.title}
      </h1>
      <p className="mt-4 max-w-2xl leading-relaxed text-mist">{mod.intro}</p>

      {mod.sections.map((section, sIndex) => {
        const Viz = section.viz ? VIZ[section.viz] : null;
        return (
          <Reveal key={sIndex} className="mt-12">
            <h2 className="font-display text-2xl font-bold">
              {section.heading}
            </h2>
            {section.body && (
              <p className="mt-3 max-w-2xl leading-relaxed text-mist">
                {section.body}
              </p>
            )}
            {section.code && (
              <CodeBlock code={section.code} title={section.codeTitle} />
            )}
            {Viz && (
              <div className="mt-4 max-w-2xl">
                <Viz />
              </div>
            )}
          </Reveal>
        );
      })}

      <nav className="mt-16 flex justify-between border-t border-line pt-6 font-mono text-sm">
        {prev ? (
          <Link
            href={`/docs/${prev.slug}`}
            className="link-slide text-mist hover:text-fog"
          >
            ← {prev.name.replace("synaptix.", "")}
          </Link>
        ) : (
          <Link href="/docs" className="link-slide text-mist hover:text-fog">
            ← instalación
          </Link>
        )}
        {next && (
          <Link
            href={`/docs/${next.slug}`}
            className="link-slide text-mist hover:text-fog"
          >
            {next.name.replace("synaptix.", "")} →
          </Link>
        )}
      </nav>
    </article>
  );
}
