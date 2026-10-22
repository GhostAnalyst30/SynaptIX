"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
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
  Boxes as EcosystemIcon,
  Shell,
  Search,
} from "lucide-react";
import { BrandLogo } from "@/components/BrandLogo";

const sections = [
  {
    label: "Getting Started",
    items: [
      { href: "/installation", label: "Installation", icon: Shell, keywords: "install pip extras requirements dl bayes viewx stats polars" },
    ],
  },
  {
    label: "ML Automation",
    items: [
      { href: "/docs/pipeline", label: "Pipelines", icon: Workflow, keywords: "autopipeline declarative clean tune evaluate detect task" },
      { href: "/docs/preprocessing", label: "Preprocessing", icon: Filter, keywords: "scaler encoder imputer datacleaner outliers split" },
    ],
  },
  {
    label: "Models",
    items: [
      { href: "/docs/supervised", label: "Supervised", icon: Target, keywords: "classification regression random forest logistic tree svm knn boosting" },
      { href: "/docs/unsupervised", label: "Unsupervised", icon: Boxes, keywords: "kmeans dbscan hierarchical gmm pca tsne clustering" },
      { href: "/docs/bayesian", label: "Bayesian Models", icon: Sigma, keywords: "bayesian ridge ard gaussian process naive bayes pymc interval" },
      { href: "/docs/neural", label: "Neural Networks", icon: Brain, keywords: "mlp cnn lstm keras tensorflow history forecast" },
      { href: "/docs/reinforcement", label: "Reinforcement", icon: Gamepad2, keywords: "qlearning sarsa dqn gridworld agent environment" },
    ],
  },
  {
    label: "Evaluation",
    items: [
      { href: "/docs/metrics", label: "Metrics", icon: Gauge, keywords: "accuracy precision recall f1 mae rmse r2 silhouette confusion report" },
      { href: "/docs/model-selection", label: "Model Selection", icon: Crosshair, keywords: "cross validate gridsearch compare models tuning cv" },
    ],
  },
  {
    label: "Data & Ecosystem",
    items: [
      { href: "/docs/datasets", label: "Datasets", icon: Database, keywords: "iris penguins titanic sp500 course completion polars backend" },
      { href: "/docs/integration", label: "ViewX & StatsLibX", icon: EcosystemIcon, keywords: "payload to_html to_report presentation diagnostics statslibx viewx ecosystem" },
    ],
  },
];

export function DocSidebar() {
  const pathname = usePathname();
  const [query, setQuery] = useState("");

  const filteredSections = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sections;
    return sections
      .map((group) => ({
        ...group,
        items: group.items.filter(
          (item) =>
            item.label.toLowerCase().includes(q) ||
            item.keywords.toLowerCase().includes(q)
        ),
      }))
      .filter((group) => group.items.length > 0);
  }, [query]);

  const isSearching = query.trim().length > 0;
  const noResults = isSearching && filteredSections.length === 0;

  return (
    <aside className="w-60 shrink-0 hidden lg:block">
      <div className="sticky top-20 py-6 pr-4 space-y-6">
        <BrandLogo size={24} href="/" className="px-3 mb-2" />

        <div className="relative px-3">
          <Search className="absolute left-6 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search docs..."
            className="w-full bg-black/20 border border-border rounded-lg pl-8 pr-3 py-1.5 text-sm text-white placeholder-muted outline-none focus:border-accent/50 focus:bg-black/30 transition-colors"
          />
        </div>

        {noResults && (
          <p className="px-3 text-sm text-muted italic">No results for &quot;{query}&quot;</p>
        )}

        {filteredSections.map((group) => (
          <div key={group.label}>
            <h4 className="font-syne text-xs font-semibold text-muted uppercase tracking-widest mb-2 px-3">
              {group.label}
            </h4>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center gap-2.5 text-sm px-3 py-1.5 rounded-lg transition-colors no-underline ${
                      active
                        ? "text-accent bg-accent/10 font-medium"
                        : "text-muted hover:text-white hover:bg-white/5"
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
