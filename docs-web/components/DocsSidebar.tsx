"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MODULES } from "@/lib/content";

export default function DocsSidebar() {
  const pathname = usePathname();

  const linkClass = (href: string) =>
    `block rounded px-3 py-1.5 font-mono text-[13px] transition-colors ${
      pathname === href
        ? "bg-ink-3 text-volt"
        : "text-mist hover:bg-ink-2 hover:text-fog"
    }`;

  return (
    <aside className="sticky top-24 hidden h-[calc(100vh-8rem)] w-56 shrink-0 overflow-y-auto lg:block">
      <p className="mb-2 px-3 font-mono text-[11px] tracking-widest text-dim uppercase">
        empezar
      </p>
      <Link href="/docs" className={linkClass("/docs")}>
        instalación
      </Link>

      <p className="mt-6 mb-2 px-3 font-mono text-[11px] tracking-widest text-dim uppercase">
        módulos
      </p>
      {MODULES.map((mod) => (
        <Link
          key={mod.slug}
          href={`/docs/${mod.slug}`}
          className={linkClass(`/docs/${mod.slug}`)}
        >
          {mod.name.replace("synaptix.", "")}
        </Link>
      ))}
    </aside>
  );
}
