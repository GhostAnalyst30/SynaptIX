"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();
  const inDocs = pathname.startsWith("/docs");

  return (
    <header className="fixed inset-x-0 top-0 z-40 border-b border-line bg-ink/80 backdrop-blur-md">
      <nav className="mx-auto flex h-14 max-w-7xl items-center justify-between px-5">
        <Link href="/" className="flex items-center gap-2">
          <span className="grid h-7 w-7 place-items-center rounded bg-volt font-mono text-sm font-bold text-ink">
            S
          </span>
          <span className="font-display text-lg font-bold tracking-tight">
            Synapt<span className="text-volt">IX</span>
          </span>
        </Link>
        <div className="flex items-center gap-6 font-mono text-sm">
          <Link
            href="/docs"
            className={`link-slide ${inDocs ? "text-volt" : "text-mist hover:text-fog"}`}
          >
            docs
          </Link>
          <a
            href="https://github.com/GhostAnalyst30/SynaptIX"
            target="_blank"
            rel="noreferrer"
            className="link-slide text-mist hover:text-fog"
          >
            github
          </a>
          <span className="hidden rounded border border-line px-2.5 py-1 text-xs text-dim sm:block">
            v0.1.7
          </span>
        </div>
      </nav>
    </header>
  );
}
