"use client";

import { useState } from "react";

/** Resaltador mínimo de Python basado en regex, sin dependencias. */
function highlightPython(code: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const tokens: { regex: RegExp; cls: string }[] = [
    { regex: /(#.*$)/gm, cls: "tok-com" },
    { regex: /("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"\n]*"|'[^'\n]*')/g, cls: "tok-str" },
    {
      regex:
        /\b(import|from|as|def|class|return|if|elif|else|for|while|in|not|and|or|None|True|False|with|try|except|raise|lambda|yield|pass|print)\b/g,
      cls: "tok-kw",
    },
    { regex: /\b(\d+\.?\d*)\b/g, cls: "tok-num" },
    { regex: /\b([A-Z][A-Za-z0-9_]*)\b/g, cls: "tok-cls" },
    { regex: /\b([a-z_][a-z0-9_]*)\s*(?=\()/g, cls: "tok-fn" },
  ];

  // Tokenización simple por posiciones para evitar anidar spans.
  type Span = { start: number; end: number; cls: string };
  const spans: Span[] = [];

  for (const { regex, cls } of tokens) {
    let match: RegExpExecArray | null;
    const re = new RegExp(regex.source, regex.flags);
    while ((match = re.exec(code)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      if (!spans.some((s) => start < s.end && end > s.start)) {
        spans.push({ start, end, cls });
      }
    }
  }

  spans.sort((a, b) => a.start - b.start);

  let html = "";
  let cursor = 0;
  for (const span of spans) {
    html += escape(code.slice(cursor, span.start));
    html += `<span class="${span.cls}">${escape(code.slice(span.start, span.end))}</span>`;
    cursor = span.end;
  }
  html += escape(code.slice(cursor));
  return html;
}

export default function CodeBlock({
  code,
  title,
}: {
  code: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group my-4 overflow-hidden rounded-lg border border-line bg-ink-2">
      <div className="flex items-center justify-between border-b border-line px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-rose/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber/70" />
          <span className="h-2.5 w-2.5 rounded-full bg-volt/70" />
          {title && (
            <span className="ml-3 font-mono text-xs text-dim">{title}</span>
          )}
        </div>
        <button
          onClick={copy}
          className="rounded border border-line px-2 py-0.5 font-mono text-xs text-mist transition-colors hover:border-volt hover:text-volt"
        >
          {copied ? "copiado ✓" : "copiar"}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-fog">
        <code dangerouslySetInnerHTML={{ __html: highlightPython(code) }} />
      </pre>
    </div>
  );
}
