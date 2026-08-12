export default function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-5 py-8 sm:flex-row">
        <p className="font-mono text-xs text-dim">
          SynaptIX v0.1.7 — MIT — Emmanuel Ascendra
        </p>
        <p className="font-mono text-xs text-dim">
          hecho con Python, NumPy y café
        </p>
      </div>
    </footer>
  );
}
