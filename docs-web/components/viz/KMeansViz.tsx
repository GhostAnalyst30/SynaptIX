"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Point = { x: number; y: number; cluster: number };
type Centroid = { x: number; y: number };

const COLORS = ["#a3ff45", "#5eead4", "#ffc857"];
const W = 480;
const H = 340;
const K = 3;
const N = 60;

function makePoints(): Point[] {
  const centers = [
    { x: 120, y: 100 },
    { x: 340, y: 90 },
    { x: 230, y: 250 },
  ];
  const points: Point[] = [];
  for (let i = 0; i < N; i++) {
    const c = centers[i % 3];
    points.push({
      x: c.x + (Math.random() - 0.5) * 130,
      y: c.y + (Math.random() - 0.5) * 110,
      cluster: -1,
    });
  }
  return points;
}

function randomCentroids(): Centroid[] {
  return Array.from({ length: K }, () => ({
    x: 60 + Math.random() * (W - 120),
    y: 50 + Math.random() * (H - 100),
  }));
}

/**
 * Visualización interactiva de K-Means: asignación de puntos al centroide
 * más cercano y recálculo de centroides, paso a paso.
 */
export default function KMeansViz() {
  const [points, setPoints] = useState<Point[]>([]);
  const [centroids, setCentroids] = useState<Centroid[]>([]);
  const [iteration, setIteration] = useState(0);
  const [running, setRunning] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const reset = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    setPoints(makePoints());
    setCentroids(randomCentroids());
    setIteration(0);
    setRunning(false);
  }, []);

  useEffect(() => {
    reset();
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [reset]);

  const step = useCallback(() => {
    setPoints((prev) =>
      prev.map((p) => {
        let best = 0;
        let bestDist = Infinity;
        centroids.forEach((c, index) => {
          const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
          if (d < bestDist) {
            bestDist = d;
            best = index;
          }
        });
        return { ...p, cluster: best };
      }),
    );
    setCentroids((prev) =>
      prev.map((c, index) => {
        const members = points.filter((p) => p.cluster === index);
        if (members.length === 0) return c;
        return {
          x: members.reduce((sum, p) => sum + p.x, 0) / members.length,
          y: members.reduce((sum, p) => sum + p.y, 0) / members.length,
        };
      }),
    );
    setIteration((i) => i + 1);
  }, [centroids, points]);

  const toggleRun = () => {
    if (running) {
      if (timer.current) clearInterval(timer.current);
      setRunning(false);
    } else {
      setRunning(true);
      timer.current = setInterval(step, 900);
    }
  };

  // Al llegar a convergencia visual, detener
  useEffect(() => {
    if (iteration >= 8 && timer.current) {
      clearInterval(timer.current);
      setRunning(false);
    }
  }, [iteration]);

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-ink-2">
      <div className="flex items-center justify-between border-b border-line px-4 py-2">
        <span className="font-mono text-xs text-dim">
          KMeans(n_clusters=3) — iteración {iteration}
        </span>
        <div className="flex gap-2">
          <button
            onClick={step}
            className="rounded border border-line px-2 py-0.5 font-mono text-xs text-mist hover:border-volt hover:text-volt"
          >
            paso
          </button>
          <button
            onClick={toggleRun}
            className="rounded border border-line px-2 py-0.5 font-mono text-xs text-mist hover:border-volt hover:text-volt"
          >
            {running ? "pausa" : "auto"}
          </button>
          <button
            onClick={reset}
            className="rounded border border-line px-2 py-0.5 font-mono text-xs text-mist hover:border-volt hover:text-volt"
          >
            reset
          </button>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {points.map((p, index) => (
          <circle
            key={index}
            cx={p.x}
            cy={p.y}
            r={4}
            fill={p.cluster >= 0 ? COLORS[p.cluster] : "#5f7a6c"}
            opacity={0.85}
            style={{ transition: "fill 0.5s" }}
          />
        ))}
        {centroids.map((c, index) => (
          <g
            key={index}
            style={{
              transform: `translate(${c.x}px, ${c.y}px)`,
              transition: "transform 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
            }}
          >
            <circle r={10} fill="none" stroke={COLORS[index]} strokeWidth={2} />
            <circle r={3} fill={COLORS[index]} />
          </g>
        ))}
      </svg>
    </div>
  );
}
