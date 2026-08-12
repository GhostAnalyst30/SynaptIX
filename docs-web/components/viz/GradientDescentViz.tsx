"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const W = 480;
const H = 300;

// Función de pérdida de juguete y su derivada
const loss = (x: number) => 0.09 * (x - 5.4) ** 2 + 0.4 + 0.35 * Math.sin(x * 1.4);
const dLoss = (x: number) =>
  0.18 * (x - 5.4) + 0.49 * Math.cos(x * 1.4);

const toSvgX = (x: number) => (x / 10) * W;
const toSvgY = (y: number) => H - 36 - y * 70;

/**
 * Descenso de gradiente animado: una bola desciende por la curva de
 * pérdida siguiendo el gradiente, con learning rate ajustable.
 */
export default function GradientDescentViz() {
  const [x, setX] = useState(1.0);
  const [lr, setLr] = useState(0.35);
  const [steps, setSteps] = useState(0);
  const [running, setRunning] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const curve = Array.from({ length: 121 }, (_, i) => {
    const px = (i / 120) * 10;
    return `${toSvgX(px)},${toSvgY(loss(px))}`;
  }).join(" ");

  const stop = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    setRunning(false);
  }, []);

  const reset = useCallback(() => {
    stop();
    setX(1.0);
    setSteps(0);
  }, [stop]);

  const toggleRun = () => {
    if (running) {
      stop();
      return;
    }
    setRunning(true);
    timer.current = setInterval(() => {
      setX((prev) => {
        const next = prev - lr * dLoss(prev);
        return Math.max(0.2, Math.min(9.8, next));
      });
      setSteps((s) => s + 1);
    }, 350);
  };

  useEffect(() => {
    if (steps >= 40) stop();
  }, [steps, stop]);

  useEffect(() => () => stop(), [stop]);

  const gradient = dLoss(x);

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-ink-2">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2">
        <span className="font-mono text-xs text-dim">
          w -= lr * grad — paso {steps} | pérdida {loss(x).toFixed(3)}
        </span>
        <div className="flex items-center gap-2">
          <label className="font-mono text-xs text-mist">
            lr
            <input
              type="range"
              min={0.05}
              max={1.2}
              step={0.05}
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="ml-1 w-20 accent-[#a3ff45]"
            />
            <span className="ml-1 text-volt">{lr.toFixed(2)}</span>
          </label>
          <button
            onClick={toggleRun}
            className="rounded border border-line px-2 py-0.5 font-mono text-xs text-mist hover:border-volt hover:text-volt"
          >
            {running ? "pausa" : "entrenar"}
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
        {/* Curva de pérdida */}
        <polyline
          points={curve}
          fill="none"
          stroke="#5eead4"
          strokeWidth={2}
          opacity={0.8}
        />
        {/* Vector de gradiente */}
        <line
          x1={toSvgX(x)}
          y1={toSvgY(loss(x))}
          x2={toSvgX(x) - gradient * 90}
          y2={toSvgY(loss(x))}
          stroke="#ff6b81"
          strokeWidth={2}
          markerEnd="url(#arrow)"
        />
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6b81" />
          </marker>
        </defs>
        {/* Bola */}
        <circle
          cx={toSvgX(x)}
          cy={toSvgY(loss(x))}
          r={8}
          fill="#a3ff45"
          style={{ transition: "cx 0.3s ease, cy 0.3s ease" }}
        />
        <text
          x={toSvgX(x)}
          y={toSvgY(loss(x)) - 16}
          textAnchor="middle"
          fill="#e9f4ed"
          fontSize={11}
          fontFamily="monospace"
        >
          w={x.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}
