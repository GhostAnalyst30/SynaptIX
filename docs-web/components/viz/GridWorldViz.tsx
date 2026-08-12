"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const ROWS = 5;
const COLS = 5;
const OBSTACLES = new Set(["1,1", "2,3", "3,1"]);
const GOAL = [4, 4];
const CELL = 72;

type Pos = [number, number];

const ACTIONS: Pos[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];

/**
 * Agente Q-Learning simplificado corriendo en el navegador: explora el
 * GridWorld, choca con obstáculos y converge a la ruta óptima.
 */
export default function GridWorldViz() {
  const [agent, setAgent] = useState<Pos>([0, 0]);
  const [episode, setEpisode] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [running, setRunning] = useState(false);
  const [trail, setTrail] = useState<string[]>([]);
  const qTable = useRef<number[][]>(
    Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0]),
  );
  const pos = useRef<Pos>([0, 0]);
  const eps = useRef(1.0);
  const episodeCount = useRef(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stateId = (p: Pos) => p[0] * COLS + p[1];

  const stepOnce = useCallback(() => {
    const state = stateId(pos.current);
    // epsilon-greedy
    const q = qTable.current[state];
    const action =
      Math.random() < eps.current
        ? Math.floor(Math.random() * 4)
        : q.indexOf(Math.max(...q));

    const [dr, dc] = ACTIONS[action];
    const next: Pos = [pos.current[0] + dr, pos.current[1] + dc];
    const key = `${next[0]},${next[1]}`;

    let reward: number;
    let moved = true;
    if (
      next[0] < 0 ||
      next[0] >= ROWS ||
      next[1] < 0 ||
      next[1] >= COLS ||
      OBSTACLES.has(key)
    ) {
      reward = -5;
      moved = false;
    } else if (next[0] === GOAL[0] && next[1] === GOAL[1]) {
      reward = 10;
    } else {
      reward = -1;
    }

    const landing: Pos = moved ? next : pos.current;
    const nextState = stateId(landing);
    const bestNext = Math.max(...qTable.current[nextState]);
    const done = landing[0] === GOAL[0] && landing[1] === GOAL[1];

    qTable.current[state][action] +=
      0.2 * (reward + 0.95 * bestNext * (done ? 0 : 1) - qTable.current[state][action]);

    if (done) {
      pos.current = [0, 0];
      episodeCount.current += 1;
      eps.current = Math.max(0.05, eps.current * 0.9);
      setEpisode(episodeCount.current);
      setEpsilon(eps.current);
      setTrail([]);
    } else {
      pos.current = landing;
      setTrail((t) => [...t.slice(-12), `${landing[0]},${landing[1]}`]);
    }
    setAgent([...pos.current] as Pos);
  }, []);

  const toggleRun = () => {
    if (running) {
      if (timer.current) clearInterval(timer.current);
      setRunning(false);
    } else {
      setRunning(true);
      timer.current = setInterval(stepOnce, 90);
    }
  };

  const reset = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    qTable.current = Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0]);
    pos.current = [0, 0];
    eps.current = 1.0;
    episodeCount.current = 0;
    setAgent([0, 0]);
    setEpisode(0);
    setEpsilon(1.0);
    setTrail([]);
    setRunning(false);
  }, []);

  useEffect(() => () => {
    if (timer.current) clearInterval(timer.current);
  }, []);

  const arrows = ["↑", "→", "↓", "←"];

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-ink-2">
      <div className="flex items-center justify-between border-b border-line px-4 py-2">
        <span className="font-mono text-xs text-dim">
          QLearningAgent — episodio {episode} | ε = {epsilon.toFixed(2)}
        </span>
        <div className="flex gap-2">
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
      <svg viewBox={`0 0 ${COLS * CELL} ${ROWS * CELL}`} className="w-full">
        {Array.from({ length: ROWS }).map((_, r) =>
          Array.from({ length: COLS }).map((_, c) => {
            const key = `${r},${c}`;
            const isObstacle = OBSTACLES.has(key);
            const isGoal = r === GOAL[0] && c === GOAL[1];
            const inTrail = trail.includes(key);
            const q = qTable.current[r * COLS + c];
            const maxQ = Math.max(...q);
            const learned = maxQ > 0.5 && !isObstacle && !isGoal;

            return (
              <g key={key} transform={`translate(${c * CELL}, ${r * CELL})`}>
                <rect
                  width={CELL}
                  height={CELL}
                  fill={
                    isObstacle
                      ? "#16201c"
                      : isGoal
                        ? "rgba(163,255,69,0.16)"
                        : inTrail
                          ? "rgba(94,234,212,0.07)"
                          : "transparent"
                  }
                  stroke="#223129"
                />
                {isObstacle && (
                  <text
                    x={CELL / 2}
                    y={CELL / 2 + 5}
                    textAnchor="middle"
                    fill="#5f7a6c"
                    fontSize={18}
                  >
                    ▦
                  </text>
                )}
                {isGoal && (
                  <text
                    x={CELL / 2}
                    y={CELL / 2 + 6}
                    textAnchor="middle"
                    fill="#a3ff45"
                    fontSize={20}
                  >
                    ★
                  </text>
                )}
                {learned && (
                  <text
                    x={CELL / 2}
                    y={CELL / 2 + 5}
                    textAnchor="middle"
                    fill="#5eead4"
                    opacity={0.55}
                    fontSize={15}
                  >
                    {arrows[q.indexOf(maxQ)]}
                  </text>
                )}
              </g>
            );
          }),
        )}
        {/* Agente */}
        <circle
          cx={agent[1] * CELL + CELL / 2}
          cy={agent[0] * CELL + CELL / 2}
          r={13}
          fill="#a3ff45"
          style={{ transition: "cx 0.08s linear, cy 0.08s linear" }}
        />
      </svg>
      <p className="border-t border-line px-4 py-2 font-mono text-[11px] text-dim">
        las flechas muestran la política aprendida (mejor acción por celda)
      </p>
    </div>
  );
}
