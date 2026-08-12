"""
Entornos de aprendizaje por refuerzo.

Incluye ``GridWorld``, un entorno de rejilla clásico para demos y
enseñanza. La interfaz sigue el estilo de Gymnasium:

    state = env.reset()
    state, reward, done = env.step(action)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class GridWorld:
    """Entorno de rejilla 2D con obstáculos y una meta.

    El agente parte de ``start`` y debe llegar a ``goal`` evitando
    obstáculos. Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda.

    Recompensas:
        +10 al llegar a la meta, -1 por paso, -5 por chocar con un
        obstáculo o el borde (el agente no se mueve).

    Parameters
    ----------
    rows : int, default=4
        Filas de la rejilla.
    cols : int, default=4
        Columnas de la rejilla.
    start : tuple, default=(0, 0)
        Celda inicial ``(fila, columna)``.
    goal : tuple, optional
        Celda meta. Por defecto, la esquina inferior derecha.
    obstacles : list de tuplas, optional
        Celdas bloqueadas.
    max_steps : int, default=200
        Pasos máximos por episodio.

    Ejemplo
    -------
    >>> from synaptix.reinforcement import GridWorld, QLearningAgent
    >>> env = GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])
    >>> agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    >>> agent.train(env, episodes=500)
    >>> env.render(agent.policy())
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: Tuple[int, int] = (0, 0),
        goal: Optional[Tuple[int, int]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 200,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal if goal is not None else (rows - 1, cols - 1)
        self.obstacles = set(obstacles or [])
        self.max_steps = max_steps

        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("start y goal no pueden ser obstáculos")

        self.n_states = rows * cols
        self.n_actions = 4
        self._position = start
        self._steps = 0

    # ------------------------------------------------------------------

    def state_id(self, position: Tuple[int, int]) -> int:
        """Convierte una celda ``(fila, col)`` a un id de estado entero."""
        return position[0] * self.cols + position[1]

    def position_of(self, state: int) -> Tuple[int, int]:
        """Convierte un id de estado a celda ``(fila, col)``."""
        return divmod(state, self.cols)

    def reset(self) -> int:
        """Reinicia el episodio y devuelve el estado inicial."""
        self._position = self.start
        self._steps = 0
        return self.state_id(self._position)

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Ejecuta una acción.

        Parameters
        ----------
        action : int
            0=arriba, 1=derecha, 2=abajo, 3=izquierda.

        Returns
        -------
        tuple
            ``(nuevo_estado, recompensa, terminado)``.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Acción inválida: {action}")

        self._steps += 1
        d_row, d_col = self.ACTIONS[action]
        new_pos = (self._position[0] + d_row, self._position[1] + d_col)

        out_of_bounds = not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols)
        blocked = new_pos in self.obstacles

        if out_of_bounds or blocked:
            reward = -5.0
        else:
            self._position = new_pos
            reward = 10.0 if new_pos == self.goal else -1.0

        done = self._position == self.goal or self._steps >= self.max_steps
        return self.state_id(self._position), reward, done

    # ------------------------------------------------------------------

    def render(self, policy: Optional[np.ndarray] = None) -> None:
        """Imprime la rejilla en consola.

        Parameters
        ----------
        policy : ndarray, optional
            Vector de acciones por estado; si se da, dibuja flechas con
            la mejor acción de cada celda.
        """
        print()
        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                pos = (row, col)
                if pos == self.goal:
                    cells.append(" G ")
                elif pos in self.obstacles:
                    cells.append(" # ")
                elif pos == self.start and policy is None:
                    cells.append(" S ")
                elif policy is not None:
                    action = int(policy[self.state_id(pos)])
                    cells.append(f" {self.ACTION_NAMES[action]} ")
                else:
                    cells.append(" . ")
            print("|" + "|".join(cells) + "|")
        print()
