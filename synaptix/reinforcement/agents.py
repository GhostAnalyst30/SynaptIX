"""
Agentes de aprendizaje por refuerzo.

- ``QLearningAgent`` : Q-Learning tabular (off-policy), desde cero con NumPy.
- ``SARSAAgent``     : SARSA tabular (on-policy), desde cero con NumPy.
- ``DQNAgent``       : Deep Q-Network con Keras (requiere tensorflow).

Ejemplo
-------
>>> from synaptix.reinforcement import GridWorld, QLearningAgent
>>> env = GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])
>>> agent = QLearningAgent(env.n_states, env.n_actions)
>>> rewards = agent.train(env, episodes=500)
>>> agent.plot_rewards()
>>> env.render(agent.policy())
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class _TabularAgent:
    """Base para agentes tabulares con política epsilon-greedy.

    Parameters
    ----------
    n_states : int
        Número de estados del entorno.
    n_actions : int
        Número de acciones posibles.
    alpha : float, default=0.1
        Tasa de aprendizaje.
    gamma : float, default=0.99
        Factor de descuento de recompensas futuras.
    epsilon : float, default=1.0
        Probabilidad inicial de explorar (acción aleatoria).
    epsilon_min : float, default=0.01
        Valor mínimo de epsilon.
    epsilon_decay : float, default=0.995
        Factor multiplicativo de decaimiento por episodio.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions))
        self.rewards_history: List[float] = []

    def act(self, state: int, greedy: bool = False) -> int:
        """Elige una acción con política epsilon-greedy.

        Parameters
        ----------
        state : int
            Estado actual.
        greedy : bool, default=False
            Si es True, siempre elige la mejor acción conocida.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def policy(self) -> np.ndarray:
        """Mejor acción aprendida para cada estado."""
        return np.argmax(self.q_table, axis=1)

    def plot_rewards(self, window: int = 20) -> None:
        """Grafica la recompensa por episodio y su media móvil.

        Parameters
        ----------
        window : int, default=20
            Tamaño de ventana de la media móvil.
        """
        rewards = np.array(self.rewards_history)
        plt.figure(figsize=(8, 4))
        plt.plot(rewards, alpha=0.3, label="Recompensa")
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window - 1, len(rewards)),
                moving_avg,
                color="crimson",
                label=f"Media móvil ({window})",
            )
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        plt.title(f"Entrenamiento: {type(self).__name__}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgent(_TabularAgent):
    """Agente Q-Learning tabular (off-policy).

    Actualización::

        Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    """

    def train(self, env, episodes: int = 500, verbose: bool = False) -> List[float]:
        """Entrena el agente en un entorno.

        Parameters
        ----------
        env : objeto entorno
            Debe implementar ``reset() -> state`` y
            ``step(action) -> (state, reward, done)``.
        episodes : int, default=500
            Número de episodios de entrenamiento.
        verbose : bool, default=False
            Imprime progreso cada 100 episodios.

        Returns
        -------
        list de float
            Recompensa total por episodio.
        """
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)

                best_next = np.max(self.q_table[next_state])
                target = reward + self.gamma * best_next * (not done)
                self.q_table[state, action] += self.alpha * (
                    target - self.q_table[state, action]
                )

                state = next_state
                total_reward += reward

            self.rewards_history.append(total_reward)
            self._decay_epsilon()

            if verbose and (episode + 1) % 100 == 0:
                avg = np.mean(self.rewards_history[-100:])
                print(
                    f"Episodio {episode + 1}/{episodes} | "
                    f"recompensa media (últ. 100): {avg:.2f} | epsilon: {self.epsilon:.3f}"
                )

        return self.rewards_history


class SARSAAgent(_TabularAgent):
    """Agente SARSA tabular (on-policy).

    Actualización::

        Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))

    donde ``a'`` es la acción realmente tomada en ``s'``.
    """

    def train(self, env, episodes: int = 500, verbose: bool = False) -> List[float]:
        """Entrena el agente (misma interfaz que :meth:`QLearningAgent.train`)."""
        for episode in range(episodes):
            state = env.reset()
            action = self.act(state)
            total_reward = 0.0
            done = False

            while not done:
                next_state, reward, done = env.step(action)
                next_action = self.act(next_state)

                target = reward + self.gamma * self.q_table[next_state, next_action] * (
                    not done
                )
                self.q_table[state, action] += self.alpha * (
                    target - self.q_table[state, action]
                )

                state, action = next_state, next_action
                total_reward += reward

            self.rewards_history.append(total_reward)
            self._decay_epsilon()

            if verbose and (episode + 1) % 100 == 0:
                avg = np.mean(self.rewards_history[-100:])
                print(
                    f"Episodio {episode + 1}/{episodes} | "
                    f"recompensa media (últ. 100): {avg:.2f} | epsilon: {self.epsilon:.3f}"
                )

        return self.rewards_history


class DQNAgent:
    """Deep Q-Network: Q-Learning con red neuronal (requiere tensorflow).

    Para entornos con estados continuos o espacios grandes donde una
    tabla Q no es viable. Usa experience replay y una red objetivo.

    Parameters
    ----------
    state_dim : int
        Dimensión del vector de estado.
    n_actions : int
        Número de acciones.
    hidden_layers : tuple, default=(64, 64)
        Neuronas por capa oculta.
    gamma : float, default=0.99
        Factor de descuento.
    epsilon_decay : float, default=0.995
        Decaimiento de epsilon por episodio.
    buffer_size : int, default=10000
        Tamaño del buffer de experiencia.
    batch_size : int, default=64
        Tamaño de lote para el replay.

    Ejemplo
    -------
    >>> agent = DQNAgent(state_dim=4, n_actions=2)
    >>> # entrenar con un entorno cuyo estado sea un vector de 4 valores
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_layers: tuple = (64, 64),
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_every: int = 10,
    ):
        try:
            from keras.layers import Dense
            from keras.models import Sequential
            from keras.optimizers import Adam
        except ImportError as error:
            raise ImportError(
                "DQNAgent requiere tensorflow/keras. Instala con: "
                "pip install synaptix[dl]"
            ) from error

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.memory: deque = deque(maxlen=buffer_size)
        self.rewards_history: List[float] = []

        def build_network():
            model = Sequential()
            model.add(Dense(hidden_layers[0], activation="relu", input_dim=state_dim))
            for units in hidden_layers[1:]:
                model.add(Dense(units, activation="relu"))
            model.add(Dense(n_actions, activation="linear"))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
            return model

        self.model = build_network()
        self.target_model = build_network()
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Elige una acción con política epsilon-greedy."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q_values = self.model.predict(np.asarray(state).reshape(1, -1), verbose=0)
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done) -> None:
        """Guarda una transición en el buffer de experiencia."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> None:
        """Entrena la red con un lote aleatorio del buffer."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch], dtype=bool)

        q_current = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        targets = q_current.copy()
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(
            q_next, axis=1
        ) * (~dones)

        self.model.fit(states, targets, epochs=1, verbose=0)

    def train(self, env, episodes: int = 200, verbose: bool = True) -> List[float]:
        """Entrena el agente en un entorno con estados vectoriales.

        El entorno debe implementar ``reset() -> state`` y
        ``step(action) -> (state, reward, done)`` donde ``state`` es un
        vector de dimensión ``state_dim``.
        """
        for episode in range(episodes):
            state = np.asarray(env.reset(), dtype=float).ravel()
            total_reward = 0.0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                next_state = np.asarray(next_state, dtype=float).ravel()

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward

            self.rewards_history.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % self.target_update_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            if verbose and (episode + 1) % 10 == 0:
                avg = np.mean(self.rewards_history[-10:])
                print(
                    f"Episodio {episode + 1}/{episodes} | "
                    f"recompensa media (últ. 10): {avg:.2f} | epsilon: {self.epsilon:.3f}"
                )

        return self.rewards_history

    def plot_rewards(self, window: int = 10) -> None:
        """Grafica la recompensa por episodio y su media móvil."""
        rewards = np.array(self.rewards_history)
        plt.figure(figsize=(8, 4))
        plt.plot(rewards, alpha=0.3, label="Recompensa")
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window - 1, len(rewards)),
                moving_avg,
                color="crimson",
                label=f"Media móvil ({window})",
            )
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa total")
        plt.title("Entrenamiento: DQN")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
