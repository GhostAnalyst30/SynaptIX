"""
synaptix.reinforcement - Aprendizaje por refuerzo.

Entornos:
    GridWorld : rejilla 2D con obstáculos y meta.

Agentes:
    QLearningAgent : Q-Learning tabular (off-policy).
    SARSAAgent     : SARSA tabular (on-policy).
    DQNAgent       : Deep Q-Network con Keras (requiere tensorflow).

Ejemplo
-------
>>> from synaptix.reinforcement import GridWorld, QLearningAgent
>>> env = GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])
>>> agent = QLearningAgent(env.n_states, env.n_actions)
>>> agent.train(env, episodes=500)
>>> agent.plot_rewards()
>>> env.render(agent.policy())  # política aprendida con flechas
"""

from .agents import DQNAgent, QLearningAgent, SARSAAgent
from .environments import GridWorld

__all__ = ["GridWorld", "QLearningAgent", "SARSAAgent", "DQNAgent"]
