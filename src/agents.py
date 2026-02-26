from __future__ import annotations
import numpy as np


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.15, gamma: float = 0.98,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995,
                 seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def act(self, state: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def learn(self, s: int, a: int, r: float, s2: int, done: bool):
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
