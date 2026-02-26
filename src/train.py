from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .agents import QLearningAgent
from .environments import GridworldBase, StepResult

@dataclass
class EpisodeLog:
    total_reward: float
    steps: int
    info: Dict

def train(agent: QLearningAgent, env: GridworldBase, episodes: int = 500) -> List[EpisodeLog]:
    logs: List[EpisodeLog] = []
    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        done = False
        steps = 0
        last_info = {}
        while not done:
            a = agent.act(s)
            res: StepResult = env.step(a)
            s2 = res.obs
            agent.learn(s, a, res.reward, s2, res.terminated)
            s = s2
            total += res.reward
            done = res.terminated
            steps += 1
            last_info = res.info
        agent.decay_epsilon()
        logs.append(EpisodeLog(total_reward=float(total), steps=steps, info=last_info))
    return logs

def moving_average(x: List[float], window: int = 50) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")
