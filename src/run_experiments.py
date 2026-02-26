from __future__ import annotations
import os
import matplotlib.pyplot as plt

from .environments import make_env
from .agents import QLearningAgent
from .train import train, moving_average

def plot_rewards(rewards, title, out_path):
    ma = moving_average(rewards, window=50)
    plt.figure()
    plt.plot(rewards, label="Episode Reward")
    if len(ma) > 0:
        plt.plot(range(len(rewards) - len(ma), len(rewards)), ma, label="Moving Avg (50)")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run(env_name: str, episodes: int = 500):
    env = make_env(env_name)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    logs = train(agent, env, episodes=episodes)
    rewards = [l.total_reward for l in logs]
    os.makedirs("images", exist_ok=True)
    img_path = os.path.join("images", f"{env_name}_rewards.png")
    plot_rewards(rewards, f"{env_name} - Training Rewards", img_path)
    print(f"Saved plot: {img_path}")

if __name__ == "__main__":
    for name in ["interrupt_safe", "interrupt_unsafe", "side_effects", "reward_gaming"]:
        run(name)
