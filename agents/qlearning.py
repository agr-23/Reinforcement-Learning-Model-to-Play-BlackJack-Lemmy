"""
Tabular Q-learning baseline for BlackjackEnv (US rules, H17, no surrender).
- Actions: 0=Hit, 1=Stand, 2=Double, 3=Split (the agent will only choose legal actions).
- Tabular state (compact): (player_total, usable_ace, dealer_up, can_double, can_split)
  Note: deliberately NOT using count/TC to keep the table manageable.

Usage:
  # Train
  python agents/qlearning.py train --episodes 200000 --seed 7 --save models/qtable.pkl

  # Evaluate (greedy policy)
  python agents/qlearning.py eval --episodes 20000 --seed 123 --load models/qtable.pkl
"""

import os
import sys
import math
import time
import pickle
import random
import argparse
from collections import defaultdict


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Import environment
from src.env.blackjack_env import (
    BlackjackEnv, Rules,
    ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT
)

ActionList = (ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT)

# ---------------------------
# Discrete state (tabular key)
# ---------------------------
def state_key(obs: dict) -> tuple:
    """
    Reduce continuous environment state to a discrete key for the Q-table.
    """
    return (
        int(obs["player_total"]),     # 0..31 (normally <= 31)
        int(obs["usable_ace"]),       # 0/1
        int(obs["dealer_up"]),        # 2..11 (11 = Ace)
        int(obs["can_double"]),       # 0/1
        int(obs["can_split"]),        # 0/1
    )

# ---------------------------
# Tabular Q-learning agent
# ---------------------------
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=5e-6, seed=None):
        """
        Epsilon-greedy policy with linear decay per step.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = random.Random(seed)
        # Q[(state_tuple)][action] -> float
        self.Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self._steps = 0

    def policy(self, obs: dict, legal_actions) -> int:
        """Epsilon-greedy policy restricted to legal actions."""
        self._steps += 1
        # Epsilon decay
        if self.eps > self.eps_end:
            self.eps = max(self.eps_end, self.eps - self.eps_decay)

        s = state_key(obs)
        qvals = self.Q[s]

        # Random exploration
        if self.rng.random() < self.eps:
            return self.rng.choice(legal_actions)

        # Greedy action among legal ones
        best_a, best_q = None, -1e9
        for a in legal_actions:
            if qvals[a] > best_q:
                best_q = qvals[a]
                best_a = a
        return best_a

    def update(self, s, a, r, sp, legal_next, done: bool):
        """Standard Q-learning update rule."""
        qsa = self.Q[s][a]
        if done or not legal_next:
            target = r
        else:
            max_next = max(self.Q[sp][an] for an in legal_next)
            target = r + self.gamma * max_next
        self.Q[s][a] = qsa + self.alpha * (target - qsa)

    # Save / load utilities
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0], d)

# ---------------------------
# Training loop
# ---------------------------
def train(args):
    seed = args.seed if args.seed is not None else 7
    env = BlackjackEnv(Rules(n_decks=6, h17=True, peek=True), seed=seed)
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma,
                           eps_start=args.eps_start, eps_end=args.eps_end,
                           eps_decay=args.eps_decay, seed=seed)

    episodes = args.episodes
    log_every = max(1000, episodes // 100)

    total_reward = 0.0
    wins = losses = pushes = 0

    t0 = time.time()
    for ep in range(1, episodes + 1):
        obs, r, done, info = env.reset()
        s = state_key(obs)

        ep_return = 0.0  # cumulative reward across subhands
        # Play until all subhands are resolved
        while not done:
            legal = env.available_actions()
            a = agent.policy(obs, legal)

            obs_next, r, done, _ = env.step(a)
            sp = state_key(obs_next)
            legal_next = env.available_actions() if not done else []

            agent.update(s, a, r, sp, legal_next, done)

            ep_return += r
            obs, s = obs_next, sp

        total_reward += ep_return
        # Classify outcome: win / loss / push
        if ep_return > 1e-12: wins += 1
        elif ep_return < -1e-12: losses += 1
        else: pushes += 1

        if ep % log_every == 0:
            wr = wins / ep
            avg_ret = total_reward / ep
            print(f"[train] ep={ep}/{episodes} | winrate={wr:.3f} | avg_return={avg_ret:.3f} | eps={agent.eps:.3f}")

    # Save Q-table
    if args.save:
        agent.save(args.save)
        print("Saved Q-table to", args.save)

# ---------------------------
# Greedy evaluation
# ---------------------------
def evaluate(args):
    seed = args.seed if args.seed is not None else 123
    env = BlackjackEnv(Rules(n_decks=6, h17=True, peek=True), seed=seed)
    agent = QLearningAgent(alpha=0.0, gamma=1.0, eps_start=0.0, eps_end=0.0, eps_decay=0.0, seed=seed)
    assert args.load and os.path.exists(args.load), f"Q-table not found: {args.load}"
    agent.load(args.load)

    episodes = args.episodes
    total_reward = 0.0
    wins = losses = pushes = 0

    for ep in range(1, episodes + 1):
        obs, r, done, info = env.reset()
        ep_return = 0.0
        while not done:
            legal = env.available_actions()
            # Pure greedy (eps=0)
            s = state_key(obs)
            qvals = agent.Q[s]
            # Choose best legal action
            a = max(legal, key=lambda aa: qvals[aa])
            obs, r, done, _ = env.step(a)
            ep_return += r

        total_reward += ep_return
        if ep_return > 1e-12: wins += 1
        elif ep_return < -1e-12: losses += 1
        else: pushes += 1

    print(f"[eval] episodes={episodes} | winrate={wins/episodes:.3f} | "
          f"avg_return={total_reward/episodes:.3f} | W/L/P={wins}/{losses}/{pushes}")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_t = sub.add_parser("train", help="Train Q-learning")
    ap_t.add_argument("--episodes", type=int, default=200_000)
    ap_t.add_argument("--alpha", type=float, default=0.1)
    ap_t.add_argument("--gamma", type=float, default=0.99)
    ap_t.add_argument("--eps-start", type=float, default=1.0)
    ap_t.add_argument("--eps-end", type=float, default=0.05)
    ap_t.add_argument("--eps-decay", type=float, default=5e-6)
    ap_t.add_argument("--seed", type=int, default=7)
    ap_t.add_argument("--save", type=str, default="models/qtable.pkl")

    ap_e = sub.add_parser("eval", help="Evaluate greedy policy")
    ap_e.add_argument("--episodes", type=int, default=20_000)
    ap_e.add_argument("--seed", type=int, default=123)
    ap_e.add_argument("--load", type=str, required=True)

    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()