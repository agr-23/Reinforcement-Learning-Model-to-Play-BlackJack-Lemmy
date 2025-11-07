"""
Blackjack DQN Model Comparison Tool

This script provides functionality to evaluate and compare different DQN model checkpoints
in the Blackjack environment. It's particularly useful for comparing the performance of
different training stages (pretrained, fine-tuned, and merged models).

Key Features:
- Supports multiple checkpoint formats (state_dict, policy_state)
- Evaluates models using consistent environment settings
- Calculates key performance metrics:
  * Win rate (percentage of winning hands)
  * Average reward per episode
- Handles state preprocessing identically to training

Default Evaluation Setup:
- Tests each model over 2000 episodes
- Compares three model variants:
  1. Pretrained model (from supervised learning)
  2. Fine-tuned model (from RL training)
  3. Merged model (combination of both)

Usage:
    python agents/compare_models.py

Output Example:
    Evaluating models...
    
    Evaluating pretrain...
    pretrain: win_rate=43.25% | avg_reward=-0.135
    
    Evaluating finetune...
    finetune: win_rate=45.80% | avg_reward=-0.084
    
    Evaluating merged...
    merged: win_rate=44.95% | avg_reward=-0.102

Note: The script automatically normalizes state values and handles the same
action space as during training (Hit, Stand, Double, Split).
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import torch.nn.functional as F
from models.dqn_net import DQNNet
from src.env.blackjack_env import BlackjackEnv

# === same preprocessing as training ===
def obs_to_tensor(obs):
    player_total = float(obs.get('player_total', 0)) / 21.0
    usable_ace = float(obs.get('usable_ace', 0))
    dealer_up = float(obs.get('dealer_up', 0) - 2) / 9.0
    true_count = float(obs.get('true_count', 0)) / 10.0
    cards_remaining = float(obs.get('cards_remaining', 0)) / 312.0
    hand_index = float(obs.get('hand_index', 0)) / 4.0
    num_hands = float(obs.get('num_hands', 1)) / 4.0
    can_double = float(obs.get('can_double', 0))
    can_split = float(obs.get('can_split', 0))
    return np.array(
        [player_total, usable_ace, dealer_up, true_count, cards_remaining,
         hand_index, num_hands, can_double, can_split],
        dtype=np.float32
    )

def evaluate_model(path, n_episodes=5000):
    env = BlackjackEnv()
    agent = DQNNet(state_dim=9, action_dim=4)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

# Detect different checkpoint formats
    if "state_dict" in ckpt:
     agent.load_state_dict(ckpt["state_dict"])
    elif "policy_state" in ckpt:  # extended for other possible formats
     agent.load_state_dict(ckpt["policy_state"])
    else:
     agent.load_state_dict(ckpt)

    agent.eval()

    rewards = []
    for _ in range(n_episodes):
        obs, _, done, _ = env.reset()
        s = obs_to_tensor(obs)
        total_reward = 0

        while not done:
            state_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(state_t)
                action = torch.argmax(q_values, dim=1).item()

            obs_next, r, done, _ = env.step(action)
            s = obs_to_tensor(obs_next)
            total_reward += r

        rewards.append(total_reward)

    rewards = np.array(rewards)
    win_rate = np.sum(rewards > 0) / len(rewards)
    mean_reward = np.mean(rewards)
    return win_rate, mean_reward


if __name__ == "__main__":
    models = {
        "pretrain": "models/dqn_pretrain/checkpoint_pretrain.pth",
        "finetune": "models/dqn_finetune/checkpoint_5540000.pth",
        "merged": "models/dqn_merged/checkpoint_merged.pth",
    }

    print("Evaluating models...\n")
    for name, path in models.items():
        print(f"Evaluating {name}...")
        win_rate, mean_reward = evaluate_model(path, n_episodes=2000)
        print(f"{name}: win_rate={win_rate*100:.2f}% | avg_reward={mean_reward:.3f}\n")