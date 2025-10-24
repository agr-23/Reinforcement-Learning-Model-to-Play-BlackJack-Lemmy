"""
Train a DQN agent to play Blackjack using replay buffer and target network.
Supports resume, periodic evaluation, and CSV logging for plotting.
"""
# COMANDS FOR DEVELOPMENT TESTING:
# python agents/train_dqn.py --eval-only --load models/dqn_run/checkpoint_latest.pth -> EVAL ONLY WITHOUT EXPLORATION
# python agents/train_dqn.py --resume --load models/dqn_run/checkpoint_latest.pth -> RESUME TRAINING

import os
import argparse
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# --- Local imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.env.blackjack_env import BlackjackEnv, Rules
from models.dqn_net import DQNNet
from src.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Helper ===
def obs_to_tensor(obs):
    """Convert env observation dict to a normalized feature vector."""
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

# === Evaluation function ===
def evaluate_policy(env, net, episodes=1000, seed=0):
    rng = random.Random(seed)
    wins = losses = pushes = 0
    total_reward = 0.0

    net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, r, done, _ = env.reset()
            s = obs_to_tensor(obs)
            ep_return = 0.0

            while not done:
                legal = env.available_actions()
                s_t = torch.from_numpy(s).to(DEVICE).unsqueeze(0)
                qvals_t = net(s_t).squeeze(0)
                mask = torch.full_like(qvals_t, -1e9)
                mask[legal] = 0.0
                a = int((qvals_t + mask).argmax().item())
                obs_next, r, done, _ = env.step(a)
                s = obs_to_tensor(obs_next)
                ep_return += r

            total_reward += ep_return
            if ep_return > 0:
                wins += 1
            elif ep_return < 0:
                losses += 1
            else:
                pushes += 1
    print("\n=== GREEDY EVALUATION ===")
    print(f"Episodes: {episodes}")
    print(f"Win rate:  {wins / episodes:.3f}")
    print(f"Draw rate: {pushes / episodes:.3f}")
    print(f"Loss rate: {losses / episodes:.3f}")
    print(f"Avg return: {total_reward / episodes:.3f}")

    return wins, losses, pushes, total_reward

# === Main Training Loop ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-update", type=int, default=2000)
    parser.add_argument("--start-train", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--save-dir", type=str, default="models/dqn_runs", help="Directory to save checkpoints and logs")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=2_000_000)
    parser.add_argument("--eval-episodes", type=int, default=5000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load", type=str, help="Path to checkpoint to load")
    args = parser.parse_args()

    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Setup ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = Path("logs/dqn_results.csv")

    env = BlackjackEnv(Rules(n_decks=6, h17=True, peek=True), seed=args.seed)
    state_dim, action_dim = 9, 4

    policy_net = DQNNet(state_dim, action_dim).to(DEVICE)
    target_net = DQNNet(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)

    global_steps = 0
    eps = args.eps_start

    # --- Load checkpoint if needed ---
    if args.load:
      ckpt = torch.load(args.load, map_location=DEVICE, weights_only=False)
      policy_net.load_state_dict(ckpt["policy_state"])
      target_net.load_state_dict(ckpt.get("target_state", ckpt["policy_state"]))
      optimizer.load_state_dict(ckpt["optimizer_state"])
      replay.load(ckpt.get("replay", None))
      global_steps = ckpt.get("global_steps", 0)
      eps = ckpt.get("eps", args.eps_start)
      start_episode = ckpt.get("episode", 0)
      print(f"Loaded checkpoint: {args.load} | steps={global_steps:,} | eps={eps:.3f} | episode={start_episode:,}")

    # --- Eval only mode ---
    if args.eval_only:
      print("[DQN Eval-only] Loading checkpoint and running greedy evaluation...")
      eps = 0.0  # Greedy (no exploration)
      wins, losses, pushes, total_reward = evaluate_policy(env, policy_net, episodes=50000, seed=args.seed)
      print(f"Results: Wins={wins}, Losses={losses}, Pushes={pushes}, Avg Return={total_reward/50000:.3f}")
      return

    # --- Prepare logging ---
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_steps", "episode", "winrate", "avg_return"])

    # === Training Loop ===
    for ep in range(start_episode + 1, args.episodes + 1):
        obs, r, done, _ = env.reset()
        s = obs_to_tensor(obs)
        ep_return = 0.0

        while not done:
            legal = env.available_actions()
            if not legal:
                break

            eps = max(
                args.eps_end,
                args.eps_start - (global_steps / args.eps_decay_steps) * (args.eps_start - args.eps_end)
            )

            if random.random() < eps:
                a = random.choice(legal)
            else:
                s_t = torch.from_numpy(s).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    qvals_t = policy_net(s_t).squeeze(0)
                    mask = torch.full_like(qvals_t, -1e9)
                    mask[legal] = 0.0
                    a = int((qvals_t + mask).argmax().item())

            obs_next, r, done, _ = env.step(a)
            s_next = obs_to_tensor(obs_next)
            replay.push(s, a, r, s_next, done)
            s = s_next
            ep_return += r
            global_steps += 1

            # --- Train step ---
            if len(replay) >= args.start_train and len(replay) >= args.batch_size:
                batch = replay.sample(args.batch_size)
                states = torch.from_numpy(np.vstack([b[0] for b in batch])).to(DEVICE)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=DEVICE)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
                next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).to(DEVICE)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=DEVICE)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + args.gamma * next_q * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                if global_steps % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # --- Evaluate ---
            if global_steps % args.eval_every == 0:
                wins, losses, pushes, total_reward_eval = evaluate_policy(env, policy_net, episodes=args.eval_episodes, seed=args.seed)
                winrate = wins / args.eval_episodes
                avg_ret = total_reward_eval / args.eval_episodes
                print(f"[Eval] steps={global_steps:,} | winrate={winrate:.3f} | avg_return={avg_ret:.3f} | eps={eps:.3f}")

                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_steps, ep, winrate, avg_ret])

                ckpt = {
                    "policy_state": policy_net.state_dict(),
                    "target_state": target_net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "eps": eps,
                    "global_steps": global_steps,
                    "replay": replay.save_state()
                }
                torch.save(ckpt, os.path.join(args.save_dir, f"checkpoint_{global_steps}.pth"))
                torch.save(ckpt, os.path.join(args.save_dir, "checkpoint_latest.pth"))

        if ep % 100 == 0:
            print(f"Episode {ep:,}/{args.episodes:,} | Return={ep_return:.2f} | Buffer={len(replay)} | eps={eps:.3f}")

    # --- Final save ---
    ckpt = {
        "policy_state": policy_net.state_dict(),
        "target_state": target_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "eps": eps,
        "global_steps": global_steps,
        "replay": replay.save_state(),
        "episodes": ep,
    }
    torch.save(ckpt, os.path.join(args.save_dir, "checkpoint_final.pth"))
    print("Training finished — all checkpoints saved.")

if __name__ == "__main__":
    main()