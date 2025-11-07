"""
Enhanced DQN Trainer for Blackjack (Lemmy) – Masked Double DQN
- Enmascara acciones ilegales (Double/Split) en el TARGET con info de next_state
- Double DQN + Huber loss
- Funciona con 1 mazo (curriculum) y sin last_drawn_cards() en el env
- Heurística Hi-Lo opcional para evaluación
- --- Integrado: Estrategia básica como política mixta durante entrenamiento ---
"""

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


# ----------------- Hi-Lo helpers -----------------
def hilo_count(card):
    if card in [2, 3, 4, 5, 6]:
        return +1
    elif card in [10, 11]:  # 10,J,Q,K,As(11)
        return -1
    else:
        return 0


def update_hilo_state(obs, running_count, decks_remaining):
    true_count = running_count / max(1, decks_remaining) if decks_remaining > 0 else running_count
    obs["true_count"] = float(np.clip(true_count, -10, 10))
    return obs


# ----------------- Estrategia básica -----------------
def basic_strategy(obs):
    pt = obs.get("player_total", 0)
    dealer_up = obs.get("dealer_up", 2)
    usable_ace = obs.get("usable_ace", 0)
    can_double = obs.get("can_double", 0)
    can_split = obs.get("can_split", 0)

    if can_split and pt in [20, 22]:  # pares 10,10 o A,A
        return 3
    if pt >= 17:
        return 1
    if pt <= 11 and can_double:
        return 2
    return 0


# ----------------- Obs -> vector -----------------
def obs_to_tensor(obs):
    player_total    = float(obs.get("player_total", 0)) / 21.0
    usable_ace      = float(obs.get("usable_ace", 0))
    dealer_up       = (float(obs.get("dealer_up", 0)) - 2) / 9.0
    true_count      = np.tanh(float(obs.get("true_count", 0)) / 5.0)
    cards_remaining = float(obs.get("cards_remaining", 0)) / 312.0
    hand_index      = float(obs.get("hand_index", 0)) / 4.0
    num_hands       = float(obs.get("num_hands", 1)) / 4.0
    can_double      = float(obs.get("can_double", 0))
    can_split       = float(obs.get("can_split", 0))
    episode_stage   = float(obs.get("episode_stage", 0)) / 3.0
    return np.array(
        [player_total, usable_ace, dealer_up, true_count, cards_remaining,
         hand_index, num_hands, can_double, can_split, episode_stage],
        dtype=np.float32
    )


# ----------------- Heurística baseline -----------------
def heuristic_hilo_policy(obs):
    pt = obs["player_total"]
    tc = obs.get("true_count", 0.0)
    can_double = obs.get("can_double", 0)
    if pt >= 17: a = 1
    elif pt <= 11: a = 0
    else:
        a = 1 if (tc >= 3 and pt >= 12) else 0
    if pt in [10, 11] and tc > 1 and can_double:
        a = 2
    return a


# ----------------- Evaluación -----------------
def evaluate_policy(env, net, episodes=1000, seed=0, use_heuristic=False):
    wins = losses = pushes = 0
    total_reward = 0.0
    net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, r, done, _ = env.reset()
            ep_return = 0.0
            running_count = 0
            while not done:
                legal = env.available_actions()
                if not legal:
                    done = True
                    break

                if hasattr(env, "last_drawn_cards"):
                    cards = env.last_drawn_cards()
                    if cards:
                        running_count += sum(hilo_count(c) for c in cards)
                decks_rem = obs.get("cards_remaining", 312) / 52.0
                obs = update_hilo_state(obs, running_count, decks_rem)

                s = torch.from_numpy(obs_to_tensor(obs)).to(DEVICE).unsqueeze(0)
                if use_heuristic:
                    a = heuristic_hilo_policy(obs)
                else:
                    q = net(s).squeeze(0)
                    mask = torch.full_like(q, -1e9)
                    mask[legal] = 0.0
                    a = int((q + mask).argmax().item())

                obs, r, done, _ = env.step(a)
                ep_return += r
            total_reward += ep_return
            if ep_return > 0: wins += 1
            elif ep_return < 0: losses += 1
            else: pushes += 1

    winrate = wins / episodes
    print(f"\n[Evaluation] Episodes={episodes}")
    print(f"Win rate: {winrate:.3f} | Draw: {pushes/episodes:.3f} | Loss: {losses/episodes:.3f}")
    print(f"Avg return: {total_reward/episodes:.3f}")
    return wins, losses, pushes, total_reward


# ----------------- Carga segura de checkpoint -----------------
def load_checkpoint_safe(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in ckpt["policy_state"].items()
                       if k in model_dict and v.size() == model_dict[k].size()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"[INFO] Cargadas {len(pretrained_dict)} capas del checkpoint. "
          f"Las demás inicializadas desde cero.")


# ----------------- Entrenamiento -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-update", type=int, default=2000)
    parser.add_argument("--start-train", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--save-dir", type=str, default="models/dqn_runs")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load", type=str)
    parser.add_argument("--use-heuristic", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True); os.makedirs("logs", exist_ok=True)
    log_path = Path("logs/dqn_results.csv")

    env = BlackjackEnv(Rules(n_decks=1, h17=True, peek=True), seed=args.seed)
    state_dim, action_dim = 10, 4

    policy_net = DQNNet(state_dim, action_dim).to(DEVICE)
    target_net = DQNNet(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)

    global_steps = 0
    eps = args.eps_start
    start_episode = 0

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    if args.load:
        load_checkpoint_safe(policy_net, args.load)
        target_net.load_state_dict(policy_net.state_dict())
        ckpt = torch.load(args.load, map_location=DEVICE)

        # Ignorar optimizer_state si es incompatible
        if "optimizer_state" in ckpt:
            print("[INFO] Ignorando optimizer_state del checkpoint (arquitectura incompatible)")

        replay.load(ckpt.get("replay", None))
        global_steps = ckpt.get("global_steps", 0)
        eps = ckpt.get("eps", args.eps_start)
        start_episode = ckpt.get("episode", 0)
        print(f"[INFO] Loaded checkpoint: {args.load} | steps={global_steps:,} | eps={eps:.3f}")

    if args.eval_only:
        print("[Eval-only]")
        wins, losses, pushes, total_reward = evaluate_policy(
            env, policy_net, episodes=args.eval_episodes, seed=args.seed, use_heuristic=args.use_heuristic
        )
        print(f"Results -> Win={wins/args.eval_episodes:.3f} | AvgRet={total_reward/args.eval_episodes:.3f}")
        return

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["global_steps", "episode", "winrate", "avg_return"])

    # ----------------- Loop de entrenamiento -----------------
    for ep in range(start_episode + 1, args.episodes + 1):
        obs, r, done, _ = env.reset()
        running_count = 0
        s = obs_to_tensor(obs)
        ep_return = 0.0

        while not done:
            legal = env.available_actions()
            if not legal:
                done = True
                break

            if hasattr(env, "last_drawn_cards"):
                cards = env.last_drawn_cards()
                if cards:
                    running_count += sum(hilo_count(c) for c in cards)
            decks_rem = obs.get("cards_remaining", 312) / 52.0
            obs = update_hilo_state(obs, running_count, decks_rem)
            s = obs_to_tensor(obs)

            eps = max(args.eps_end,
                      args.eps_start - (global_steps / args.eps_decay_steps) * (args.eps_start - args.eps_end))

            p_policy = 0.8
            if random.random() < eps:
                a = random.choice(legal)
            else:
                if random.random() < p_policy:
                    s_t = torch.from_numpy(s).to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        qvals = policy_net(s_t).squeeze(0)
                        mask = torch.full_like(qvals, -1e9)
                        mask[legal] = 0.0
                        a = int((qvals + mask).argmax().item())
                else:
                    a_basic = basic_strategy(obs)
                    if a_basic in legal:
                        a = a_basic
                    else:
                        a = random.choice(legal)

            obs_next, r, done, _ = env.step(a)
            s_next = obs_to_tensor(obs_next)
            replay.push(s, a, r, s_next, done)
            s = s_next
            obs = obs_next
            ep_return += r
            global_steps += 1

            if len(replay) >= args.start_train and len(replay) >= args.batch_size:
                batch = replay.sample(args.batch_size)
                states = torch.from_numpy(np.vstack([b[0] for b in batch])).to(DEVICE)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=DEVICE)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
                next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).to(DEVICE)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=DEVICE)

                q_sa = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    can_double_next = (next_states[:, 7] > 0.5).float().unsqueeze(1)
                    can_split_next  = (next_states[:, 8] > 0.5).float().unsqueeze(1)

                    B = next_states.size(0)
                    mask_next = torch.zeros((B, 4), device=DEVICE)
                    mask_next[:, 2] = torch.where(can_double_next.squeeze(1) > 0.5,
                                                  torch.tensor(0.0, device=DEVICE),
                                                  torch.tensor(-1e9, device=DEVICE))
                    mask_next[:, 3] = torch.where(can_split_next.squeeze(1) > 0.5,
                                                  torch.tensor(0.0, device=DEVICE),
                                                  torch.tensor(-1e9, device=DEVICE))

                    q_next_policy = policy_net(next_states) + mask_next
                    next_actions = q_next_policy.argmax(1)

                    q_next_target = target_net(next_states) + mask_next
                    next_q = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target = rewards + args.gamma * next_q * (1.0 - dones)

                loss = nn.SmoothL1Loss()(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                if global_steps % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if global_steps % args.eval_every == 0:
                wins, losses, pushes, total_reward_eval = evaluate_policy(
                    env, policy_net, episodes=args.eval_episodes, seed=args.seed
                )
                winrate = wins / args.eval_episodes
                avg_ret = total_reward_eval / args.eval_episodes
                print(f"[Eval] steps={global_steps:,} | win={winrate:.3f} | avg_return={avg_ret:.3f} | eps={eps:.3f}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([global_steps, ep, winrate, avg_ret])
                ckpt = {
                    "policy_state": policy_net.state_dict(),
                    "target_state": target_net.state_dict(),
                    "eps": eps,
                    "global_steps": global_steps,
                    "replay": replay.save_state(),
                    "episode": ep,
                }
                torch.save(ckpt, os.path.join(args.save_dir, "checkpoint_latest.pth"))

        if ep % 100 == 0:
            print(f"Episode {ep:,}/{args.episodes:,} | Return={ep_return:.2f} | eps={eps:.3f}")

    print("Training finished")
    torch.save(policy_net.state_dict(), os.path.join(args.save_dir, "dqn_final.pth"))


if __name__ == "__main__":
    main()
