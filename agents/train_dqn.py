import os
import argparse
import random
import csv
import numpy as np
import numpy._core.multiarray
from torch.serialization import add_safe_globals

add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype
])
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import pandas as pd
import ast
import torch

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


# ----------------- Estrategia básica MEJORADA -----------------
def basic_strategy(obs):
    """
    Estrategia básica mejorada para Blackjack (H17, DAS, No Surrender)
    Basada en tablas estándar: https://wizardofodds.com/games/blackjack/strategy/4-decks/
    """
    pt = obs.get("player_total", 0)
    dealer_up = obs.get("dealer_up", 2)
    usable_ace = obs.get("usable_ace", 0)
    can_double = obs.get("can_double", 0)
    can_split = obs.get("can_split", 0)

    # Si podemos split, aplicar reglas para pares
    if can_split:
        # Inferir el valor del par basado en el total
        # Para pares de ases (total=12) y 8s siempre split
        if pt == 12:  # Par de ases
            return 3  # Split
        elif pt == 16:  # Par de 8s
            return 3  # Split
        elif pt == 20:  # Par de 10s - nunca split
            return 1  # Stand
        elif pt == 18:  # Par de 9s
            return 3 if dealer_up in [2,3,4,5,6,8,9] else 1
        elif pt == 14:  # Par de 7s
            return 3 if dealer_up in [2,3,4,5,6,7] else 0
        elif pt == 12:  # Par de 6s (pero cuidado con ases)
            return 3 if dealer_up in [2,3,4,5,6] else 0
        elif pt == 10:  # Par de 5s - tratar como 10
            return 2 if can_double and dealer_up in [2,3,4,5,6,7,8,9] else 0
        elif pt == 8:  # Par de 4s
            return 0  # Hit
        elif pt == 6:  # Par de 3s
            return 3 if dealer_up in [2,3,4,5,6,7] else 0
        elif pt == 4:  # Par de 2s
            return 3 if dealer_up in [2,3,4,5,6,7] else 0

    # Manos suaves (con as usable)
    if usable_ace:
        if pt >= 20:
            return 1  # Stand
        elif pt == 19:
            return 2 if dealer_up == 6 and can_double else 1  # Double vs 6, else stand
        elif pt == 18:
            if dealer_up in [2,3,4,5,6]:
                return 2 if can_double else 1  # Double
            elif dealer_up in [7,8]:
                return 1  # Stand
            else:
                return 0  # Hit
        elif pt == 17:
            return 2 if dealer_up in [3,4,5,6] and can_double else 0
        elif pt in [15,16]:
            return 2 if dealer_up in [4,5,6] and can_double else 0
        elif pt in [13,14]:
            return 2 if dealer_up in [5,6] and can_double else 0
        else:
            return 0  # Hit

    # Manos duras (sin as usable)
    if pt >= 17:
        return 1  # Stand
    elif pt >= 13:
        return 1 if dealer_up in [2,3,4,5,6] else 0  # Stand vs 2-6, else hit
    elif pt == 12:
        return 1 if dealer_up in [4,5,6] else 0  # Stand vs 4-6, else hit
    elif pt == 11:
        return 2 if can_double else 0  # Double
    elif pt == 10:
        return 2 if can_double and dealer_up in [2,3,4,5,6,7,8,9] else 0
    elif pt == 9:
        return 2 if can_double and dealer_up in [3,4,5,6] else 0
    else:
        return 0  # Hit


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
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in ckpt["policy_state"].items()
                       if k in model_dict and v.size() == model_dict[k].size()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"[INFO] Cargadas {len(pretrained_dict)} capas del checkpoint. "
          f"Las demás inicializadas desde cero.")


# ----------------- Preentrenamiento US_HSDP (streaming 5M) -----------------
def _total_and_usable_ace(cards):
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    usable = 1 if aces > 0 and total <= 21 else 0
    return total, usable

def _can_split_from_pair(cards):
    return len(cards) == 2 and cards[0] == cards[1]

def _map_first_action(actions_taken):
    """
    actions_taken como string con lista de listas: \"[['P','H','S'],['H','S']]\" o \"[['S']]\".
    Tomamos SOLO la primera acción del primer sub-hand.
    """
    try:
        seq = ast.literal_eval(actions_taken)
        if not isinstance(seq, list) or not seq or not isinstance(seq[0], list) or not seq[0]:
            return None
        a = str(seq[0][0]).upper()
        if   a == 'H': return 0
        elif a == 'S': return 1
        elif a == 'D': return 2
        elif a == 'P': return 3
        else:
            return None
    except Exception:
        return None

def pretrain_from_csv_US_HSDP_streaming(
    net,
    csv_path,
    optimizer,
    limit_rows=5_000_000,
    batch_size=2048,
    chunk_rows=200_000,
    epochs=1
):
    """
    Lee 'data/blackjack_US_HSDP.csv' en streaming y entrena clasificación (acción) sobre el
    primer turno de la mano inicial.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] No se encontró {csv_path}, se omite preentrenamiento.")
        return

    usecols = ["dealer_up", "initial_hand", "actions_taken"]

    print(f"[INFO] Massive pretrain (US_HSDP) from {csv_path} | limit={limit_rows:,} "
          f"| chunk={chunk_rows:,} | batch={batch_size} | epochs={epochs}")

    criterion = nn.CrossEntropyLoss()
    seen = 0

    for ep in range(1, epochs + 1):
        print(f"[Pretrain-US] Epoch {ep}/{epochs}")
        reader = pd.read_csv(
            csv_path,
            usecols=usecols,
            chunksize=chunk_rows,
            iterator=True,
            on_bad_lines="skip",
            low_memory=True
        )
        for chunk in reader:
            if seen >= limit_rows:
                break

            chunk = chunk.sample(frac=1).reset_index(drop=True)

            features = []
            labels = []

            for _, row in chunk.iterrows():
                try:
                    hand = ast.literal_eval(row["initial_hand"])
                    if not isinstance(hand, list) or len(hand) < 2:
                        continue

                    # Saltar BJ natural (estado terminal)
                    if (len(hand) == 2) and (11 in hand) and (10 in hand):
                        continue

                    action = _map_first_action(row["actions_taken"])
                    if action is None:
                        continue

                    dealer_up = int(row["dealer_up"])
                    can_double = 1 if len(hand) == 2 else 0
                    can_split = 1 if _can_split_from_pair(hand) else 0

                    # Omitir acciones ilegales
                    if action == 2 and not can_double:
                        continue
                    if action == 3 and not can_split:
                        continue

                    pt, ua = _total_and_usable_ace(hand)

                    obs_vec = obs_to_tensor({
                        "player_total": pt,
                        "dealer_up": dealer_up,
                        "usable_ace": ua,
                        "can_double": can_double,
                        "can_split": can_split,
                    })

                    features.append(obs_vec)
                    labels.append(action)

                    seen += 1
                    if seen >= limit_rows:
                        break
                except Exception:
                    continue

            if not features:
                continue

            X = torch.tensor(np.asarray(features), dtype=torch.float32, device=DEVICE)
            y = torch.tensor(np.asarray(labels),  dtype=torch.long,    device=DEVICE)

            losses = []
            n = X.size(0)
            for start in range(0, n, batch_size):
                xb = X[start:start+batch_size]
                yb = y[start:start+batch_size]

                logits = net(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
                losses.append(loss.item())

            print(f"[Pretrain-US] +{n:,} filas (total {seen:,}/{limit_rows:,}) "
                  f"| loss={np.mean(losses):.4f}")

            if seen >= limit_rows:
                break

        if seen >= limit_rows:
            break

    print(f"[Pretrain-US] Hecho. Filas consumidas: {min(seen, limit_rows):,}")


# ----------------- Entrenamiento -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)  # Reducido para mejor estabilidad
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=500_000)  # Aumentado
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-update", type=int, default=1)  # Para soft updates
    parser.add_argument("--start-train", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=1000)  # Evaluación más frecuente
    parser.add_argument("--save-dir", type=str, default="models/dqn_runs")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=500_000)  # Decay más lento
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load", type=str)
    parser.add_argument("--use-heuristic", action="store_true")
    parser.add_argument("--save-every-episodes", type=int, default=5000,
                        help="Guardar checkpoint cada N episodios")
    parser.add_argument("--pretrain-csv", type=str, default=None,
                        help="Ruta CSV para preentrenamiento (US_HSDP schema)")
    parser.add_argument("--pretrain-limit", type=int, default=5_000_000,
                        help="Filas a consumir para preentrenamiento")

    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True); os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    log_path = Path("logs/dqn_results.csv")

    env = BlackjackEnv(Rules(n_decks=1, h17=True, peek=True), seed=args.seed)
    state_dim, action_dim = 10, 4

    # Red más grande
    policy_net = DQNNet(state_dim, action_dim, hidden=(512, 512, 256)).to(DEVICE)
    target_net = DQNNet(state_dim, action_dim, hidden=(512, 512, 256)).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)

    global_steps = 0
    eps = args.eps_start
    start_episode = 0

    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.load:
        load_checkpoint_safe(policy_net, args.load)
        target_net.load_state_dict(policy_net.state_dict())
        ckpt = torch.load(args.load, map_location=DEVICE, weights_only=False)

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

    # --- Preentrenamiento con dataset ---
    csv_big = args.pretrain_csv or "data/blackjack_US_HSDP.csv"
    if os.path.exists(csv_big):
        pretrain_from_csv_US_HSDP_streaming(
            policy_net,
            csv_big,
            optimizer,
            limit_rows=args.pretrain_limit,
            batch_size=2048,
            chunk_rows=200_000,
            epochs=1
        )
        target_net.load_state_dict(policy_net.state_dict())
    else:
        csv_small = "data/blackjack.csv"
        if os.path.exists(csv_small):
           pretrain_from_csv_US_HSDP_streaming(policy_net, csv_small, optimizer, epochs=3)
        target_net.load_state_dict(policy_net.state_dict())

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["global_steps", "episode", "winrate", "avg_return"])

    # ----------------- Loop de entrenamiento -----------------
    criterion = nn.MSELoss(reduction="mean")  # Cambiado a MSE para mejor convergencia
    tau = 0.005  # Para soft updates

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

            # Política mixta dinámica
            p_policy = max(0.3, 1.0 - global_steps / (args.episodes * 10))
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
                    a = a_basic if a_basic in legal else random.choice(legal)

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

                loss = criterion(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                # Soft update en cada paso
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

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

        # -------- Guardados por episodio --------
        if ep % args.save_every_episodes == 0:
            ckpt = {
                "policy_state": policy_net.state_dict(),
                "target_state": target_net.state_dict(),
                "eps": eps,
                "global_steps": global_steps,
                "replay": replay.save_state(),
                "episode": ep,
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"checkpoint_ep{ep}.pth"))
            print(f"[Checkpoint] Saved checkpoint at episode {ep} -> {args.save_dir}/checkpoint_ep{ep}.pth")

        if ep % 10_000 == 0:
            ckpt = {
                "policy_state": policy_net.state_dict(),
                "target_state": target_net.state_dict(),
                "eps": eps,
                "global_steps": global_steps,
                "replay": replay.save_state(),
                "episode": ep,
            }
            out_path = os.path.join("data", f"CHK_{ep}-PTH")
            torch.save(ckpt, out_path)
            print(f"[Checkpoint] Saved (data) {out_path}")

    print("Training finished")
    torch.save(policy_net.state_dict(), os.path.join(args.save_dir, "dqn_final.pth"))


if __name__ == "__main__":
    main()