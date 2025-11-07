"""
Pretrain DQNNet by behavior cloning from a CSV dataset of Blackjack hands.
Saves checkpoint compatible with agents/train_dqn.py in models/dqn_pretrain/.
Updated to include correct feature dimension (10) matching train_dqn.py.
"""
import os
import argparse
import math
import csv
from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.dqn_net import DQNNet

# map dataset action letters to integers used in env: H=0, S=1, D=2, P=3
ACTION_MAP = {'H': 0, 'S': 1, 'D': 2, 'P': 3}


def parse_initial_hand(cell):
    """
    Try to parse initial_hand column. Supports:
    - Python list string like "[10, 11]"
    - Comma-separated "10 11" or "10,11"
    - Single value "20" etc -> return list
    Returns list[int]
    """
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return list(cell)
    s = str(cell).strip()
    # try literal_eval
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        if isinstance(v, int):
            return [v]
    except Exception:
        pass
    # fallback split by non-digit
    parts = re.findall(r'\d+', s)
    return [int(p) for p in parts]


def get_first_valid_action(cell):
    """
    actions_taken may be a string with sequence like "H;S" or "H" or "S,D".
    Return first action letter that maps to ACTION_MAP, or None.
    """
    if pd.isna(cell):
        return None
    s = str(cell).upper()
    # take first occurrence of H,S,D,P
    m = re.search(r'[HSDP]', s)
    if not m:
        return None
    return m.group(0)


def make_feature_from_row(row):
    """
    Build same features as obs_to_tensor in train_dqn:
    [player_total/21, usable_ace, dealer_up_norm, true_count/10,
     cards_remaining/312, hand_index/4, num_hands/4, can_double, can_split, episode_stage/3]
    """
    hand = parse_initial_hand(row.get('initial_hand', None))
    if not hand:
        if pd.notna(row.get('player_final_value')):
            total = int(row['player_final_value'])
            hand = [total]
        else:
            return None
    total = sum(hand)
    usable_ace = 1 if 11 in hand else 0
    dealer_up = int(row.get('dealer_up', 2))
    true_count = int(row.get('run_count', 0))
    cards_remaining = int(row.get('cards_remaining', 312))
    hand_index = 0
    num_hands = 1
    can_double = 1 if len(hand) == 2 else 0
    can_split = 0
    if len(hand) == 2 and (hand[0] == hand[1] or (hand[0] == 10 and hand[1] == 10)):
        can_split = 1

    player_total_n = float(total) / 21.0
    usable_ace_n = float(usable_ace)
    dealer_up_n = float(dealer_up - 2) / 9.0
    true_count_n = float(true_count) / 10.0
    cards_remaining_n = float(cards_remaining) / 312.0
    hand_index_n = float(hand_index) / 4.0
    num_hands_n = float(num_hands) / 4.0
    can_double_n = float(can_double)
    can_split_n = float(can_split)
    episode_stage = 1  # Asumimos etapa de juego en dataset, normalizada abajo
    episode_stage_n = float(episode_stage) / 3.0

    return np.array([player_total_n, usable_ace_n, dealer_up_n, true_count_n,
                     cards_remaining_n, hand_index_n, num_hands_n, can_double_n, can_split_n, episode_stage_n], dtype=np.float32)


def build_dataset_from_csv(path, sample_size=3_000_000, chunksize=200_000, max_rows=None):
    """
    Stream CSV in chunks, filter rows with first action in H/S/D/P, build features and labels.
    Returns X (numpy float32) and y (int).
    """
    X_list = []
    y_list = []
    scanned = 0
    taken = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for idx, row in chunk.iterrows():
            scanned += 1
            if max_rows and scanned > max_rows:
                break
            act = get_first_valid_action(row.get('actions_taken', ''))
            if act is None or act not in ACTION_MAP:
                continue
            feat = make_feature_from_row(row)
            if feat is None:
                continue
            X_list.append(feat)
            y_list.append(ACTION_MAP[act])
            taken += 1
            if taken >= sample_size:
                break
        if max_rows and scanned > max_rows:
            break
        if taken >= sample_size:
            break
        print(f"Scanned {scanned:,} rows, collected {taken:,}", end='\r')
    print(f"\nCollected {taken:,} examples from {scanned:,} scanned rows.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train_model(X, y, hidden=(128,128), epochs=5, batch_size=1024, lr=1e-4,
                weight_decay=1e-6, device='cpu', save_dir='models/dqn_pretrain'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    state_dim = X.shape[1]
    action_dim = 4
    model = DQNNet(state_dim, action_dim, hidden=hidden).to(device)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float('inf')
    best_state = None
    global_steps = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            global_steps += xb.size(0)
        train_loss = running_loss / train_size

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
        val_loss = vloss / val_size
        print(f"Epoch {epoch}/{epochs} — train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    ckpt = {
        'policy_state': best_state if best_state is not None else model.state_dict(),
        'target_state': best_state if best_state is not None else model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'eps': 1.0,
        'global_steps': 0,
        'replay': None
    }
    save_path = os.path.join(save_dir, 'checkpoint_pretrain.pth')
    torch.save(ckpt, save_path)
    print(f"Saved pretrain checkpoint to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/blackjack_US_HSDP")
    parser.add_argument("--sample-size", type=int, default=3_000_000)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models/dqn_pretrain")
    args = parser.parse_args()

    print("Building dataset (this may take a while)...")
    X, y = build_dataset_from_csv(args.csv, sample_size=args.sample_size, chunksize=args.chunksize)
    print("Training model (behavior cloning)...")
    ckpt_path = train_model(X, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            device=args.device, save_dir=args.save_dir)
    print("Done. Checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
