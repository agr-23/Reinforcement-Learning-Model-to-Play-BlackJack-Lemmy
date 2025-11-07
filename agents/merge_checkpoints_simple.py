"""
DQN Checkpoint Merger for Blackjack Agent

This script provides functionality to merge two DQN model checkpoints using linear interpolation,
typically used to combine a pretrained model with a fine-tuned one. This can help in balancing
between general knowledge (from pretraining) and specific adaptations (from fine-tuning).

Key Features:
- Safe loading of checkpoints with numpy array reconstruction support
- Linear interpolation between model parameters using mixing factor α
- Support for different checkpoint formats (raw state dict, policy state, etc.)
- Automatic parameter compatibility checking

The merger performs: merged_param = α * fine_tuned_param + (1-α) * pretrained_param

Usage Examples:
1. Equal weight merge (α=0.5):
   python agents/merge_checkpoints_simple.py \
     --dqn models/dqn_finetune/checkpoint_5540000.pth \
     --pre models/dqn_pretrain/checkpoint_pretrain.pth \
     --out models/dqn_merged/checkpoint_merged.pth \
     --alpha 0.5

2. Favor fine-tuned model (α=0.8):
   python agents/merge_checkpoints_simple.py \
     --dqn models/dqn_finetune/checkpoint_latest.pth \
     --pre models/dqn_pretrain/checkpoint_pretrain.pth \
     --out models/dqn_merged/merged_ft_heavy.pth \
     --alpha 0.8

Parameters:
--dqn: Path to fine-tuned checkpoint
--pre: Path to pretrained checkpoint
--out: Path to save merged checkpoint
--alpha: Mixing factor (0.0-1.0), higher values favor the fine-tuned model
"""

import torch
import argparse
from collections import OrderedDict

# COMMAND TO RUN THIS SCRIPT:
#python agents/merge_checkpoints.py \
 # --dqn models/dqn_finetune/checkpoint_5540000.pth \ -> ENSURE THAT IS THE LAST FINETUNED CHECKPOINT
  #--pre models/dqn_pretrain/checkpoint_pretrain.pth \
  #--out models/dqn_merged/checkpoint_merged.pth \
  #--alpha 0.5

def safe_load_state(path):
    print(f"Loading checkpoint from {path}")
    torch.serialization.add_safe_globals([__import__("numpy")._core.multiarray._reconstruct])

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # if is a dict, try to extract state dict
    if isinstance(ckpt, dict):
        if "policy_state" in ckpt:
            return ckpt["policy_state"]
        elif "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt  # fallback


def merge_dicts(sd1, sd2, alpha):
    merged = OrderedDict()
    for k, v1 in sd1.items():
        v2 = sd2.get(k)
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            merged[k] = alpha * v1 + (1 - alpha) * v2
        else:
            merged[k] = v1
    return merged


def merge_checkpoints(dqn_path, pre_path, out_path, alpha=0.5):
    sd_dqn = safe_load_state(dqn_path)
    sd_pre = safe_load_state(pre_path)

    print("Merging model parameters only...")
    merged = merge_dicts(sd_dqn, sd_pre, alpha)

    # Save only the state_dict compatible with DQNNet
    torch.save({"state_dict": merged}, out_path)
    print(f"Merge completed and saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two DQN checkpoints")
    parser.add_argument("--dqn", type=str, required=True, help="Path to the long DQN checkpoint")
    parser.add_argument("--pre", type=str, required=True, help="Path to the pre-trained checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Path to save the merged checkpoint")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mixing factor (0.0–1.0)")

    args = parser.parse_args()
    merge_checkpoints(args.dqn, args.pre, args.out, args.alpha)