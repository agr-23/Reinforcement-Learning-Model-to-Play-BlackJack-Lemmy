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