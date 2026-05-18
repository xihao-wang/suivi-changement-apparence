#!/usr/bin/env python3
"""Evaluate a temporal model on GT-built pair samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_sort.temporal_model import TemporalAttentionScorer
from tools.train_temporal_model import TemporalPairDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate temporal pair accuracy.")
    parser.add_argument("--pair_npz", required=True)
    parser.add_argument("--temporal_model_ckpt", required=True)
    parser.add_argument("--target_gt_id", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    model = TemporalAttentionScorer(
        feature_dim=int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_heads=int(ckpt.get("num_heads", 4)),
        history_len=int(ckpt.get("history_len", 5)),
        long_history_len=int(ckpt.get("long_history_len", 30)),
        use_long_memory=bool(ckpt.get("use_long_memory", True)),
    )
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'state_dict' or 'model_state_dict'.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    ckpt = torch.load(args.temporal_model_ckpt, map_location="cpu")
    history_len = int(ckpt.get("history_len", 5))
    long_history_len = int(ckpt.get("long_history_len", 30))
    dataset = TemporalPairDataset(
        args.pair_npz,
        history_len=history_len,
        long_history_len=long_history_len,
    )
    if args.target_gt_id:
        target_ids = set(args.target_gt_id)
        indices = [
            idx for idx, track_id in enumerate(dataset.track_id.tolist())
            if int(track_id) in target_ids
        ]
        if not indices:
            raise ValueError(f"No samples found for target_gt_id={sorted(target_ids)}")
        eval_set = Subset(dataset, indices)
    else:
        target_ids = None
        eval_set = dataset

    model = load_model(args.temporal_model_ckpt, args.device)
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

    total = correct = pos_total = pos_correct = neg_total = neg_correct = 0
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for det, hist, long_hist, short_len, long_len, label, _group_id, _track_id in loader:
            det = det.to(args.device)
            hist = hist.to(args.device)
            long_hist = long_hist.to(args.device)
            short_len = short_len.to(args.device)
            long_len = long_len.to(args.device)
            label = label.to(args.device)
            logits = model(
                det,
                hist,
                long_hist_feat=long_hist,
                short_hist_len=short_len,
                long_hist_len=long_len,
                return_attention=False,
            )
            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            labels_bool = label >= 0.5
            correct += int((preds == labels_bool).sum().item())
            total += int(label.numel())
            pos_mask = labels_bool
            neg_mask = ~labels_bool
            pos_total += int(pos_mask.sum().item())
            neg_total += int(neg_mask.sum().item())
            pos_correct += int((preds[pos_mask] == labels_bool[pos_mask]).sum().item())
            neg_correct += int((preds[neg_mask] == labels_bool[neg_mask]).sum().item())
            probs_all.append(probs.detach().cpu().numpy())
            labels_all.append(label.detach().cpu().numpy())

    probs_np = np.concatenate(probs_all) if probs_all else np.asarray([], dtype=np.float32)
    labels_np = np.concatenate(labels_all) if labels_all else np.asarray([], dtype=np.float32)
    pos_probs = probs_np[labels_np >= 0.5]
    neg_probs = probs_np[labels_np < 0.5]

    print(f"pair_npz: {args.pair_npz}")
    print(f"target_gt_ids: {sorted(target_ids) if target_ids else 'all'}")
    print(f"samples: {total}")
    print(f"positive_samples: {pos_total}")
    print(f"negative_samples: {neg_total}")
    print(f"accuracy: {correct / max(total, 1):.6f}")
    print(f"positive_accuracy: {pos_correct / max(pos_total, 1):.6f}")
    print(f"negative_accuracy: {neg_correct / max(neg_total, 1):.6f}")
    print(f"positive_prob_mean: {float(pos_probs.mean()) if len(pos_probs) else float('nan'):.6f}")
    print(f"negative_prob_mean: {float(neg_probs.mean()) if len(neg_probs) else float('nan'):.6f}")


if __name__ == "__main__":
    main()
