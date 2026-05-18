#!/usr/bin/env python3
"""Evaluate learned-only association with Hungarian matching on GT pair samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_sort.temporal_model import TemporalAttentionScorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal model scores as a standalone Hungarian matcher."
    )
    parser.add_argument("--pair_npz", required=True)
    parser.add_argument("--temporal_model_ckpt", required=True)
    parser.add_argument("--target_gt_id", type=int, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class IndexedPairDataset(Dataset):
    def __init__(self, npz_path: str, indices: np.ndarray):
        data = np.load(npz_path)
        self.indices = indices.astype(np.int64)
        self.det_feat = data["det_feat"].astype(np.float32)
        self.short_hist_feat = data["short_hist_feat"].astype(np.float32)
        self.long_hist_feat = data["long_hist_feat"].astype(np.float32)
        self.short_hist_len = data["short_hist_len"].astype(np.int64)
        self.long_hist_len = data["long_hist_len"].astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        return (
            idx,
            torch.from_numpy(self.det_feat[idx]),
            torch.from_numpy(self.short_hist_feat[idx]),
            torch.from_numpy(self.long_hist_feat[idx]),
            torch.tensor(int(self.short_hist_len[idx]), dtype=torch.long),
            torch.tensor(int(self.long_hist_len[idx]), dtype=torch.long),
        )


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
    data = np.load(args.pair_npz)
    target_ids = set(args.target_gt_id)
    selected_indices = np.flatnonzero(np.isin(data["track_id"], list(target_ids)))
    if len(selected_indices) == 0:
        raise ValueError(f"No samples found for target_gt_id={sorted(target_ids)}")

    model = load_model(args.temporal_model_ckpt, args.device)
    dataset = IndexedPairDataset(args.pair_npz, selected_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs_by_index = {}
    with torch.no_grad():
        for idx, det, hist, long_hist, short_len, long_len in loader:
            det = det.to(args.device)
            hist = hist.to(args.device)
            long_hist = long_hist.to(args.device)
            short_len = short_len.to(args.device)
            long_len = long_len.to(args.device)
            logits = model(
                det,
                hist,
                long_hist_feat=long_hist,
                short_hist_len=short_len,
                long_hist_len=long_len,
                return_attention=False,
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for sample_idx, prob in zip(idx.numpy().tolist(), probs.tolist()):
                probs_by_index[int(sample_idx)] = float(prob)

    grouped = defaultdict(list)
    for idx in selected_indices.tolist():
        grouped[int(data["frame"][idx])].append(idx)

    frame_total = frame_correct = 0
    pair_total = pair_correct = 0
    skipped_frames = 0
    for frame, indices in sorted(grouped.items()):
        track_ids = sorted({int(data["track_id"][idx]) for idx in indices})
        det_indices = sorted({int(data["det_index"][idx]) for idx in indices})
        score_matrix = np.full((len(track_ids), len(det_indices)), -np.inf, dtype=np.float64)
        label_matrix = np.zeros((len(track_ids), len(det_indices)), dtype=np.float32)
        track_to_row = {track_id: row for row, track_id in enumerate(track_ids)}
        det_to_col = {det_idx: col for col, det_idx in enumerate(det_indices)}

        for idx in indices:
            row = track_to_row[int(data["track_id"][idx])]
            col = det_to_col[int(data["det_index"][idx])]
            score_matrix[row, col] = probs_by_index[idx]
            label_matrix[row, col] = float(data["label"][idx])

        # Require every evaluated target row to have at least one valid positive pair.
        valid_rows = np.any(label_matrix >= 0.5, axis=1)
        if not np.any(valid_rows):
            skipped_frames += 1
            continue
        score_matrix = score_matrix[valid_rows]
        label_matrix = label_matrix[valid_rows]
        if score_matrix.shape[1] < score_matrix.shape[0]:
            skipped_frames += 1
            continue

        if not np.all(np.isfinite(score_matrix)):
            raise ValueError(f"Incomplete score matrix at frame {frame}")

        learned_cost = 1.0 - score_matrix
        row_ind, col_ind = linear_sum_assignment(learned_cost)
        chosen_labels = label_matrix[row_ind, col_ind] >= 0.5
        pair_correct += int(chosen_labels.sum())
        pair_total += int(len(chosen_labels))
        frame_correct += int(bool(np.all(chosen_labels)))
        frame_total += 1

    print(f"pair_npz: {args.pair_npz}")
    print(f"target_gt_ids: {sorted(target_ids)}")
    print(f"evaluated_frames: {frame_total}")
    print(f"skipped_frames: {skipped_frames}")
    print(f"selected_pairs: {pair_total}")
    print(f"pair_hungarian_accuracy: {pair_correct / max(pair_total, 1):.6f}")
    print(f"frame_hungarian_accuracy: {frame_correct / max(frame_total, 1):.6f}")


if __name__ == "__main__":
    main()
