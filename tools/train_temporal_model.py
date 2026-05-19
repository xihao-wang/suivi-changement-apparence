#!/usr/bin/env python3
"""Minimal trainer for TemporalAttentionScorer."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_sort.temporal_model import TemporalAttentionScorer


def _build_group_ids(frame, det_index, source_index=0):
    group_map = {}
    group_ids = []
    for frame_id, det_id in zip(frame.tolist(), det_index.tolist()):
        key = (int(source_index), int(frame_id), int(det_id))
        if key not in group_map:
            group_map[key] = len(group_map)
        group_ids.append(group_map[key])
    return np.asarray(group_ids, dtype=np.int64)


def _build_track_group_ids(frame, track_id, source_index=0):
    """Group by (source, frame, track_id) for track-centric ranking.

    Within each group: same track, multiple detections from the same frame.
    Loss direction: for a given track, its correct detection should score
    higher than all wrong detections (column-wise complement to detection-centric).
    """
    group_map = {}
    group_ids = []
    for frame_id, tid in zip(frame.tolist(), track_id.tolist()):
        key = (int(source_index), int(frame_id), int(tid))
        if key not in group_map:
            group_map[key] = len(group_map)
        group_ids.append(group_map[key])
    return np.asarray(group_ids, dtype=np.int64)


class TemporalPairDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        history_len: int = 5,
        long_history_len: int = 30,
    ):
        data = np.load(npz_path)
        required = {
            "det_feat",
            "short_hist_feat",
            "long_hist_feat",
            "short_hist_len",
            "long_hist_len",
            "label",
            "frame",
            "track_id",
            "det_index",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"{npz_path} is missing required fields for the memory schema: {missing}"
            )
        self.history_len = history_len
        self.long_history_len = long_history_len
        self.det_feat = data["det_feat"].astype(np.float32)
        self.short_hist_feat = data["short_hist_feat"].astype(np.float32)
        self.long_hist_feat = data["long_hist_feat"].astype(np.float32)
        self.short_hist_len = data["short_hist_len"].astype(np.int64)
        self.long_hist_len = data["long_hist_len"].astype(np.int64)
        self.label = data["label"].astype(np.float32)
        self.frame = data["frame"].astype(np.int64)
        self.track_id = data["track_id"].astype(np.int64)
        self.det_index = data["det_index"].astype(np.int64)
        self.group_id = _build_group_ids(self.frame, self.det_index)
        self.track_group_id = _build_track_group_ids(self.frame, self.track_id)
        if self.short_hist_feat.shape[1] != self.history_len:
            raise ValueError(
                f"{npz_path} short_hist_feat has length {self.short_hist_feat.shape[1]}, expected {self.history_len}"
            )
        if self.long_hist_feat.shape[1] != self.long_history_len:
            raise ValueError(
                f"{npz_path} long_hist_feat has length {self.long_hist_feat.shape[1]}, expected {self.long_history_len}"
            )

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        det = torch.from_numpy(self.det_feat[idx])
        hist = torch.from_numpy(self.short_hist_feat[idx])
        long_hist = torch.from_numpy(self.long_hist_feat[idx])
        short_len = torch.tensor(int(self.short_hist_len[idx]), dtype=torch.long)
        long_len = torch.tensor(int(self.long_hist_len[idx]), dtype=torch.long)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        group_id = torch.tensor(int(self.group_id[idx]), dtype=torch.long)
        track_id = torch.tensor(int(self.track_id[idx]), dtype=torch.long)
        track_group_id = torch.tensor(int(self.track_group_id[idx]), dtype=torch.long)
        return det, hist, long_hist, short_len, long_len, label, group_id, track_id, track_group_id


class MultiTemporalPairDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[str],
        history_len: int = 5,
        long_history_len: int = 30,
    ):
        if not npz_paths:
            raise ValueError("npz_paths must not be empty")
        self.history_len = history_len
        self.long_history_len = long_history_len

        det_feats = []
        short_hist_feats = []
        long_hist_feats = []
        short_hist_lens = []
        long_hist_lens = []
        labels = []
        frames = []
        track_ids = []
        det_indices = []
        group_ids = []
        track_group_ids = []
        feature_dim = None

        group_offset = 0
        track_group_offset = 0
        for source_index, npz_path in enumerate(npz_paths):
            data = np.load(npz_path)
            required = {
                "det_feat",
                "short_hist_feat",
                "long_hist_feat",
                "short_hist_len",
                "long_hist_len",
                "label",
                "frame",
                "track_id",
                "det_index",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    f"{npz_path} is missing required fields for the memory schema: {missing}"
                )
            det_feat = data["det_feat"].astype(np.float32)
            short_hist_feat = data["short_hist_feat"].astype(np.float32)
            long_hist_feat = data["long_hist_feat"].astype(np.float32)
            short_hist_len = data["short_hist_len"].astype(np.int64)
            long_hist_len = data["long_hist_len"].astype(np.int64)
            label = data["label"].astype(np.float32)
            frame = data["frame"].astype(np.int64)
            track_id = data["track_id"].astype(np.int64)
            det_index = data["det_index"].astype(np.int64)
            group_id = _build_group_ids(frame, det_index, source_index=source_index)
            group_id = group_id + group_offset
            group_offset = int(group_id.max()) + 1 if len(group_id) else group_offset
            track_group_id = _build_track_group_ids(frame, track_id, source_index=source_index)
            track_group_id = track_group_id + track_group_offset
            track_group_offset = int(track_group_id.max()) + 1 if len(track_group_id) else track_group_offset

            if feature_dim is None:
                feature_dim = det_feat.shape[1]
            elif det_feat.shape[1] != feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {feature_dim}, got {det_feat.shape[1]} from {npz_path}"
                )
            if short_hist_feat.shape[1] != self.history_len:
                raise ValueError(
                    f"{npz_path} short_hist_feat has length {short_hist_feat.shape[1]}, expected {self.history_len}"
                )
            if long_hist_feat.shape[1] != self.long_history_len:
                raise ValueError(
                    f"{npz_path} long_hist_feat has length {long_hist_feat.shape[1]}, expected {self.long_history_len}"
                )

            det_feats.append(det_feat)
            short_hist_feats.append(short_hist_feat)
            long_hist_feats.append(long_hist_feat)
            short_hist_lens.append(short_hist_len)
            long_hist_lens.append(long_hist_len)
            labels.append(label)
            frames.append(frame)
            track_ids.append(track_id)
            det_indices.append(det_index)
            group_ids.append(group_id)
            track_group_ids.append(track_group_id)

        self.det_feat = np.concatenate(det_feats, axis=0)
        self.short_hist_feat = np.concatenate(short_hist_feats, axis=0)
        self.long_hist_feat = np.concatenate(long_hist_feats, axis=0)
        self.short_hist_len = np.concatenate(short_hist_lens, axis=0)
        self.long_hist_len = np.concatenate(long_hist_lens, axis=0)
        self.label = np.concatenate(labels, axis=0)
        self.frame = np.concatenate(frames, axis=0)
        self.track_id = np.concatenate(track_ids, axis=0)
        self.det_index = np.concatenate(det_indices, axis=0)
        self.group_id = np.concatenate(group_ids, axis=0)
        self.track_group_id = np.concatenate(track_group_ids, axis=0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        det = torch.from_numpy(self.det_feat[idx])
        hist = torch.from_numpy(self.short_hist_feat[idx])
        long_hist = torch.from_numpy(self.long_hist_feat[idx])
        short_len = torch.tensor(int(self.short_hist_len[idx]), dtype=torch.long)
        long_len = torch.tensor(int(self.long_hist_len[idx]), dtype=torch.long)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        group_id = torch.tensor(int(self.group_id[idx]), dtype=torch.long)
        track_id = torch.tensor(int(self.track_id[idx]), dtype=torch.long)
        track_group_id = torch.tensor(int(self.track_group_id[idx]), dtype=torch.long)
        return det, hist, long_hist, short_len, long_len, label, group_id, track_id, track_group_id


class SameDetectionBatchSampler(Sampler):
    def __init__(self, group_ids, batch_size: int, seed: int = 42, shuffle: bool = True):
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        groups = {}
        for idx, group_id in enumerate(self.group_ids.tolist()):
            groups.setdefault(int(group_id), []).append(idx)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_indices = np.arange(len(self.groups))
        if self.shuffle:
            rng.shuffle(group_indices)

        batch = []
        for group_idx in group_indices.tolist():
            group = list(self.groups[group_idx])
            if self.shuffle:
                rng.shuffle(group)
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch
        self.epoch += 1

    def __len__(self):
        return max(1, math.ceil(len(self.group_ids) / max(self.batch_size, 1)))


def same_detection_ranking_loss(logits, labels, group_ids, margin=0.2, hard_negative_topk=1):
    losses = []
    unique_groups = torch.unique(group_ids)
    for group_id in unique_groups:
        group_mask = group_ids == group_id
        group_logits = logits[group_mask]
        group_labels = labels[group_mask]
        pos_logits = group_logits[group_labels >= 0.5]
        neg_logits = group_logits[group_labels < 0.5]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            continue
        hardest_pos = pos_logits.min()
        k = min(max(int(hard_negative_topk), 1), neg_logits.numel())
        hard_negs = torch.topk(neg_logits, k=k, largest=True).values
        losses.append(torch.relu(margin - (hardest_pos - hard_negs)).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def infonce_loss(logits, labels, group_ids, temperature=1.0):
    """InfoNCE / multi-negative ranking loss.

    For each (frame, detection) group, the correct track must dominate via
    softmax over all competing tracks:

        L = -log( exp(pos_logit / T) / sum_j exp(logit_j / T) )

    Unlike the hinge-based ranking loss, the gradient never vanishes once the
    margin is satisfied: every wrong track that scores high still contributes
    to the loss, forcing the model to keep pushing incorrect scores down.
    """
    losses = []
    temperature = max(float(temperature), 1e-8)
    unique_groups = torch.unique(group_ids)
    for group_id in unique_groups:
        mask = group_ids == group_id
        group_logits = logits[mask] / temperature
        group_labels = labels[mask]
        pos_mask = group_labels >= 0.5
        if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
            continue
        # -log P(correct | all candidates)  =  logsumexp(all) - pos_logit
        log_partition = torch.logsumexp(group_logits, dim=0)
        loss = (log_partition - group_logits[pos_mask]).mean()
        losses.append(loss)
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal attention scorer.")
    parser.add_argument("--pair_npz", default=None, help="Single dataset path; used with random split if train/val npz are not provided")
    parser.add_argument("--train_pair_npz", nargs="+", default=None, help="One or more training dataset npz files")
    parser.add_argument("--val_pair_npz", nargs="+", default=None, help="One or more validation dataset npz files")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--history_len", type=int, default=5, choices=[2, 3, 5])
    parser.add_argument("--long_history_len", type=int, default=30)
    parser.add_argument("--short_only", action="store_true", help="Disable the long-memory branch and train a short-only scorer.")
    parser.add_argument("--ranking_weight", type=float, default=1.0, help="Weight for same-detection ranking loss.")
    parser.add_argument("--ranking_loss_type", default="hinge", choices=["hinge", "infonce"],
                        help="Type of same-detection ranking loss: 'hinge' (original margin loss) or 'infonce' (multi-negative softmax loss).")
    parser.add_argument("--ranking_margin", type=float, default=0.2, help="Required logit margin (hinge only).")
    parser.add_argument("--hard_negative_topk", type=int, default=1, help="Number of hardest negatives per group (hinge only).")
    parser.add_argument("--infonce_temperature", type=float, default=1.0,
                        help="Temperature for InfoNCE loss. Lower values produce sharper distributions (infonce only).")
    parser.add_argument("--track_ranking_weight", type=float, default=0.0,
                        help="Weight for track-centric InfoNCE loss: for a given track, correct detection scores "
                             "higher than wrong detections. Complements detection-centric ranking. 0 = disabled.")
    parser.add_argument("--no_group_batches", action="store_true", help="Disable same-detection grouped training batches.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for det, hist, long_hist, short_len, long_len, label, _group_id, _track_id, _track_group_id in loader:
            det = det.to(device)
            hist = hist.to(device)
            long_hist = long_hist.to(device)
            short_len = short_len.to(device)
            long_len = long_len.to(device)
            label = label.to(device)
            logits = model(
                det,
                hist,
                long_hist_feat=long_hist,
                short_hist_len=short_len,
                long_hist_len=long_len,
                return_attention=False,
            )
            loss = criterion(logits, label)
            total_loss += float(loss.item()) * det.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += int((preds == label).sum().item())
            total += det.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_explicit_split = args.train_pair_npz is not None or args.val_pair_npz is not None
    if use_explicit_split:
        if not args.train_pair_npz or not args.val_pair_npz:
            raise ValueError("When using explicit split, both --train_pair_npz and --val_pair_npz must be provided")
        train_set = MultiTemporalPairDataset(
            args.train_pair_npz,
            history_len=args.history_len,
            long_history_len=args.long_history_len,
        )
        val_set = MultiTemporalPairDataset(
            args.val_pair_npz,
            history_len=args.history_len,
            long_history_len=args.long_history_len,
        )
        if len(train_set) == 0:
            raise ValueError("Empty training pair dataset")
        if len(val_set) == 0:
            raise ValueError("Empty validation pair dataset")
        feature_dim = train_set.det_feat.shape[1]
        if val_set.det_feat.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch between train ({feature_dim}) and val ({val_set.det_feat.shape[1]}) datasets"
            )
        labels = train_set.label
    else:
        if not args.pair_npz:
            raise ValueError("Provide --pair_npz or both --train_pair_npz and --val_pair_npz")
        dataset = TemporalPairDataset(
            args.pair_npz,
            history_len=args.history_len,
            long_history_len=args.long_history_len,
        )
        if len(dataset) == 0:
            raise ValueError("Empty pair dataset")

        val_size = max(1, int(len(dataset) * args.val_ratio))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        feature_dim = dataset.det_feat.shape[1]
        labels = dataset.label

    if not args.no_group_batches and hasattr(train_set, "group_id"):
        train_sampler = SameDetectionBatchSampler(
            train_set.group_id,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=True,
        )
        train_loader = DataLoader(train_set, batch_sampler=train_sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = TemporalAttentionScorer(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        history_len=args.history_len,
        long_history_len=args.long_history_len,
        use_long_memory=not args.short_only,
    ).to(args.device)

    num_pos = float(labels.sum())
    num_neg = float(len(labels) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=args.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.ranking_loss_type == "infonce":
        ranking_desc = f"infonce (T={args.infonce_temperature})"
    else:
        ranking_desc = f"hinge (margin={args.ranking_margin}, topk={args.hard_negative_topk})"
    track_desc = f" + track-centric infonce (w={args.track_ranking_weight})" if args.track_ranking_weight > 0 else ""
    print(
        "Training objective: "
        f"BCE + {args.ranking_loss_type} ranking [{ranking_desc}]{track_desc} "
        f"(ranking_weight={args.ranking_weight}, "
        f"group_batches={not args.no_group_batches and hasattr(train_set, 'group_id')})"
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        running_bce_loss = 0.0
        running_rank_loss = 0.0
        running_track_rank_loss = 0.0
        ranking_batches = 0

        for det, hist, long_hist, short_len, long_len, label, group_id, _track_id, track_group_id in train_loader:
            det = det.to(args.device)
            hist = hist.to(args.device)
            long_hist = long_hist.to(args.device)
            short_len = short_len.to(args.device)
            long_len = long_len.to(args.device)
            label = label.to(args.device)
            group_id = group_id.to(args.device)
            track_group_id = track_group_id.to(args.device)

            optimizer.zero_grad()
            logits = model(
                det,
                hist,
                long_hist_feat=long_hist,
                short_hist_len=short_len,
                long_hist_len=long_len,
                return_attention=False,
            )
            bce_loss = criterion(logits, label)
            if args.ranking_loss_type == "infonce":
                rank_loss = infonce_loss(
                    logits,
                    label,
                    group_id,
                    temperature=args.infonce_temperature,
                )
            else:
                rank_loss = same_detection_ranking_loss(
                    logits,
                    label,
                    group_id,
                    margin=args.ranking_margin,
                    hard_negative_topk=args.hard_negative_topk,
                )
            track_rank_loss = (
                infonce_loss(logits, label, track_group_id, temperature=args.infonce_temperature)
                if args.track_ranking_weight > 0.0
                else logits.new_tensor(0.0)
            )
            loss = bce_loss + args.ranking_weight * rank_loss + args.track_ranking_weight * track_rank_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * det.size(0)
            running_bce_loss += float(bce_loss.item()) * det.size(0)
            running_rank_loss += float(rank_loss.item()) * det.size(0)
            running_track_rank_loss += float(track_rank_loss.item()) * det.size(0)
            if float(rank_loss.item()) > 0.0:
                ranking_batches += 1
            total += det.size(0)

        train_loss = running_loss / max(total, 1)
        train_bce_loss = running_bce_loss / max(total, 1)
        train_rank_loss = running_rank_loss / max(total, 1)
        train_track_rank_loss = running_track_rank_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "history_len": args.history_len,
            "long_history_len": args.long_history_len,
            "use_long_memory": not args.short_only,
            "train_loss": train_loss,
            "train_bce_loss": train_bce_loss,
            "train_rank_loss": train_rank_loss,
            "ranking_weight": args.ranking_weight,
            "ranking_loss_type": args.ranking_loss_type,
            "ranking_margin": args.ranking_margin,
            "hard_negative_topk": args.hard_negative_topk,
            "infonce_temperature": args.infonce_temperature,
            "track_ranking_weight": args.track_ranking_weight,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(ckpt, save_dir / "best.pt")

        track_rank_str = (
            f" | train_track_rank={train_track_rank_loss:.4f}"
            if args.track_ranking_weight > 0.0
            else ""
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_bce={train_bce_loss:.4f} | "
            f"train_rank={train_rank_loss:.4f}"
            f"{track_rank_str} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )
    print(f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch:03d}")


if __name__ == "__main__":
    main()
