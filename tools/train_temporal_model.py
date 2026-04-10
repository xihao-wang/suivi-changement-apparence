#!/usr/bin/env python3
"""Minimal trainer for TemporalAttentionScorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_sort.temporal_model import TemporalAttentionScorer


class TemporalPairDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.det_feat = data["det_feat"].astype(np.float32)
        self.p_t = data["p_t"].astype(np.float32)
        self.p_t_i = data["p_t_i"].astype(np.float32)
        self.p_t_2i = data["p_t_2i"].astype(np.float32)
        self.label = data["label"].astype(np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        det = torch.from_numpy(self.det_feat[idx])
        hist = torch.from_numpy(np.stack([self.p_t[idx], self.p_t_i[idx], self.p_t_2i[idx]], axis=0))
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        return det, hist, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal attention scorer.")
    parser.add_argument("--pair_npz", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
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
        for det, hist, label in loader:
            det = det.to(device)
            hist = hist.to(device)
            label = label.to(device)
            logits = model(det, hist, return_attention=False)
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

    dataset = TemporalPairDataset(args.pair_npz)
    if len(dataset) == 0:
        raise ValueError("Empty pair dataset")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    feature_dim = dataset.det_feat.shape[1]
    model = TemporalAttentionScorer(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    ).to(args.device)

    labels = dataset.label
    num_pos = float(labels.sum())
    num_neg = float(len(labels) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=args.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for det, hist, label in train_loader:
            det = det.to(args.device)
            hist = hist.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()
            logits = model(det, hist, return_attention=False)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * det.size(0)
            total += det.size(0)

        train_loss = running_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best.pt")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
