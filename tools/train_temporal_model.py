#!/usr/bin/env python3
"""Train the patch-token temporal matcher."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_sort.temporal_model import TemporalAttentionScorer


def frame_to_image_path(sequence_dir: str, frame_idx: int) -> str:
    return str(Path(sequence_dir) / "img1" / f"{frame_idx:06d}.jpg")


def clip_tlwh(tlwh, image_shape):
    x, y, w, h = [float(v) for v in tlwh]
    img_h, img_w = image_shape[:2]
    x1 = max(0, min(img_w - 1, int(round(x))))
    y1 = max(0, min(img_h - 1, int(round(y))))
    x2 = max(x1 + 1, min(img_w, int(round(x + w))))
    y2 = max(y1 + 1, min(img_h, int(round(y + h))))
    return x1, y1, x2, y2


def crop_and_resize(image, tlwh, out_h, out_w, keep_aspect_ratio=True):
    x1, y1, x2, y2 = clip_tlwh(tlwh, image.shape)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if not keep_aspect_ratio:
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return resized

    crop_h, crop_w = crop.shape[:2]
    scale = min(out_w / max(crop_w, 1), out_h / max(crop_h, 1))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y_off = (out_h - new_h) // 2
    x_off = (out_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def image_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    return tensor


def load_patch_pair_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)
    sequence_dir = str(data["sequence_dir"].item())
    return {
        "sequence_dir": sequence_dir,
        "init_frame": data["init_frame"].astype(np.int32),
        "init_bbox": data["init_bbox"].astype(np.float32),
        "det_frame": data["det_frame"].astype(np.int32),
        "det_bbox": data["det_bbox"].astype(np.float32),
        "hist_frame": data["hist_frame"].astype(np.int32),
        "hist_bbox": data["hist_bbox"].astype(np.float32),
        "label": data["label"].astype(np.float32),
        "frame": data["frame"].astype(np.int32),
        "track_id": data["track_id"].astype(np.int32),
        "det_index": data["det_index"].astype(np.int32),
    }


class PatchTemporalPairDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[str],
        history_indices: list[int],
        image_height: int,
        image_width: int,
        keep_aspect_ratio: bool,
    ):
        if not npz_paths:
            raise ValueError("npz_paths must not be empty")
        self.history_indices = history_indices
        self.image_height = image_height
        self.image_width = image_width
        self.keep_aspect_ratio = keep_aspect_ratio

        sequence_dirs = []
        init_frames = []
        init_bboxes = []
        det_frames = []
        det_bboxes = []
        hist_frames = []
        hist_bboxes = []
        labels = []
        frames = []
        track_ids = []
        det_indices = []

        for npz_path in npz_paths:
            payload = load_patch_pair_npz(npz_path)
            num_samples = len(payload["label"])
            sequence_dirs.extend([payload["sequence_dir"]] * num_samples)
            init_frames.append(payload["init_frame"])
            init_bboxes.append(payload["init_bbox"])
            det_frames.append(payload["det_frame"])
            det_bboxes.append(payload["det_bbox"])
            hist_frames.append(payload["hist_frame"][:, history_indices])
            hist_bboxes.append(payload["hist_bbox"][:, history_indices, :])
            labels.append(payload["label"])
            frames.append(payload["frame"])
            track_ids.append(payload["track_id"])
            det_indices.append(payload["det_index"])

        self.sequence_dirs = np.asarray(sequence_dirs)
        self.init_frame = np.concatenate(init_frames, axis=0)
        self.init_bbox = np.concatenate(init_bboxes, axis=0)
        self.det_frame = np.concatenate(det_frames, axis=0)
        self.det_bbox = np.concatenate(det_bboxes, axis=0)
        self.hist_frame = np.concatenate(hist_frames, axis=0)
        self.hist_bbox = np.concatenate(hist_bboxes, axis=0)
        self.label = np.concatenate(labels, axis=0)
        self.frame = np.concatenate(frames, axis=0)
        self.track_id = np.concatenate(track_ids, axis=0)
        self.det_index = np.concatenate(det_indices, axis=0)

        group_to_id = {}
        group_ids = []
        next_group_id = 0
        for seq_dir, frame, track_id in zip(self.sequence_dirs, self.frame, self.track_id):
            key = (str(seq_dir), int(frame), int(track_id))
            if key not in group_to_id:
                group_to_id[key] = next_group_id
                next_group_id += 1
            group_ids.append(group_to_id[key])
        self.group_ids = np.asarray(group_ids, dtype=np.int64)

    def __len__(self):
        return len(self.label)

    def _read_crop(self, sequence_dir, frame_idx, tlwh):
        image_path = frame_to_image_path(sequence_dir, int(frame_idx))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        crop = crop_and_resize(
            image,
            tlwh,
            self.image_height,
            self.image_width,
            keep_aspect_ratio=self.keep_aspect_ratio,
        )
        return image_to_tensor(crop)

    def __getitem__(self, idx):
        sequence_dir = self.sequence_dirs[idx]
        init_crop = self._read_crop(sequence_dir, self.init_frame[idx], self.init_bbox[idx])
        det_crop = self._read_crop(sequence_dir, self.det_frame[idx], self.det_bbox[idx])

        hist_items = []
        for frame_idx, tlwh in zip(self.hist_frame[idx], self.hist_bbox[idx]):
            hist_items.append(self._read_crop(sequence_dir, frame_idx, tlwh))
        hist_crops = torch.stack(hist_items, dim=0)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        group_id = torch.tensor(self.group_ids[idx], dtype=torch.long)
        return init_crop, det_crop, hist_crops, label, group_id


class GroupBatchSampler(BatchSampler):
    def __init__(self, group_ids, batch_size: int, shuffle: bool):
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

        unique_ids, first_indices = np.unique(self.group_ids, return_index=True)
        order = np.argsort(first_indices)
        self.ordered_group_ids = unique_ids[order].tolist()
        self.group_to_indices = {
            int(group_id): np.where(self.group_ids == group_id)[0].tolist()
            for group_id in unique_ids
        }

    def __iter__(self):
        group_ids = list(self.ordered_group_ids)
        if self.shuffle:
            random.shuffle(group_ids)

        batch = []
        batch_count = 0
        for group_id in group_ids:
            indices = self.group_to_indices[int(group_id)]
            group_size = len(indices)
            if batch and batch_count + group_size > self.batch_size:
                yield batch
                batch = []
                batch_count = 0
            batch.extend(indices)
            batch_count += group_size
            if batch_count >= self.batch_size:
                yield batch
                batch = []
                batch_count = 0
        if batch:
            yield batch

    def __len__(self):
        num_batches = 0
        batch_count = 0
        for group_id in self.ordered_group_ids:
            group_size = len(self.group_to_indices[int(group_id)])
            if batch_count > 0 and batch_count + group_size > self.batch_size:
                num_batches += 1
                batch_count = 0
            batch_count += group_size
            if batch_count >= self.batch_size:
                num_batches += 1
                batch_count = 0
        if batch_count > 0:
            num_batches += 1
        return num_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Train a patch-token temporal matcher.")
    parser.add_argument("--train_pair_npz", nargs="+", required=True)
    parser.add_argument("--val_pair_npz", nargs="+", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_stages", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument(
        "--stage_dims",
        type=str,
        default=None,
        help="Comma-separated stage embedding dims. Defaults to paper-style values.",
    )
    parser.add_argument(
        "--stage_heads",
        type=str,
        default=None,
        help="Comma-separated stage attention heads. Defaults to paper-style values.",
    )
    parser.add_argument(
        "--stage_depths",
        type=str,
        default=None,
        help="Comma-separated number of MAM blocks per stage. Defaults to paper-style values.",
    )
    parser.add_argument(
        "--stage_kernels",
        type=str,
        default=None,
        help="Comma-separated stage embedding kernel sizes.",
    )
    parser.add_argument(
        "--stage_strides",
        type=str,
        default=None,
        help="Comma-separated stage embedding strides.",
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--history_len", type=int, default=3, choices=[2, 3])
    parser.add_argument(
        "--history_indices",
        type=str,
        default=None,
        help="Comma-separated selection from {0,1,2}: 0=df_t, 1=df_t-i, 2=df_t-2i. "
             "If omitted, uses the first --history_len indices.",
    )
    parser.add_argument(
        "--no_keep_aspect_ratio",
        action="store_true",
        help="Disable aspect-ratio-preserving resize+padding and use direct resize instead.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--objective",
        choices=["bce", "ranking", "hybrid"],
        default="hybrid",
        help="Training objective.",
    )
    parser.add_argument(
        "--ranking_margin",
        type=float,
        default=0.2,
        help="Margin used by ranking objective.",
    )
    parser.add_argument(
        "--ranking_weight",
        type=float,
        default=1.0,
        help="Weight of ranking term when objective=hybrid.",
    )
    parser.add_argument(
        "--disable_pin_memory",
        action="store_true",
        help="Disable DataLoader pin_memory.",
    )
    parser.add_argument(
        "--disable_persistent_workers",
        action="store_true",
        help="Disable DataLoader persistent_workers when num_workers > 0.",
    )
    return parser.parse_args()


def parse_history_indices(arg: str | None, history_len: int) -> list[int]:
    if arg is None:
        return [0, 1, 2][:history_len]
    indices = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if not indices:
        raise ValueError("--history_indices must not be empty")
    if any(i not in (0, 1, 2) for i in indices):
        raise ValueError("--history_indices values must be chosen from {0,1,2}")
    return indices


def parse_int_list(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    values = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty stage configuration list")
    return values


def pairwise_ranking_loss(logits, labels, group_ids, margin: float):
    losses = []
    unique_groups = torch.unique(group_ids)
    for group_id in unique_groups:
        mask = group_ids == group_id
        group_logits = logits[mask]
        group_labels = labels[mask]
        pos = group_logits[group_labels >= 0.5]
        neg = group_logits[group_labels < 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        losses.append(torch.relu(margin - diff).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_objective_loss(
    objective,
    logits,
    labels,
    group_ids,
    bce_criterion,
    ranking_margin,
    ranking_weight,
):
    if objective == "bce":
        return bce_criterion(logits, labels)
    ranking = pairwise_ranking_loss(logits, labels, group_ids, ranking_margin)
    if objective == "ranking":
        return ranking
    bce = bce_criterion(logits, labels)
    return bce + ranking_weight * ranking


def evaluate(model, loader, bce_criterion, device, objective, ranking_margin, ranking_weight):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for init_crop, det_crop, hist_crops, label, group_id in loader:
            init_crop = init_crop.to(device)
            det_crop = det_crop.to(device)
            hist_crops = hist_crops.to(device)
            label = label.to(device)
            group_id = group_id.to(device)
            logits = model(init_crop, det_crop, hist_crops, return_attention=False)
            loss = compute_objective_loss(
                objective,
                logits,
                label,
                group_id,
                bce_criterion,
                ranking_margin,
                ranking_weight,
            )
            total_loss += float(loss.item()) * det_crop.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += int((preds == label).sum().item())
            total += det_crop.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    args = parse_args()
    history_indices = parse_history_indices(args.history_indices, args.history_len)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(
        f"Training objective: {args.objective} "
        f"(ranking_margin={args.ranking_margin}, ranking_weight={args.ranking_weight})"
    )
    stage_dims = parse_int_list(args.stage_dims)
    stage_heads = parse_int_list(args.stage_heads)
    stage_depths = parse_int_list(args.stage_depths)
    stage_kernels = parse_int_list(args.stage_kernels)
    stage_strides = parse_int_list(args.stage_strides)

    train_set = PatchTemporalPairDataset(
        args.train_pair_npz,
        history_indices=history_indices,
        image_height=args.image_height,
        image_width=args.image_width,
        keep_aspect_ratio=not args.no_keep_aspect_ratio,
    )
    val_set = PatchTemporalPairDataset(
        args.val_pair_npz,
        history_indices=history_indices,
        image_height=args.image_height,
        image_width=args.image_width,
        keep_aspect_ratio=not args.no_keep_aspect_ratio,
    )
    if len(train_set) == 0:
        raise ValueError("Empty training pair dataset")
    if len(val_set) == 0:
        raise ValueError("Empty validation pair dataset")

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (not args.disable_pin_memory),
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = (
            not args.disable_persistent_workers
        )

    if args.objective in {"ranking", "hybrid"}:
        train_loader = DataLoader(
            train_set,
            batch_sampler=GroupBatchSampler(
                train_set.group_ids,
                batch_size=args.batch_size,
                shuffle=True,
            ),
            **dataloader_kwargs,
        )
        val_loader = DataLoader(
            val_set,
            batch_sampler=GroupBatchSampler(
                val_set.group_ids,
                batch_size=args.batch_size,
                shuffle=False,
            ),
            **dataloader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            **dataloader_kwargs,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            **dataloader_kwargs,
        )

    model = TemporalAttentionScorer(
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        history_len=len(history_indices),
        num_stages=args.num_stages,
        stage_dims=stage_dims,
        stage_heads=stage_heads,
        stage_depths=stage_depths,
        stage_kernels=stage_kernels,
        stage_strides=stage_strides,
    ).to(args.device)

    labels = train_set.label
    num_pos = float(labels.sum())
    num_neg = float(len(labels) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=args.device)

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        total = 0

        for init_crop, det_crop, hist_crops, label, group_id in train_loader:
            init_crop = init_crop.to(args.device)
            det_crop = det_crop.to(args.device)
            hist_crops = hist_crops.to(args.device)
            label = label.to(args.device)
            group_id = group_id.to(args.device)

            optimizer.zero_grad()
            logits = model(init_crop, det_crop, hist_crops, return_attention=False)
            loss = compute_objective_loss(
                args.objective,
                logits,
                label,
                group_id,
                bce_criterion,
                args.ranking_margin,
                args.ranking_weight,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * det_crop.size(0)
            total += det_crop.size(0)

        train_loss = running_loss / max(total, 1)
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            bce_criterion,
            args.device,
            args.objective,
            args.ranking_margin,
            args.ranking_weight,
        )
        epoch_time = time.perf_counter() - epoch_start

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "image_height": args.image_height,
            "image_width": args.image_width,
            "patch_size": args.patch_size,
            "hidden_dim": model.hidden_dim,
            "num_heads": model.num_heads,
            "num_stages": model.num_stages,
            "stage_dims": model.stage_dims,
            "stage_heads": model.stage_heads,
            "stage_depths": model.stage_depths,
            "stage_kernels": model.stage_kernels,
            "stage_strides": model.stage_strides,
            "history_len": len(history_indices),
            "history_indices": history_indices,
            "keep_aspect_ratio": not args.no_keep_aspect_ratio,
            "objective": args.objective,
            "ranking_margin": args.ranking_margin,
            "ranking_weight": args.ranking_weight,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(ckpt, save_dir / "best.pt")

        print(
            f"Epoch {epoch:03d} | "
            f"time={epoch_time:.1f}s | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    print(
        f"Training complete. Best validation loss: {best_val_loss:.4f} "
        f"at epoch {best_epoch:03d}"
    )


if __name__ == "__main__":
    main()
