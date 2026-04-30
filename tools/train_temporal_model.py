#!/usr/bin/env python3
"""Train the patch-token temporal matcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

        self.sequence_dirs = np.asarray(sequence_dirs)
        self.init_frame = np.concatenate(init_frames, axis=0)
        self.init_bbox = np.concatenate(init_bboxes, axis=0)
        self.det_frame = np.concatenate(det_frames, axis=0)
        self.det_bbox = np.concatenate(det_bboxes, axis=0)
        self.hist_frame = np.concatenate(hist_frames, axis=0)
        self.hist_bbox = np.concatenate(hist_bboxes, axis=0)
        self.label = np.concatenate(labels, axis=0)

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
        return init_crop, det_crop, hist_crops, label


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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for init_crop, det_crop, hist_crops, label in loader:
            init_crop = init_crop.to(device)
            det_crop = det_crop.to(device)
            hist_crops = hist_crops.to(device)
            label = label.to(device)
            logits = model(init_crop, det_crop, hist_crops, return_attention=False)
            loss = criterion(logits, label)
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = TemporalAttentionScorer(
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        history_len=len(history_indices),
    ).to(args.device)

    labels = train_set.label
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
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for init_crop, det_crop, hist_crops, label in train_loader:
            init_crop = init_crop.to(args.device)
            det_crop = det_crop.to(args.device)
            hist_crops = hist_crops.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()
            logits = model(init_crop, det_crop, hist_crops, return_attention=False)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * det_crop.size(0)
            total += det_crop.size(0)

        train_loss = running_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "image_height": args.image_height,
            "image_width": args.image_width,
            "patch_size": args.patch_size,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "history_len": len(history_indices),
            "history_indices": history_indices,
            "keep_aspect_ratio": not args.no_keep_aspect_ratio,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(ckpt, save_dir / "best.pt")

        print(
            f"Epoch {epoch:03d} | "
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
