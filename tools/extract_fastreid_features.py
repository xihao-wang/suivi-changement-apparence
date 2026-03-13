#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract FastReID features for MOT-format detections and save as StrongSORT .npy"
    )
    parser.add_argument("--fastreid_root", required=True, help="Path to the fast-reid repository")
    parser.add_argument("--config_file", required=True, help="Path to a FastReID config .yml")
    parser.add_argument("--weights", required=True, help="Path to FastReID model weights")
    parser.add_argument("--sequence_dir", required=True, help="Path to the MOT-style sequence directory")
    parser.add_argument("--detections_txt", required=True, help="Path to MOT-style detections txt")
    parser.add_argument("--output_npy", required=True, help="Path to output .npy file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--min_score", type=float, default=0.6, help="Minimum detection score to keep")
    parser.add_argument("--batch_size", type=int, default=32, help="Feature extraction batch size")
    return parser.parse_args()


def load_fastreid(fastreid_root):
    sys.path.insert(0, str(Path(fastreid_root).resolve()))
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer
    from fastreid.utils.checkpoint import Checkpointer

    return get_cfg, DefaultTrainer, Checkpointer


def build_model(args):
    get_cfg, DefaultTrainer, Checkpointer = load_fastreid(args.fastreid_root)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    model = DefaultTrainer.build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(torch.device(args.device))
    return cfg, model


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


def clip_box(box, width, height):
    x, y, w, h = box
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def main():
    args = parse_args()
    cfg, model = build_model(args)
    transform = get_transform()

    detections = np.loadtxt(args.detections_txt, delimiter=",", ndmin=2)
    detections = detections[detections[:, 6] >= args.min_score]
    if detections.size == 0:
        raise ValueError("No detections remained after score filtering.")

    image_dir = Path(args.sequence_dir) / "img1"
    if not image_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {image_dir}")

    all_rows = []
    min_frame = int(detections[:, 0].min())
    max_frame = int(detections[:, 0].max())
    device = torch.device(args.device)

    with torch.no_grad():
        for frame_id in range(min_frame, max_frame + 1):
            frame_rows = detections[detections[:, 0] == frame_id]
            if len(frame_rows) == 0:
                continue

            image_path = image_dir / f"{frame_id:06d}.jpg"
            if not image_path.exists():
                print(f"Skipping missing frame image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size

            valid_rows = []
            patches = []
            for row in frame_rows:
                clipped = clip_box(row[2:6], img_w, img_h)
                if clipped is None:
                    continue
                patch = image.crop(clipped)
                patches.append(transform(patch))
                valid_rows.append(row)

            if not patches:
                continue

            features = []
            for patch_batch in batched(patches, args.batch_size):
                batch_tensor = torch.stack(patch_batch, dim=0).to(device)
                outputs = model(batch_tensor).detach().cpu().numpy()
                features.append(outputs)

            feature_mat = np.concatenate(features, axis=0)
            rows_mat = np.asarray(valid_rows, dtype=np.float32)
            all_rows.append(np.concatenate([rows_mat, feature_mat], axis=1))

    if not all_rows:
        raise ValueError("No valid cropped detections were processed.")

    output = np.concatenate(all_rows, axis=0)
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output, allow_pickle=False)
    print(f"Saved features to {output_path}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
