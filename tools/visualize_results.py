#!/usr/bin/env python3
"""Visualize MOT-format tracking results on sequence images.

Usage:
  python tools/visualize_results.py \
    --sequence_dir ./data/MOTChallenge/MOT17/train/MOT17-02-FRCNN \
    --result_txt ./results/MOT17-02-FRCNN.txt \
    --out_video ./results/vis/MOT17-02-FRCNN.mp4 --fps 25

The result txt should be in MOT challenge format per line:
frame, id, x, y, w, h, 1, -1, -1, -1
"""
import os
import argparse
from collections import defaultdict
import cv2


def parse_results(result_txt):
    frames = defaultdict(list)
    with open(result_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            frames[frame].append((tid, int(x), int(y), int(w), int(h)))
    return frames


def id2color(tid):
    # deterministic color per id
    r = (37 * tid) % 255
    g = (17 * tid + 50) % 255
    b = (29 * tid + 80) % 255
    return (int(b), int(g), int(r))


def visualize(sequence_dir, result_txt, out_video=None, out_dir=None, fps=25, show=False):
    img_dir = os.path.join(sequence_dir, 'img1')
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")

    frames = parse_results(result_txt)

    img_files = {int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
                 for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))}
    if not img_files:
        raise FileNotFoundError(f"No images found in {img_dir}")

    min_idx = min(img_files.keys())
    max_idx = max(img_files.keys())

    writer = None
    if out_video:
        # probe size from first frame
        first_path = img_files[min_idx]
        img = cv2.imread(first_path)
        h, w = img.shape[:2]
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video, fourcc, float(fps), (w, h))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for idx in range(min_idx, max_idx + 1):
        if idx not in img_files:
            continue
        img_path = img_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            continue

        for det in frames.get(idx, []):
            tid, x, y, w_box, h_box = det
            color = id2color(tid)
            pt1 = (x, y)
            pt2 = (x + w_box, y + h_box)
            cv2.rectangle(img, pt1, pt2, color, 2)
            cv2.putText(img, str(tid), (x, max(y - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if show:
            cv2.imshow('vis', img)
            key = cv2.waitKey(int(1000.0 / fps))
            if key == 27:
                break

        if writer is not None:
            writer.write(img)

        if out_dir:
            out_path = os.path.join(out_dir, f"{idx:06d}.jpg")
            cv2.imwrite(out_path, img)

    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_dir', required=True)
    parser.add_argument('--result_txt', required=True)
    parser.add_argument('--out_video', default=None)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    visualize(args.sequence_dir, args.result_txt, args.out_video, args.out_dir, args.fps, args.show)


if __name__ == '__main__':
    main()
