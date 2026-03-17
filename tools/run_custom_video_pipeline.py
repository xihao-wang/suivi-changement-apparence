#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    bytetrack_root = (repo_root.parent / "ByteTrack").resolve()
    fastreid_root = (repo_root.parent / "fast-reid").resolve()
    conda_root = Path.home() / "miniconda3" / "envs"
    bytetrack_python = conda_root / "bytetrack-gpu" / "bin" / "python3"
    fastreid_python = conda_root / "fastreid" / "bin" / "python"

    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: video -> detections -> features -> StrongSORT -> visualization"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--seq", required=True, help="Sequence name, e.g. YT-02")
    parser.add_argument("--dataset", default="CustomDemo", help="Custom dataset name")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Dataset split")

    parser.add_argument("--root_dataset", default="data", help="Root dataset directory used by StrongSORT")
    parser.add_argument("--det_dir", default="data/detections", help="Directory to store MOT detections txt")
    parser.add_argument("--npy_dir", default=None, help="Directory to store StrongSORT .npy features")
    parser.add_argument("--result_dir", default="results", help="Directory to store tracking txt")
    parser.add_argument("--vis_dir", default="results/vis", help="Directory to store visualization videos")

    parser.add_argument("--bytetrack_root", default=str(bytetrack_root), help="Path to ByteTrack repository")
    parser.add_argument("--bytetrack_python", default=str(bytetrack_python), help="Python executable for ByteTrack")
    parser.add_argument("--bytetrack_exp", default="exps/example/mot/yolox_x_mix_det.py", help="ByteTrack exp file")
    parser.add_argument(
        "--bytetrack_ckpt",
        default=str(bytetrack_root / "pretrained" / "bytetrack_x_mot17.pth.tar"),
        help="ByteTrack checkpoint path",
    )
    parser.add_argument("--bytetrack_device", default="gpu", choices=["cpu", "gpu"], help="ByteTrack device")
    parser.add_argument("--bytetrack_fp16", action="store_true", help="Enable ByteTrack fp16")
    parser.add_argument("--bytetrack_fuse", action="store_true", help="Enable ByteTrack fuse")

    parser.add_argument("--fastreid_root", default=str(fastreid_root), help="Path to FastReID repository")
    parser.add_argument("--fastreid_python", default=str(fastreid_python), help="Python executable for FastReID")
    parser.add_argument(
        "--fastreid_config",
        default=str(fastreid_root / "configs" / "DukeMTMC" / "bagtricks_S50.yml"),
        help="FastReID config file",
    )
    parser.add_argument(
        "--fastreid_weights",
        default=str(fastreid_root / "weights" / "duke_bot_S50.pth"),
        help="FastReID model weights",
    )
    parser.add_argument("--fastreid_device", default="cuda", choices=["cpu", "cuda"], help="FastReID device")

    parser.add_argument("--result_stem", default=None, help="Base filename for result txt and visualization")
    parser.add_argument("--fps", type=int, default=None, help="Override fps written to seqinfo.ini and visualization")

    parser.add_argument("--ema", action="store_true", help="Enable EMA")
    parser.add_argument("--nsa", action="store_true", help="Enable NSA")
    parser.add_argument("--mc", action="store_true", help="Enable MC")
    parser.add_argument("--woc", action="store_true", help="Enable woC")
    parser.add_argument("--gsi", action="store_true", help="Enable GSI post-processing")
    parser.add_argument("--aflink", action="store_true", help="Enable AFLink post-processing")
    parser.add_argument("--ecc", action="store_true", help="Enable ECC")
    parser.add_argument("--aflink_weights", default="data/StrongSORT_data/AFLink_epoch20.pth", help="AFLink weights")
    parser.add_argument("--ecc_json", default=None, help="ECC json file path")

    parser.add_argument("--skip_extract", action="store_true", help="Skip frame extraction and seqinfo generation")
    parser.add_argument("--skip_detect", action="store_true", help="Skip ByteTrack detection generation")
    parser.add_argument("--skip_features", action="store_true", help="Skip FastReID feature extraction")
    parser.add_argument("--skip_track", action="store_true", help="Skip StrongSORT tracking")
    parser.add_argument("--skip_vis", action="store_true", help="Skip visualization")
    return parser.parse_args()


def run(cmd, cwd=None, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ffprobe_video(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return width, height, fps


def write_seqinfo(seq_dir, seq_name, fps, frame_count, width, height):
    text = "\n".join(
        [
            "[Sequence]",
            f"name={seq_name}",
            "imDir=img1",
            f"frameRate={fps}",
            f"seqLength={frame_count}",
            f"imWidth={width}",
            f"imHeight={height}",
            "imExt=.jpg",
        ]
    )
    (seq_dir / "seqinfo.ini").write_text(text + "\n")


def newest_detection(track_vis_dir, video_stem):
    matches = sorted(track_vis_dir.glob(f"*_det.txt"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No *_det.txt found in {track_vis_dir}")
    return matches[-1]


def run_strongsort(args, sequence_dir, detection_npy, result_txt):
    original_argv = sys.argv[:]
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    added_repo_root = False
    try:
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
            added_repo_root = True
        sys.argv = ["strongsort_pipeline", args.dataset, args.split]
        import opts as opts_module
        from deep_sort_app import run as strongsort_run

        opt = opts_module.opt
        opt.dataset = args.dataset
        opt.mode = args.split
        opt.root_dataset = args.root_dataset
        opt.dir_dataset = str(Path(args.root_dataset) / args.dataset / ("train" if args.split == "val" else "test"))
        opt.dir_dets = str(Path(detection_npy).parent)
        opt.dir_save = str(Path(result_txt).parent)
        opt.sequences = [args.seq]
        opt.BoT = True
        opt.ECC = args.ecc
        opt.NSA = args.nsa
        opt.EMA = args.ema
        opt.MC = args.mc
        opt.woC = args.woc
        opt.AFLink = args.aflink
        opt.GSI = args.gsi
        opt.path_AFLink = args.aflink_weights
        opt.nms_max_overlap = 1.0
        opt.min_confidence = 0.6
        opt.min_detection_height = 0
        opt.max_cosine_distance = 0.4
        if opt.MC:
            opt.max_cosine_distance += 0.05
        opt.nn_budget = 1 if opt.EMA else 100
        if opt.ECC:
            ecc_path = args.ecc_json or str(Path("data/StrongSORT_data") / f"{args.dataset}_ECC_{args.split}.json")
            with open(ecc_path) as f:
                opt.ecc = json.load(f)

        strongsort_run(
            sequence_dir=str(sequence_dir),
            detection_file=str(detection_npy),
            output_file=str(result_txt),
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=False,
        )

        if opt.AFLink:
            import torch
            from AFLink.AppFreeLink import AFLink, LinkData, PostLinker

            model = PostLinker()
            model.load_state_dict(torch.load(opt.path_AFLink, map_location="cpu"))
            dataset = LinkData("", "")
            linker = AFLink(
                path_in=str(result_txt),
                path_out=str(result_txt),
                model=model,
                dataset=dataset,
                thrT=(0, 30),
                thrS=75,
                thrP=0.05,
            )
            linker.link()

        if opt.GSI:
            from GSI import GSInterpolation

            GSInterpolation(
                path_in=str(result_txt),
                path_out=str(result_txt),
                interval=20,
                tau=10,
            )
    finally:
        if added_repo_root and sys.path and sys.path[0] == repo_root_str:
            sys.path.pop(0)
        sys.argv = original_argv


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    result_stem = args.result_stem or args.seq
    sequence_dir = Path(args.root_dataset) / args.dataset / args.split / args.seq
    img_dir = sequence_dir / "img1"
    det_dir = Path(args.det_dir)
    if args.npy_dir is None:
        npy_dir = Path("data/StrongSORT_data") / f"{args.dataset}_{args.split}_YOLOX+BoT"
    else:
        npy_dir = Path(args.npy_dir)
    result_dir = Path(args.result_dir)
    vis_dir = Path(args.vis_dir)
    detections_txt = det_dir / f"{args.seq}.txt"
    detection_npy = npy_dir / f"{args.seq}.npy"
    result_txt = result_dir / f"{result_stem}.txt"
    vis_video = vis_dir / f"{result_stem}.mp4"

    det_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        img_dir.mkdir(parents=True, exist_ok=True)
        width, height, fps_float = ffprobe_video(video_path)
        fps = args.fps or int(round(fps_float))
        run(["ffmpeg", "-y", "-i", str(video_path), str(img_dir / "%06d.jpg")])
        frame_count = len(list(img_dir.glob("*.jpg")))
        write_seqinfo(sequence_dir, args.seq, fps, frame_count, width, height)

    if not args.skip_detect:
        bytetrack_root = Path(args.bytetrack_root).resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(bytetrack_root)
        cmd = [
            args.bytetrack_python,
            "tools/demo_track.py",
            "video",
            "-f",
            args.bytetrack_exp,
            "-c",
            args.bytetrack_ckpt,
            "--path",
            str(video_path),
            "--device",
            args.bytetrack_device,
            "--save_result",
        ]
        if args.bytetrack_fp16:
            cmd.append("--fp16")
        if args.bytetrack_fuse:
            cmd.append("--fuse")
        run(cmd, cwd=str(bytetrack_root), env=env)
        latest_det = newest_detection(bytetrack_root / "YOLOX_outputs" / "yolox_x_mix_det" / "track_vis", video_path.stem)
        shutil.copy2(latest_det, detections_txt)

    if not args.skip_features:
        cmd = [
            args.fastreid_python,
            "tools/extract_fastreid_features.py",
            "--fastreid_root",
            args.fastreid_root,
            "--config_file",
            args.fastreid_config,
            "--weights",
            args.fastreid_weights,
            "--sequence_dir",
            str(sequence_dir),
            "--detections_txt",
            str(detections_txt),
            "--output_npy",
            str(detection_npy),
            "--device",
            args.fastreid_device,
        ]
        run(cmd, cwd=str(repo_root))

    if not args.skip_track:
        run_strongsort(args, sequence_dir, detection_npy, result_txt)

    if not args.skip_vis:
        fps_for_vis = args.fps
        if fps_for_vis is None:
            _, _, fps_float = ffprobe_video(video_path)
            fps_for_vis = int(round(fps_float))
        cmd = [
            sys.executable,
            "tools/visualize_results.py",
            "--sequence_dir",
            str(sequence_dir),
            "--result_txt",
            str(result_txt),
            "--out_video",
            str(vis_video),
            "--fps",
            str(fps_for_vis),
        ]
        run(cmd, cwd=str(repo_root))
        h264_tmp = vis_video.with_name(f"{vis_video.stem}.tmp_h264{vis_video.suffix}")
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(vis_video),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(h264_tmp),
            ],
            cwd=str(repo_root),
        )
        h264_tmp.replace(vis_video)

    print("Pipeline finished.")
    print(f"Detections txt: {detections_txt}")
    print(f"Detection+features npy: {detection_npy}")
    print(f"Tracking result txt: {result_txt}")
    print(f"Visualization video: {vis_video}")


if __name__ == "__main__":
    main()
