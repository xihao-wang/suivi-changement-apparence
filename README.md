# suivi-changement-apparence

StrongSORT-based tracking project for **appearance change** analysis on custom videos.

This repository is mainly used in an **offline pipeline**:

```text
video -> frames (img1) -> detections (.txt) -> detections + features (.npy) -> tracking (.txt) -> visualization (.mp4)
```

The codebase contains:
- a custom video pipeline,
- several appearance-memory modifications on top of StrongSORT,
- ablation switches to test each added technique separately,
- debug tools to inspect per-frame matching.

---

## 1. What This Project Uses

For the current project, the important blocks are:

- **ByteTrack**: person detections
- **FastReID / BoT**: appearance features
- **DeepSORT / StrongSORT core**: ID association
- **Our modifications**:
  - STM + LTM
  - delayed long-memory initialization
  - memory-aware matching
  - top-k matching
  - appearance trend

---

## 2. Minimal Environment

Recommended environments:

- `strongsort` for tracking / visualization
- `fastreid` for feature extraction
- `bytetrack-gpu` for detection

Minimal Python packages for the tracking part:

```bash
conda create -n strongsort python=3.8 -y
conda activate strongsort
pip install torch torchvision torchaudio
pip install opencv-python scipy scikit-learn
```

If you only want to run tracking from an existing `.npy`, only the `strongsort` environment is needed.

---

## 3. Expected Input Format

### Sequence folder

Each custom sequence should follow the MOT-style layout:

```text
data/CustomDemo/test/<SEQ>/
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── seqinfo.ini
```

Example `seqinfo.ini`:

```ini
[Sequence]
name=YT-03
imDir=img1
frameRate=30
seqLength=2169
imWidth=1148
imHeight=2038
imExt=.jpg
```

### Detection + feature file

StrongSORT does **not** read the raw detection `.txt` directly.  
It reads a `.npy` file containing:

- MOT detection columns
- one appearance feature per detection

Example:

```text
data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy
```

---

## 4. Recommended Workflow

### Option A: one-line custom pipeline

This is the easiest entry point when starting from a raw video:

```bash
python3 tools/run_custom_video_pipeline.py \
  --video downloads/333.mp4 \
  --seq YT-03 \
  --result_dir results/ours \
  --vis_dir results/vis \
  --result_stem YT-03_run
```

Useful flags:

- `--skip_extract`: skip frame extraction
- `--skip_detect`: skip ByteTrack detection
- `--skip_features`: skip FastReID feature extraction
- `--skip_track`: skip StrongSORT tracking
- `--skip_vis`: skip video rendering

Example when frames, detections and features are already prepared:

```bash
python3 tools/run_custom_video_pipeline.py \
  --video downloads/333.mp4 \
  --seq YT-03 \
  --result_dir results/ours \
  --vis_dir results/vis \
  --result_stem YT-03_run \
  --skip_extract \
  --skip_detect \
  --skip_features
```

### Option B: tracking only from an existing `.npy`

If the sequence folder and `.npy` already exist, run StrongSORT directly:

```bash
python3 strong_sort.py CustomDemo test --BoT --root_dataset data --dir_save results/ours
```

This is the preferred entry point for:

- repeated experiments,
- ablation studies,
- debugging the tracking logic only.

---

## 5. Manual Workflow

Use this only if you want to run each stage manually.

### Step 1: extract frames

```bash
mkdir -p data/CustomDemo/test/YT-03/img1
ffmpeg -i downloads/333.mp4 data/CustomDemo/test/YT-03/img1/%06d.jpg
```

### Step 2: generate detections with ByteTrack

Run ByteTrack in your local ByteTrack repository and save the MOT-style detection file.

Expected output example:

```text
data/detections/YT-03.txt
```

### Step 3: extract ReID features and build `.npy`

```bash
python tools/extract_fastreid_features.py \
  --fastreid_root <FastReID_ROOT> \
  --config_file <FastReID_CONFIG> \
  --weights <FastReID_WEIGHTS> \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detections_txt data/detections/YT-03.txt \
  --output_npy data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy \
  --device cuda
```

### Step 4: run tracking

```bash
python3 strong_sort.py CustomDemo test --BoT --root_dataset data --dir_save results/ours
```

### Step 5: render the tracking result

```bash
python3 tools/visualize_results.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --result_txt results/ours/YT-03.txt \
  --out_video results/vis/YT-03.mp4 \
  --fps 30
```

---

## 6. Ablation Cases

The project supports explicit ablation cases through:

```bash
--ablation_case
```

Available cases:

1. `1_bot`  
   BoT only

2. `2_stm_ltm`  
   BoT + STM + LTM

3. `3_stm_ltm_memory_init`  
   BoT + STM + LTM + delayed long-memory initialization

4. `4_stm_ltm_memory_aware`  
   BoT + STM + LTM + memory-aware matching

5. `5_stm_ltm_memory_aware_topk`  
   BoT + STM + LTM + memory-aware matching + top-k

6. `6_stm_ltm_memory_aware_trend`  
   BoT + STM + LTM + memory-aware matching + appearance trend

7. `7_stm_ltm_memory_init_memory_aware`  
   BoT + STM + LTM + memory init control + memory-aware matching

8. `8_stm_ltm_memory_init_memory_aware_topk`  
   BoT + STM + LTM + memory init control + memory-aware matching + top-k

9. `9_full`  
   BoT + STM + LTM + memory init control + memory-aware matching + top-k + trend

Example:

```bash
python3 strong_sort.py CustomDemo test \
  --BoT \
  --ablation_case 8_stm_ltm_memory_init_memory_aware_topk \
  --root_dataset data \
  --dir_save results/ablation_8
```

Recommended baselines:

- external baseline: `1_bot`
- internal baseline: `2_stm_ltm`

---

## 7. Debug Tools

### Interactive matching viewer

Use the viewer to inspect:

- detections,
- tracks,
- matches / unmatched elements,
- appearance cost,
- trend cost,
- final cost,
- gated cost,
- ambiguity warnings.

```bash
python3 tools/debug_match_viewer.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detection_file data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy
```

### Keyboard shortcuts

- `a`: previous frame
- `d`: next frame

---

## 8. Output Files

Typical outputs:

```text
results/ours/YT-03_run.txt
results/vis/YT-03_run.mp4
```

The result `.txt` follows MOT format:

```text
frame,id,x,y,w,h,1,-1,-1,-1
```

---

## 9. Practical Notes

- For custom videos, use `CustomDemo test` and `--root_dataset data`.
- Do not enable `--ECC` unless the corresponding ECC json exists.
- If you already have the `.npy`, avoid rerunning detection and feature extraction.
- For ablation studies, prefer `strong_sort.py` over the full pipeline script.

---

## 10. Original StrongSORT Context

This repository started from StrongSORT, but the current work focuses mainly on:

- BoT appearance features,
- DeepSORT-style association,
- memory-based modifications for appearance change.

If you need the original benchmark-oriented StrongSORT usage on MOT17 / MOT20, refer to the upstream StrongSORT documentation.
