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
- debug tools to inspect per-frame matching,
- a learned temporal scorer trained on pair-level datasets.

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
  - learned temporal scorer

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

## 6. Ablation / Technique Switches

The modified framework is controlled by combinable flags:

- `--ltm_stm`: enable short-term and long-term memory
- `--memory_init`: enable delayed / gated long-memory writing
- `--memory_aware`: enable memory-aware matching
- `--topk`: enable top-k aggregation in memory matching
- `--trend`: enable appearance trend
- `--full`: enable all modifications together

Use one of the two entry points below:

- **Direct tracking (`strong_sort.py`)**  
  Use this when the sequence folder and the `.npy` file already exist.  
  This is the best option for repeated experiments and ablation studies.

- **Offline wrapper (`run_custom_video_pipeline.py`)**  
  Use this when you want to keep the same command style as the full custom-video pipeline.  
  If frames, detections and features are already ready, add:
  - `--skip_extract`
  - `--skip_detect`
  - `--skip_features`

Typical usage:

### Example 1: direct tracking from an existing `.npy`

```bash
python3 strong_sort.py CustomDemo test \
  --BoT \
  --ltm_stm \
  --memory_init \
  --root_dataset data \
  --dir_save results/memory_init
```

### Example 2: same experiment through the offline wrapper

```bash
python3 tools/run_custom_video_pipeline.py \
  --video downloads/333.mp4 \
  --seq YT-03 \
  --result_dir results/ours/ablation \
  --vis_dir results/vis/ablation \
  --result_stem YT-03_memory_init \
  --skip_extract \
  --skip_detect \
  --skip_features \
  --ltm_stm \
  --memory_init
```

You can replace `--ltm_stm --memory_init` by any other combination, for example:

- baseline: no extra switch
- `--ltm_stm`
- `--ltm_stm --memory_aware`
- `--ltm_stm --memory_aware --topk`
- `--ltm_stm --trend`
- `--full`

Canonical combinations:

```sh
# [1] BoT baseline
--BoT

# [3] BoT + STM + LTM
--BoT --ltm_stm

# [4] BoT + STM + LTM + memory_init
--BoT --ltm_stm --memory_init

# [5] BoT + STM + LTM + memory_aware
--BoT --ltm_stm --memory_aware

# [6] BoT + STM + LTM + memory_aware + topk
--BoT --ltm_stm --memory_aware --topk

# [7] BoT + STM + LTM + trend
--BoT --ltm_stm --trend

# [8] BoT + STM + LTM + memory_init + trend
--BoT --ltm_stm --memory_init --trend

# [9] BoT + STM + LTM + memory_aware + trend
--BoT --ltm_stm --memory_aware --trend

# [10] BoT + STM + LTM + memory_aware + topk + trend
--BoT --ltm_stm --memory_aware --topk --trend

# [11] BoT + STM + LTM + memory_init + memory_aware
--BoT --ltm_stm --memory_init --memory_aware

# [12] BoT + STM + LTM + memory_init + memory_aware + topk
--BoT --ltm_stm --memory_init --memory_aware --topk

# [13] BoT + STM + LTM + memory_init + memory_aware + trend
--BoT --ltm_stm --memory_init --memory_aware --trend

# [14] BoT + STM + LTM + memory_init + memory_aware + topk + trend
--BoT --ltm_stm --memory_init --memory_aware --topk --trend

# Full
--BoT --full
```

Recommended baselines:

- external baseline: `BoT`
- internal baseline: `BoT + STM + LTM`

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

The viewer can also use the same technique switches as the main tracking code:

- `--BoT`
- `--ltm_stm`
- `--memory_init`
- `--memory_aware`
- `--topk`
- `--trend`
- `--full`

If you do not manually override thresholds such as `min_confidence` or
`max_cosine_distance`, the viewer now follows the values parsed from
`opts.py`, so it is easier to keep it consistent with the main pipeline.

```bash
python3 tools/debug_match_viewer.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detection_file data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy
```

Example with the same switches as a tracking run:

```bash
python3 tools/debug_match_viewer.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detection_file data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy \
  --BoT \
  --ltm_stm \
  --memory_init \
  --memory_aware \
  --trend
```

### Keyboard shortcuts

- `a`: previous frame
- `d`: next frame

---

## 8. Learned Temporal Scorer

The repository now contains a **feature-level temporal scorer** for MOT matching.

### Current design

The current temporal model is implemented in:

```text
deep_sort/temporal_model.py
```

It is **not** a full reproduction of the 2024 patch-token TIE tracker.  
It is a lightweight scorer adapted to the current StrongSORT pipeline:

- `template/reference` token: a prototype built from the selected history features
- `query` token: current `det_feat`
- `history` tokens: a selectable subset of
  - `df_t`
  - `df_t-i`
  - `df_t-2i`

The model uses:

- one template self-attention branch
- one history attention branch
- explicit temporal / type embeddings on history tokens
- an MLP scorer on top of the temporal outputs

### Training sample format

Pair datasets are stored as `.npz` files built from:

- current detection feature `det_feat`
- ordered matched-detection history
  - `df_t`
  - `df_t_i`
  - `df_t_2i`
- binary label `label`

This data is built with:

```bash
python3 tools/build_temporal_pairs.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detection_file data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy \
  --result_txt results/ours/sep_combi/YT-03_case14.txt \
  --output_npz data/temporal_pairs/YT-03_pairs.npz \
  --dataset CustomDemo \
  --split test \
  --BoT \
  --ltm_stm \
  --memory_init \
  --memory_aware \
  --topk \
  --temporal_stride 2
```

### Train the temporal scorer

Standard training command:

```bash
python3 tools/train_temporal_model.py \
  --train_pair_npz $(printf "data/temporal_pairs/%s_pairs.npz " new-1 new-2 new-3 new-5 new-6 new-8 new-9) \
  --val_pair_npz data/temporal_pairs/YT-03_pairs.npz \
  --save_dir data/checkpoints/temporal_model_multi \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 50 \
  --hidden_dim 256 \
  --num_heads 4
```

### Select which history tokens are used during training

`tools/train_temporal_model.py` supports **training-time ablation** through:

```bash
--history_indices
```

History slot convention:

- `0 = df_t`
- `1 = df_t-i`
- `2 = df_t-2i`

Examples:

Use all three:

```bash
--history_indices 0,1,2
```

Use `[df_t, df_t-i]`:

```bash
--history_indices 0,1
```

Use `[df_t, df_t-2i]`:

```bash
--history_indices 0,2
```

Complete examples:

```bash
python3 tools/train_temporal_model.py \
  --train_pair_npz $(printf "data/temporal_pairs/%s_pairs.npz " new-1 new-2 new-3 new-5 new-6 new-8 new-9) \
  --val_pair_npz data/temporal_pairs/YT-03_pairs.npz \
  --save_dir data/checkpoints/ablation_hist012 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 50 \
  --hidden_dim 256 \
  --num_heads 4 \
  --history_indices 0,1,2
```

```bash
python3 tools/train_temporal_model.py \
  --train_pair_npz $(printf "data/temporal_pairs/%s_pairs.npz " new-1 new-2 new-3 new-5 new-6 new-8 new-9) \
  --val_pair_npz data/temporal_pairs/YT-03_pairs.npz \
  --save_dir data/checkpoints/ablation_hist01 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 50 \
  --hidden_dim 256 \
  --num_heads 4 \
  --history_indices 0,1
```

```bash
python3 tools/train_temporal_model.py \
  --train_pair_npz $(printf "data/temporal_pairs/%s_pairs.npz " new-1 new-2 new-3 new-5 new-6 new-8 new-9) \
  --val_pair_npz data/temporal_pairs/YT-03_pairs.npz \
  --save_dir data/checkpoints/ablation_hist02 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 50 \
  --hidden_dim 256 \
  --num_heads 4 \
  --history_indices 0,2
```

### Inspect the learned scorer in the viewer

The GUI viewer can display:

- learned temporal score matrix
- learned attention on history tokens

Example:

```bash
python3 tools/debug_match_viewer.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detection_file data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy \
  --BoT \
  --ltm_stm \
  --memory_init \
  --memory_aware \
  --topk \
  --learned_temporal \
  --temporal_model_ckpt data/checkpoints/temporal_model_multi/best.pt
```

### Notes

- `--history_len` is still accepted, but for controlled ablations `--history_indices` is preferred.
- The learned temporal scorer is currently designed for **pair scoring**, not for end-to-end box prediction.
- The current model is a **feature-level approximation** of temporal attention, not the original image-patch TIE architecture.

---

## 9. Output Files

Typical outputs:

```text
results/ours/YT-03_run.txt
results/vis/YT-03_run.mp4
```

The result `.txt` follows MOT format:

```text
frame,id,x,y,w,h,1,-1,-1,-1
```
