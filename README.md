# suivi-changement-apparence
## Data&Model Preparation

1. Download MOT17 & MOT20 from the [official website](https://motchallenge.net/).

   ```
   data/MOTChallenge
   ├── MOT17
   	│   ├── test
   	│   └── train
   └── MOT20
       ├── test
       └── train
   ```

2. Download our prepared [data](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_?usp=sharing) in Google disk (or [baidu disk](https://pan.baidu.com/s/1EtBbo-12xhjsqW5x-dYX8A?pwd=sort) with code "sort")

   ```
   path_to_dataspace
   ├── AFLink_epoch20.pth  # checkpoints for AFLink model
   ├── MOT17_ECC_test.json  # CMC model
   ├── MOT17_ECC_val.json  # CMC model
   ├── MOT17_test_YOLOX+BoT  # detections + features
   ├── MOT17_test_YOLOX+simpleCNN  # detections + features
   ├── MOT17_trainval_GT_for_AFLink  # GT to train and eval AFLink model
   ├── MOT17_val_GT_for_TrackEval  # GT to eval the tracking results.
   ├── MOT17_val_YOLOX+BoT  # detections + features
   ├── MOT17_val_YOLOX+simpleCNN  # detections + features
   ├── MOT20_ECC_test.json  # CMC model
   ├── MOT20_test_YOLOX+BoT  # detections + features
   ├── MOT20_test_YOLOX+simpleCNN  # detections + features
   ```

3. Set the paths of your dataset and other files in "opts.py", i.e., root_dataset, path_AFLink, dir_save, dir_dets, path_ECC. 

Note: If you want to generate ECC results, detections and features by yourself, please refer to the [Auxiliary tutorial](https://github.com/dyhBUPT/StrongSORT/blob/master/others/AuxiliaryTutorial.md).

## Requirements

- pytorch
- opencv
- scipy
- sklearn

For example, we have tested the following commands to create an environment for StrongSORT:

```shell
conda create -n strongsort python=3.8 -y
conda activate strongsort
pip3 install torch torchvision torchaudio
pip install opencv-python
pip install scipy
pip install scikit-learn
```

once the environment is properly configured, we can use directly
```shell
conda activate strongsort
```

## Tracking

- **Run DeepSORT on MOT17-val**

  ```shell
  python strong_sort.py MOT17 val
  ```

- **Run StrongSORT on MOT17-val**

  ```shell
  python strong_sort.py MOT17 val --BoT --ECC --NSA --EMA --MC --woC
  ```

- **Run StrongSORT++ on MOT17-val**

  ```shell
  python strong_sort.py MOT17 val --BoT --ECC --NSA --EMA --MC --woC --AFLink --GSI
  ```

- **Run StrongSORT++ on MOT17-test**

  ```shell
  python strong_sort.py MOT17 test --BoT --ECC --NSA --EMA --MC --woC --AFLink --GSI
  ```

- **Run StrongSORT++ on MOT20-test**

  ```shell
  python strong_sort.py MOT20 test --BoT --ECC --NSA --EMA --MC --woC --AFLink --GSI
  ```

## Visualization
For enregister the video, enter

```shell
python tools/visualize_results.py \
--sequence_dir ./data/MOTChallenge/MOT17/train/MOT17-02-FRCNN \
--result_txt ./results/MOT17-02-FRCNN.txt \
--out_video ./results/vis/MOT17-02-FRCNN.mp4 --fps 25
```
For watching:
```shell
ffplay results/vis/MOT17-XX-FRCNN.mp4
```

## Run On A Custom Video

This repository can also be used on a single custom video, but the full pipeline is offline:

```text
video -> frames (img1) -> detections (.txt) -> detections + features (.npy) -> StrongSORT -> result .txt -> visualization
```

The steps below summarize the workflow that has been tested in this project.

### 1. Prepare a MOT-style sequence

Assume the input video is `downloads/222.mp4` and the new sequence name is `YT-02`.

Create the sequence folder:

```shell
mkdir -p data/CustomDemo/test/YT-02/img1
```

Extract frames:

```shell
ffmpeg -i downloads/222.mp4 data/CustomDemo/test/YT-02/img1/%06d.jpg
```

Count frames:

```shell
find data/CustomDemo/test/YT-02/img1 -type f | wc -l
```

Inspect the video metadata:

```shell
ffprobe downloads/222.mp4
```

Create `data/CustomDemo/test/YT-02/seqinfo.ini` using the frame count and resolution reported above:

```ini
[Sequence]
name=YT-02
imDir=img1
frameRate=30
seqLength=1421
imWidth=1148
imHeight=2038
imExt=.jpg
```

### 2. Generate detections with ByteTrack

This project expects MOT-style detections before StrongSORT runs. We used ByteTrack for this step.

Requirements:

- a local ByteTrack clone
- a ByteTrack checkpoint such as `pretrained/bytetrack_x_mot17.pth.tar`

Next, switch to the `ByteTrack` repository and run detection on the custom video:

```shell
cd <ByteTrack_ROOT>
PYTHONPATH=$(pwd) python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --path <VIDEO_PATH> \
  --device cpu \
  --save_result
```

Note:

- In this setup, `tools/demo_track.py` was modified to save a second file named `*_det.txt`.
- This file contains MOT-style detections:
  `frame,-1,x,y,w,h,score,-1,-1,-1`

Copy the generated detection file to a stable location:

```shell
cd <THIS_REPO_ROOT>
mkdir -p data/detections
cp <ByteTrack_ROOT>/YOLOX_outputs/yolox_x_mix_det/track_vis/<timestamp>_det.txt \
   data/detections/YT-02.txt
```

### 3. Extract FastReID features and build the `.npy`

StrongSORT does not read the detection `.txt` directly. It expects a `.npy` file containing:

- MOT detection columns
- appearance features for each box

This repository includes a helper script:

```shell
tools/extract_fastreid_features.py
```

Requirements:

- a local FastReID clone
- a FastReID config, e.g. `configs/DukeMTMC/bagtricks_S50.yml`
- matching weights, e.g. `weights/duke_bot_S50.pth`

Next, switch back to this repository, activate the `fastreid` environment, and run feature extraction:

```shell
conda activate fastreid
cd <THIS_REPO_ROOT>

python tools/extract_fastreid_features.py \
  --fastreid_root <FastReID_ROOT> \
  --config_file <FastReID_ROOT>/configs/DukeMTMC/bagtricks_S50.yml \
  --weights <FastReID_ROOT>/weights/duke_bot_S50.pth \
  --sequence_dir data/CustomDemo/test/YT-02 \
  --detections_txt data/detections/YT-02.txt \
  --output_npy data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-02.npy \
  --device cpu
```

If a CUDA-enabled FastReID environment is available, `--device cuda` can be used instead.

### 4. Register the custom sequence in `opts.py`

Add the custom dataset and sequence to `opts.py`:

```python
'CustomDemo': {
    'test': [
        'YT-02'
    ]
}
```

When testing several custom videos, add them all to the same list:

```python
'CustomDemo': {
    'test': [
        'YT-01',
        'YT-02'
    ]
}
```

### 5. Run StrongSORT on the custom sequence

Because the custom data lives under `data/CustomDemo/...`, override the default dataset root:

```shell
python strong_sort.py CustomDemo test --BoT --root_dataset data
```
or,
```shell
python3 strong_sort.py CustomDemo test --BoT --root_dataset data --dir_save results/bot
```

This writes the tracking result to:

```text
results/YT-02.txt or results/bot/YT-02.txt
```

Complete example
```shell
python3 strong_sort.py CustomDemo test --BoT --root_dataset data --dir_save results/bot
python3 strong_sort.py CustomDemo test --BoT --EMA --root_dataset data --dir_save results/bot_ema
python3 strong_sort.py CustomDemo test --BoT --NSA --EMA --MC --woC --root_dataset data --dir_save results/strongsort
```


Notes:

- Do not use `--ECC` unless you have generated the corresponding `CustomDemo_ECC_test.json`.
- Do not use `val` unless you have explicitly prepared the custom sequence under a validation split.

### 6. Visualize the final tracking result

```shell
python tools/visualize_results.py \
  --sequence_dir ./data/CustomDemo/test/YT-02 \
  --result_txt ./results/YT-02.txt \
  --out_video ./results/vis/YT-02.mp4 \
  --fps 30
```

Convert the video to h264 format
```shell
ffmpeg -i ./results/vis/YT-02_bot_ema.mp4 -c:v libx264 -pix_fmt yuv420p ./results/vis/YT-02_bot_ema_h264.mp4
```

Then play the video (can be played by mp4 player):

```shell
ffplay ./results/vis/YT-02.mp4
```

### 7. With one line command


Put your own video into `downloads/`, and give a name for the sequence. 
`--result_dir results/xxx`: result ficher folder,for example `--result_dir results/strong_gsi`
`result_stem YT-xx_xxx`: generated video's name , for example `result_stem YT-03_strong_gsi`

```shell
python3 tools/run_custom_video_pipeline.py \
  --video downloads/xxx.mp4 \
  --seq YT-xx \
  --result_dir results/xxx \
  --vis_dir results/vis \
  --result_stem YT-xx_xxxx \
  --ema --nsa --mc --woc --gsi
```
also, if you want to skip some steps,

```shell
python3 ...
  -- ...
  -- ...
  --skip_extract \
  --skip_detect \
  --skip_features
```


### Summary

For each new custom video, the minimum repeatable workflow is:

1. Extract frames into `data/CustomDemo/test/<SEQ>/img1`
2. Create `seqinfo.ini`
3. Run ByteTrack and save `<SEQ>.txt` detections
4. Run `tools/extract_fastreid_features.py` to build `<SEQ>.npy`
5. Add `<SEQ>` in `opts.py`
6. Run `python strong_sort.py CustomDemo test --BoT --root_dataset data`
7. Visualize `results/<SEQ>.txt`
