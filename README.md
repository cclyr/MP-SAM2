# MP-SAM2

Official implementation of **A Multi-Prompt Transfer Learning Framework Based on
Segment Anything Model 2 for Microcrack Segmentation in Engineered Cementitious
Composite Images and Videos**.

MP-SAM2 combines an automatically generated threshold mask, a self-perceptive
prompt box (SP-Box), and Euclidean-distance-transform (EDT) positive/negative
points with progressive SAM2 transfer. The same image-domain model initializes
SAM2 memory propagation for video segmentation. No manual click, box, or mask is
required at inference time.

> Release status: the source repository is private during peer review. Image and
> video data are not included. Model weights will be published separately after
> acceptance; local checkpoints are intentionally ignored by Git.

## Results

All image results use the same independent 71-image test set. Few-shot values are
the mean and standard deviation across seeds 42, 123, and 2026.

| Training images | Accuracy (%) | IoU (%) | Dice (%) |
|---:|---:|---:|---:|
| 5 | 98.73 ± 0.05 | 81.73 ± 0.67 | 89.68 ± 0.46 |
| 10 | 98.79 ± 0.07 | 82.42 ± 1.00 | 90.13 ± 0.67 |
| 20 | 98.89 ± 0.06 | 83.89 ± 0.82 | 91.06 ± 0.52 |
| 50 | 98.98 ± 0.02 | 85.22 ± 0.18 | 91.87 ± 0.12 |
| Full-data reference | 99.05 | 86.43 | 92.59 |

## Repository layout

```text
MP-SAM2/
├── mp_sam2/              # prompts, model loading, training, metrics, video pipeline
├── scripts/
│   ├── train.py           # 5/10/50/full-data training
│   ├── infer_image.py     # single-image or directory inference
│   ├── infer_video.py     # automatic first-frame initialization + memory propagation
│   └── evaluate.py        # image-level segmentation metrics
├── configs/               # paper hyperparameters
├── sam2/                  # vendored SAM2 runtime, configs, and notices
├── checkpoints/           # local-only checkpoints; excluded from Git
├── data/                  # dataset layout only; no images or videos
└── results/               # local experiment tables; excluded from Git
```

## Installation

The experiments were run on an NVIDIA GPU. Python 3.10 and a CUDA-enabled PyTorch
build are recommended.

```bash
conda create -n mp-sam2 python=3.10 -y
conda activate mp-sam2

# Select the PyTorch command appropriate for your CUDA installation.
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Alternatively:

```bash
conda env create -f configs/environment.yml
conda activate mp-sam2
pip install -e .
```

The optional SAM2 connected-components CUDA extension is not required by the
included MP-SAM2 scripts. Video inference disables that optional post-processing
path for cross-platform compatibility.

## Checkpoints

The local release directory contains:

| File | Training budget | Seed | Publication status |
|---|---:|---:|---|
| `checkpoints/mp_sam2_5_seed42.torch` | 5 images | 42 | local only |
| `checkpoints/mp_sam2_10_seed42.torch` | 10 images | 42 | local only |
| `checkpoints/mp_sam2_50_seed42.torch` | 50 images | 42 | local only |
| `checkpoints/mp_sam2_full.torch` | full data | original run | local only |
| `checkpoints/sam2.1_hiera_large.pt` | Meta pretraining | n/a | local only; not redistributed |

The individual few-shot checkpoints use seed 42 as representative released
models. The few-shot values reported in the paper are still three-seed means.

For a fresh clone, download the SAM2 Hiera Large base checkpoint from the official
[Meta SAM2 repository](https://github.com/facebookresearch/sam2) and place it at:

```text
checkpoints/sam2.1_hiera_large.pt
```

The fine-tuned checkpoint contains the complete model state and can be used for
image inference directly. Video inference also requests the base checkpoint to
construct the SAM2 video predictor before loading the transferred state.

## Data layout

The data are not distributed in this repository. Prepare paired PNG files as:

```text
dataset/
├── images/
│   ├── 01/01_01.png
│   └── 02/02_01.png
└── masks/
    ├── 01/01_01_enhanced.png
    └── 02/02_01_enhanced.png
```

Masks must be binary, with non-zero crack pixels. Do not mix images from the same
specimen or loading sequence across development and test sets.

## Image inference

Run on one image:

```bash
python scripts/infer_image.py \
  --input path/to/image.png \
  --checkpoint checkpoints/mp_sam2_full.torch \
  --output outputs/image
```

Run on a directory by passing the directory to `--input`. For each image the
script writes a binary mask, a red overlay, and `inference.json` containing the
automatically selected prompt information.

Default prompt parameters reproduce the paper configuration:

- threshold offset: 0;
- SP-Box expansion ratio: 0.10;
- EDT positive points: 4;
- far-background negative points: 2;
- minimum negative-point distance: 6 pixels.

These values can be changed with the command-line options shown by:

```bash
python scripts/infer_image.py --help
```

## Video inference

Video inference first applies MP-SAM2 to the first RGB frame. The predicted mask
is then stored in the SAM2 memory bank and propagated to later frames.

```bash
python scripts/infer_video.py \
  --video path/to/input.mp4 \
  --base-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --checkpoint checkpoints/mp_sam2_full.torch \
  --output outputs/video_overlay.mp4 \
  --contrast 1.0
```

The command writes the overlay video, one binary mask per frame, and a JSON
metadata file. `--contrast 1.0` leaves contrast unchanged; a fixed value greater
than 1 can be used when reproducing a contrast-enhanced video protocol. The same
value is applied to every frame and does not use annotations.

## Training

Few-shot subsets are selected deterministically and balanced across the specified
categories. For example, the 5-image seed-42 run is:

```bash
python scripts/train.py \
  --images dataset/images \
  --masks dataset/masks \
  --categories 01 02 \
  --budget 5 \
  --seed 42 \
  --epochs 80 \
  --batch-size 2 \
  --base-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --output runs/mp_sam2_5_seed42.torch
```

Change `--budget` to `10` or `50` for the other released few-shot checkpoints.
To train on all valid pairs with the original 3,000/4,000/3,000-stage schedule:

```bash
python scripts/train.py \
  --images dataset/images \
  --masks dataset/masks \
  --categories 01 02 \
  --budget full \
  --seed 42 \
  --base-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --full-stage-steps 3000 4000 3000 \
  --output runs/mp_sam2_full.torch
```

The three stages train (1) the prompt encoder and mask decoder, (2) those modules
plus image-encoder blocks 30–31 and the neck, and (3) the full model. The hybrid
loss is `BCE + 2×Dice + Focal + 1.5×IoU`.

## Evaluation

```bash
python scripts/evaluate.py \
  --predictions outputs/image \
  --masks path/to/test/masks \
  --output outputs/test_metrics.json
```

Accuracy, precision, recall, F1, IoU, and Dice are first calculated per image and
then averaged, matching the image-level reporting protocol.

## Reproducibility notes

- Input image patches are resized to 1024 × 1024 by the SAM2 transform.
- Few-shot experiments use 80 equivalent epochs and stage ratios 30%/40%/30%.
- Optimizer: AdamW, learning rate `1e-5`, weight decay `0.01`, batch size `2`.
- Few-shot seeds: `42`, `123`, and `2026`.
- The fixed test set is never used for subset selection or model training.
- The repository contains no raw image data, video data, or human annotations.

## License and acknowledgment

Code is released under Apache License 2.0. The vendored SAM2 source retains
Meta's copyright notices and license. See [`LICENSE`](LICENSE) and the
third-party notices in [`sam2/licenses/`](sam2/licenses/).

## Citation

The final bibliographic record will be added after publication. Until then,
please cite the manuscript title shown at the top of this README.
