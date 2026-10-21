# MP-SAM2

Official implementation of **A Multi-Prompt Transfer Learning Framework Based on
Segment Anything Model 2 for Microcrack Segmentation in Engineered Cementitious
Composite Images and Videos**.

MP-SAM2 combines an automatically generated threshold mask, a self-perceptive
prompt box (SP-Box), and Euclidean-distance-transform (EDT) positive/negative
points with progressive SAM2 transfer. The same image-domain model initializes
SAM2 memory propagation for video segmentation. No manual click, box, or mask is
required at inference time.

Source code, training configurations, and fine-tuned model weights are publicly
available. Download the weights from
[Hugging Face](https://huggingface.co/ccllr/MP-SAM2).
Image and video datasets are not included.

## Results

The following reported results use the independent 71-image test set. Few-shot
values are the mean and standard deviation over repeated training runs.
Released files contain individual trained models.

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
│   ├── train.py           # few-shot or full-data training
│   ├── infer_image.py     # single-image or directory inference
│   ├── infer_video.py     # automatic first-frame initialization + memory propagation
│   ├── evaluate.py        # image-level segmentation metrics
│   └── download_checkpoints.py # download the released weights
├── configs/               # YAML settings read by training/inference scripts
├── sam2/                  # vendored SAM2 runtime, configs, and notices
├── checkpoints/           # weight download instructions and SHA-256 checksums
└── data/                  # dataset layout only; no images or videos
```

## Installation

The experiments were run on an NVIDIA GPU. Python 3.10 and a CUDA-enabled PyTorch
build are recommended. The commands below target the bundled SAM2 runtime;
install this repository instead of installing a separate latest SAM2 package.

```bash
git clone https://github.com/cclyr/MP-SAM2.git
cd MP-SAM2

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

Download a fine-tuned checkpoint from
[Hugging Face](https://huggingface.co/ccllr/MP-SAM2)
and place it in `checkpoints/`:

| File | Training budget | Download |
|---|---:|---|
| `mp_sam2_5.torch` | 5 images | [download](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_5.torch) |
| `mp_sam2_10.torch` | 10 images | [download](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_10.torch) |
| `mp_sam2_50.torch` | 50 images | [download](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_50.torch) |
| `mp_sam2_full.torch` | 186 images (full data) | [download](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_full.torch) |

Each checkpoint is approximately 898 MB and contains the complete model state.
For the full-data checkpoint, use the download helper:

```bash
python scripts/download_checkpoints.py --variant full
```

SHA-256 checksums are provided in
[`checkpoints/SHA256SUMS`](checkpoints/SHA256SUMS) and in the
[Hugging Face model repository](https://huggingface.co/ccllr/MP-SAM2/blob/main/SHA256SUMS).

Use `--variant 5`, `10`, `50`, or `all` to select the other files. Each released
file is one trained model. The 20-image setting in the results table is supported
by training with `--budget 20`; a 20-image checkpoint is not included.

For training and video inference, also download the SAM2.1 Hiera Large base
checkpoint from the official
[Meta SAM2 repository](https://github.com/facebookresearch/sam2):

```bash
python scripts/download_checkpoints.py --variant full --include-base
```

The fine-tuned checkpoint contains the complete model state and can be used for
image inference directly. Video inference also requests the base checkpoint to
construct the SAM2 video predictor before loading the transferred state.

The released MP-SAM2 models use the bundled `sam2_hiera_l.yaml` architecture.
The training loader transfers matching tensors from SAM2.1 Hiera Large;
SAM2.1-specific spatial and pointer-projection heads are excluded. Keep the
bundled architecture configuration when loading these MP-SAM2 weights.

## Data layout

The data are not distributed in this repository. Prepare paired PNG files as:

```text
dataset/
├── train/
│   ├── images/01/01_01.png
│   ├── images/02/02_01.png
│   ├── masks/01/01_01_enhanced.png
│   └── masks/02/02_01_enhanced.png
└── test/
    ├── images/11/11_01.png
    └── masks/11/11_01_enhanced.png
```

Masks must be binary, with non-zero crack pixels. Do not mix images from the same
specimen or loading sequence across development and test sets.
Training expects square image patches and matching masks. The training command
reads only the categories passed in `--categories`; it does not create a split.
With a custom dataset, replace those category names with your training folders.

## Image inference

Run on one image:

```bash
python scripts/infer_image.py \
  --input path/to/image.png \
  --checkpoint checkpoints/mp_sam2_full.torch \
  --output outputs/image
```

Run recursively over a test directory:

```bash
python scripts/infer_image.py \
  --input dataset/test/images \
  --checkpoint checkpoints/mp_sam2_full.torch \
  --config configs/prompt.yaml \
  --output outputs/images
```

For each image the script preserves the relative category directory and writes
`<name>_mask.png` and `<name>_overlay.png`. `inference.json` contains automatically
selected prompt information. For example, `11/11_01.png` produces
`outputs/images/11/11_01_mask.png`.

`configs/prompt.yaml` is loaded by both inference scripts. Image command-line
prompt options override the corresponding YAML values. Default settings are:

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
  --config configs/prompt.yaml \
  --output outputs/video_overlay.mp4 \
  --contrast 1.0
```

The command writes the overlay video, one binary mask per frame, and a JSON
metadata file. `--contrast 1.0` leaves contrast unchanged; a fixed value greater
than 1 can be used when reproducing a contrast-enhanced video protocol. The same
value is applied to every frame and does not use annotations.

## Training

Few-shot subsets are balanced across the specified training categories.
For example, train a 5-image model with:

```bash
python scripts/train.py \
  --images dataset/train/images \
  --masks dataset/train/masks \
  --categories 01 02 \
  --budget 5 \
  --config configs/fewshot.yaml \
  --epochs 80 \
  --batch-size 2 \
  --base-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --output runs/mp_sam2_5.torch
```

Change `--budget` to `10`, `20`, or `50` for other few-shot budgets.
To train on all valid pairs with the original 3,000/4,000/3,000-stage schedule:

```bash
python scripts/train.py \
  --images dataset/train/images \
  --masks dataset/train/masks \
  --categories 01 02 \
  --budget full \
  --config configs/full.yaml \
  --base-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --full-stage-steps 3000 4000 3000 \
  --output runs/mp_sam2_full.torch
```

The three stages train (1) the prompt encoder and mask decoder, (2) those modules
plus image-encoder blocks 30–31 and the neck, and (3) the full model. The hybrid
loss is `BCE + 2×Dice + Focal + 1.5×IoU`.

Without `--config`, `train.py` selects `configs/full.yaml` for `--budget full`
and `configs/fewshot.yaml` for a numeric budget. YAML settings control the model,
optimizer parameters, stage schedule, selected encoder blocks, loss weights,
and training point sampling. Explicit command-line options override YAML values.
Unknown YAML keys raise an error. The optimizer supported by this trainer is
AdamW. Few-shot schedules use equivalent epochs; full-data schedules use stage
iteration counts.

## Evaluation

```bash
python scripts/evaluate.py \
  --predictions outputs/images \
  --masks dataset/test/masks \
  --output outputs/test_metrics.json
```

Accuracy, precision, recall, F1, IoU, and Dice are first calculated per image and
then averaged, matching the image-level reporting protocol.
The evaluator matches nested category paths and ignores overlay images. It also
writes per-image metrics to a CSV next to the JSON summary. Metric values are
fractions in `[0, 1]`; multiply by 100 to express them as percentages.

## Training settings

- Input image patches are resized to 1024 × 1024 by the SAM2 transform.
- Few-shot experiments use 80 equivalent epochs and stage ratios 30%/40%/30%.
- Optimizer: AdamW, learning rate `1e-5`, weight decay `0.01`, batch size `2`.

## License and acknowledgment

Code and released fine-tuned weights are distributed under Apache License 2.0.
The vendored SAM2 source retains
Meta's copyright notices and license. See [`LICENSE`](LICENSE) and the
third-party notices in [`sam2/licenses/`](sam2/licenses/).

## Citation

Use the repository's [citation metadata](CITATION.cff) to cite this code release.
The associated method is described in the manuscript named at the top of this
README; an article DOI is not specified here until the final record is available.
