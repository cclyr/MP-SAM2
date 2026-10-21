# Dataset layout

The ECC image and video datasets are not included in this repository.

Training expects the following layout:

```text
dataset/
  train/
    images/01/01_01.png
    images/02/02_01.png
    masks/01/01_01_enhanced.png
    masks/02/02_01_enhanced.png
  test/
    images/11/11_01.png
    masks/11/11_01_enhanced.png
```

Training images are square RGB PNG patches. Masks are binary PNG files in which
crack pixels are non-zero and background is zero. Images and masks must align
spatially. Categories may represent specimen batches or acquisition sequences.
Keep specimen-level independence when constructing train/test splits.

Pass `--images dataset/train/images --masks dataset/train/masks` and your
training category names, for example `--categories 01 02`, to `scripts/train.py`.
It reads the specified categories and does not create or change the data split.

For image inference, `--input dataset/test/images` recursively preserves category
folders in the output. Evaluate those predictions with
`--masks dataset/test/masks`; the default ground-truth suffix is `_enhanced`.

