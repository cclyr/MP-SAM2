# Dataset placeholder

The ECC image and video datasets are not included in this repository.

Training expects the following layout:

```text
images/
  01/01_01.png
  02/02_01.png
masks/
  01/01_01_enhanced.png
  02/02_01_enhanced.png
```

Images are RGB PNG files. Masks are binary PNG files in which crack pixels are
non-zero. Categories may represent specimen batches or acquisition sequences.
Keep specimen-level independence when constructing train/test splits.

