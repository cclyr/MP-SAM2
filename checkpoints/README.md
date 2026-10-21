# Checkpoints

Download fine-tuned weights from the
[Hugging Face model repository](https://huggingface.co/ccllr/MP-SAM2)
and place them in this directory. Each file is approximately 898 MB:

| Checkpoint | Training images |
|---|---:|
| [mp_sam2_5.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_5.torch) | 5 |
| [mp_sam2_10.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_10.torch) | 10 |
| [mp_sam2_50.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_50.torch) | 50 |
| [mp_sam2_full.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_full.torch) | 186 (full data) |

Each file contains a complete model state for the bundled `sam2_hiera_l.yaml`
architecture. Each is a single trained model; paper averages cover repeated
training runs. A 20-image checkpoint is not part of this release.
All four checkpoints contain the complete model state and are released under
the repository's Apache License 2.0.

From the repository root, use the download helper:

```bash
python scripts/download_checkpoints.py --variant full
```

Use `--variant 5`, `10`, `50`, or `all` to select other models. The helper checks
downloaded MP-SAM2 files against [`SHA256SUMS`](SHA256SUMS).

Training and video inference also require `sam2.1_hiera_large.pt`, downloaded
directly from Meta under its upstream SAM2 license:

```bash
python scripts/download_checkpoints.py --variant full --include-base
```

The transfer loader selects matching SAM2.1 tensors for the bundled MP-SAM2
architecture. Image inference needs only a fine-tuned MP-SAM2 file.

Checkpoint binaries are hosted on Hugging Face and are intentionally excluded
from the GitHub source repository's Git history.
