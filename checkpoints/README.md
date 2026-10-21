# Checkpoints

Download fine-tuned weights from the
[Hugging Face model repository](https://huggingface.co/ccllr/MP-SAM2)
and place them in this directory. Each file is approximately 898 MB:

| Checkpoint | Training images | Seed |
|---|---:|---|
| [mp_sam2_5_seed42.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_5_seed42.torch) | 5 | 42 |
| [mp_sam2_10_seed42.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_10_seed42.torch) | 10 | 42 |
| [mp_sam2_50_seed42.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_50_seed42.torch) | 50 | 42 |
| [mp_sam2_full.torch](https://huggingface.co/ccllr/MP-SAM2/resolve/main/mp_sam2_full.torch) | 186 | original run |

The few-shot files are representative seed-42 checkpoints. Paper averages use
three training seeds; a single checkpoint does not represent that average.
All four checkpoints contain the complete model state and are released under
the repository's Apache License 2.0.

[`SHA256SUMS`](SHA256SUMS) lists file checksums. For example, on Linux, run from
this directory after downloading all four files:

```bash
sha256sum -c SHA256SUMS
```

Training and video inference also require `sam2.1_hiera_large.pt`. Obtain it
from [Meta's official download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
under the upstream SAM2 license. Image inference needs only a fine-tuned file.

Checkpoint binaries are hosted on Hugging Face and are intentionally excluded
from the GitHub source repository's Git history.
