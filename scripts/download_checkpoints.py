"""Download the public MP-SAM2 checkpoints without requiring a Hub login."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
HF_URL = "https://huggingface.co/ccllr/MP-SAM2/resolve/main"
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_hash: str | None = None) -> None:
    if destination.is_file() and expected_hash and sha256(destination) == expected_hash:
        print(f"Already downloaded: {destination.name}")
        return
    temporary = destination.with_name(destination.name + ".partial")
    print(f"Downloading {destination.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if expected_hash and sha256(temporary) != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {destination.name}; incomplete file: {temporary}")
    temporary.replace(destination)
    print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["full", "5", "10", "50", "all"], default="full")
    parser.add_argument("--include-base", action="store_true", help="Also download the official SAM2.1 Hiera Large checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest for digest, name in (
        line.split() for line in (ROOT / "checkpoints" / "SHA256SUMS").read_text().splitlines()
    )}
    variants = ["full", "5", "10", "50"] if args.variant == "all" else [args.variant]
    for variant in variants:
        filename = f"mp_sam2_{variant}.torch"
        download(f"{HF_URL}/{filename}", args.output_dir / filename, hashes[filename])
    if args.include_base:
        download(BASE_URL, args.output_dir / "sam2.1_hiera_large.pt")


if __name__ == "__main__":
    main()
