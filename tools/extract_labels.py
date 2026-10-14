"""Pull the small label files out of the IndustReal recording zips without extracting the frames.

The PSR baseline needs, per test recording, PSR_labels.csv and PSR_labels_raw.csv, plus the number
of RGB frames, which it normally gets by globbing the extracted jpgs. The frame count is read here
from the zip member list instead, so 24 GB of jpgs never touch the disk.

Usage:
    python tools/extract_labels.py data/industreal/zips/test_p1.zip [more zips...]
        --out data/industreal/recordings --counts data/industreal/frame_counts.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

KEEP = {
    "PSR_labels.csv",
    "PSR_labels_raw.csv",
    "PSR_labels_with_errors.csv",
    "OD_labels.json",
    "AR_labels.csv",
    "gaze.csv",
    "hands.csv",
    "pose.csv",
}
SPLITS = ("train", "val", "test")


def recording_of(parts: tuple[str, ...], default_split: str | None) -> tuple[str, str] | None:
    """Return (split, recording) for a member path.

    The published archives are laid out as ``<recording>/<modality>/<frame>.jpg`` with the split
    only in the archive's name (``test_p1.zip``), so the split falls back to ``default_split``.
    A ``.../test/<recording>/...`` layout is also accepted.
    """
    for i, p in enumerate(parts):
        if p in SPLITS and i + 1 < len(parts):
            return p, parts[i + 1]
    if default_split and len(parts) >= 2:
        return default_split, parts[0]
    return None


def split_from_name(zpath: Path) -> str | None:
    stem = zpath.stem.lower()
    for s in SPLITS:
        if stem.startswith(s):
            return s
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/industreal/recordings"))
    ap.add_argument("--counts", type=Path, default=Path("data/industreal/frame_counts.json"))
    args = ap.parse_args()

    counts: dict[str, int] = {}
    if args.counts.exists():
        counts = json.loads(args.counts.read_text())

    for zpath in args.zips:
        written: dict[str, int] = defaultdict(int)
        frames: dict[str, int] = defaultdict(int)
        default_split = split_from_name(zpath)
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                hit = recording_of(parts, default_split)
                if hit is None:
                    continue
                split, rec = hit
                name = parts[-1]
                lowered = [p.lower() for p in parts]
                if name.lower().endswith(".jpg") and "rgb" in lowered:
                    frames[f"{split}/{rec}"] += 1
                elif name in KEEP:
                    target = args.out / split / rec / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(info))
                    written[f"{split}/{rec}"] += 1
        for key in sorted(set(written) | set(frames)):
            counts[key] = frames.get(key, 0)
            print(f"{zpath.name}\t{key}\tlabel files: {written.get(key, 0)}\tframes: {frames.get(key, 0)}")

    args.counts.parent.mkdir(parents=True, exist_ok=True)
    args.counts.write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(f"wrote {args.counts} with {len(counts)} recordings")


if __name__ == "__main__":
    main()
