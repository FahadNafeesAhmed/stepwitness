"""Phase 3: train the one-class assembly localizer on renders.

Converts the render manifests into YOLO format (one class, "assembly") and fine-tunes a small
YOLOv8 model. The localizer only has to find the assembly; the verifier does the fine work on the crop.

Usage:
    python scripts/train_localizer.py --renders data/renders/v2 --out runs/phase3/localizer_v2 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_yolo(renders: Path, out: Path, val_fraction: float, seed: int, limit: int) -> Path:
    recs = []
    for m in sorted(renders.glob("manifest_seed*.json")):
        recs.extend(json.loads(m.read_text()))
    if limit:
        recs = recs[:limit]
    rng = random.Random(seed)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in recs:
        split = "val" if rng.random() < val_fraction else "train"
        src = renders / "images" / r["image"]
        dst = out / "images" / split / r["image"]
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        w, h = r["size"]
        x, y, bw, bh = r["bbox_xywh"]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        (out / "labels" / split / (Path(r["image"]).stem + ".txt")).write_text(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}\n")
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: assembly\n")
    print(f"wrote YOLO dataset with {len(recs)} images to {out}")
    return yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    yaml = to_yolo(args.renders, args.out / "dataset", args.val_fraction, args.seed, args.limit)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=str(args.out), name="train", exist_ok=True, seed=args.seed, verbose=False)
    print(f"trained localizer; weights under {args.out / 'train' / 'weights'}")


if __name__ == "__main__":
    main()
