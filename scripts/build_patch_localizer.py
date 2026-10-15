"""Fit the fallback patch localizer on renders and report its box recall on held-out renders.

Usage:
    python scripts/build_patch_localizer.py --renders data/renders/v2 --out runs/phase3/localizer_patch --n 1500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.features import Embedder  # noqa: E402
from stepwitness.localizer_patch import PatchLocalizer, fit_patch_classifier  # noqa: E402


def iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "dinov2-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = []
    for m in sorted(args.renders.glob("manifest_seed*.json")):
        recs.extend(json.loads(m.read_text()))
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    recs = recs[: args.n]
    n_val = max(100, len(recs) // 10)
    train, val = recs[n_val:], recs[:n_val]
    emb = Embedder(args.model)

    toks, labs = [], []
    for i in range(0, len(train), 16):
        batch = train[i : i + 16]
        ims = [Image.open(args.renders / "images" / r["image"]) for r in batch]
        t = emb.embed_tokens(ims, image_size=args.image_size, grid=args.grid)[:, 1:]  # (b, g*g, d)
        for r, im, tt in zip(batch, ims, t):
            W, H = r["size"]
            x, y, w, h = r["bbox_xywh"]
            lab = np.zeros((args.grid, args.grid), dtype=np.float32)
            for a in range(args.grid):
                for c in range(args.grid):
                    cx, cy = (c + 0.5) * W / args.grid, (a + 0.5) * H / args.grid
                    lab[a, c] = 1.0 if (x <= cx <= x + w and y <= cy <= y + h) else 0.0
            toks.append(tt.astype(np.float32))
            labs.append(lab.reshape(-1))
        if (i // 16) % 20 == 0:
            print(f"  tokens for {i + len(batch)}/{len(train)}")
    X = np.concatenate(toks, axis=0)
    Y = np.concatenate(labs, axis=0)
    print(f"fitting on {len(X)} patches, positive fraction {Y.mean():.3f}")
    w, b = fit_patch_classifier(X, Y)
    loc = PatchLocalizer(emb, w, b, args.image_size, args.grid)

    ious, found = [], 0
    for r in val:
        im = Image.open(args.renders / "images" / r["image"])
        box, score = loc.locate(im)
        if box is None:
            ious.append(0.0)
            continue
        found += 1
        ious.append(iou(box, r["bbox_xywh"]))
    ious = np.array(ious)
    res = {"n_val": len(val), "found": found, "mean_iou": float(ious.mean()), "recall_iou50": float((ious >= 0.5).mean()), "recall_iou30": float((ious >= 0.3).mean())}
    print(json.dumps(res, indent=1))
    loc.save(args.out)
    (args.out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
