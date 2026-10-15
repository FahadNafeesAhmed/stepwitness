"""Phase 3, version 3: patch-token features, a state head and a component-error head.

Usage:
    python scripts/build_verifier_v3.py --renders data/renders/v2 --out runs/phase3/verifier_v3 \
        --image-size 448 --grid 8 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.features import Embedder, crop_with_margin  # noqa: E402
from stepwitness.verifier_v3 import MultiTaskVerifier, evaluate_v3, train_v3  # noqa: E402


def load_manifests(renders: Path) -> list[dict]:
    recs = []
    for m in sorted(renders.glob("manifest_seed*.json")):
        for r in json.loads(m.read_text()):
            r["_dir"] = renders
            recs.append(r)
    return recs


def jitter_box(rng, box, amount):
    x, y, w, h = box
    dx, dy = rng.uniform(-amount, amount) * w, rng.uniform(-amount, amount) * h
    s = 1.0 + rng.uniform(-amount, amount)
    return [x + dx + w * (1 - s) / 2, y + dy + h * (1 - s) / 2, w * s, h * s]


def extract(recs, emb, image_size, grid, jitter, seed, cache: Path) -> np.ndarray:
    """Tokens for every render, written straight into a memory-mapped .npy so the process never holds
    the whole set in RAM (this machine has little to spare). Resumes from a partial file."""
    cache = cache.with_suffix(".npy")
    done_marker = cache.with_suffix(".done")
    dim = emb.model.config.hidden_size
    shape = (len(recs), 1 + grid * grid, dim)
    if cache.exists() and done_marker.exists():
        x = np.load(cache, mmap_mode="r")
        if x.shape == shape:
            return x
    cache.parent.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16, shape=shape)
    rng = random.Random(seed)
    t0 = time.time()
    imgs, idx = [], []
    for i, r in enumerate(recs):
        im = Image.open(r["_dir"] / "images" / r["image"])
        box = jitter_box(rng, r["bbox_xywh"], jitter) if jitter > 0 else r["bbox_xywh"]
        imgs.append(crop_with_margin(im, box))
        idx.append(i)
        im.close()
        if len(imgs) == 16 or i == len(recs) - 1:
            x[idx[0] : idx[-1] + 1] = emb.embed_tokens(imgs, image_size=image_size, grid=grid)
            imgs, idx = [], []
            if (i + 1) % 1024 == 0:
                x.flush()
                print(f"  tokens {i + 1}/{len(recs)} ({(i + 1) / (time.time() - t0):.1f} img/s)", flush=True)
    x.flush()
    done_marker.write_text("ok")
    return np.load(cache, mmap_mode="r")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "dinov2-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = load_manifests(args.renders)
    if args.limit:
        recs = recs[: args.limit]
    codes = [r["label_state"] for r in recs]
    print(f"{len(recs)} renders, {len(Counter(codes))} label codes")
    rng = random.Random(args.seed)
    val_mask = np.array([rng.random() < args.val_fraction for _ in recs])
    emb = Embedder(args.model)
    x = extract(recs, emb, args.image_size, args.grid, args.jitter, args.seed, args.out / f"tokens_{args.image_size}_{args.grid}.npz")
    del emb  # free the backbone before training; the head only needs the tokens
    import torch

    torch.cuda.empty_cache()
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    x_va = np.ascontiguousarray(x[va_idx])  # small enough to hold
    c_tr = [codes[i] for i in tr_idx]
    c_va = [codes[i] for i in va_idx]
    print(f"train {len(c_tr)}, val {len(c_va)}, token shape {x.shape[1:]}", flush=True)
    head, info = train_v3(x[tr_idx], c_tr, epochs=args.epochs, seed=args.seed, val=(x_va, c_va))
    final = evaluate_v3(head, x_va, c_va)
    print("final:", json.dumps(final, indent=1))
    v = MultiTaskVerifier(head, args.image_size, args.grid)
    v.save(args.out)
    (args.out / "results.json").write_text(json.dumps({"n_train": len(c_tr), "n_val": len(c_va), "final": final, "history": info["history"], "image_size": args.image_size, "grid": args.grid}, indent=1))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
