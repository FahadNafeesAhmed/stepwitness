"""Phase 3: build the state verifier from rendered crops.

Reads render manifests, crops each assembly box with a margin and a little box jitter (so the verifier
tolerates a sloppy localizer), embeds the crops with the frozen backbone, splits by sample index into
train and validation, builds the prototype bank, trains the head, and reports accuracy on validation:
overall, on correct states only, on error states only, and the confusion between "correct" and
"some error" for each correct state.

Usage:
    python scripts/build_verifier.py --renders data/renders/v2 --model data/models/dinov2-base \
        --out runs/phase3/verifier_v2 --val-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.features import Embedder, crop_with_margin  # noqa: E402
from stepwitness.verifier import PrototypeBank, Verifier, correct_of, error_component, head_logits, train_head  # noqa: E402


def load_manifests(renders: Path) -> list[dict]:
    recs = []
    for m in sorted(renders.glob("manifest_seed*.json")):
        for r in json.loads(m.read_text()):
            r["_dir"] = renders
            recs.append(r)
    return recs


def jitter_box(rng: random.Random, box: list[float], amount: float) -> list[float]:
    x, y, w, h = box
    dx, dy = rng.uniform(-amount, amount) * w, rng.uniform(-amount, amount) * h
    s = 1.0 + rng.uniform(-amount, amount)
    return [x + dx + w * (1 - s) / 2, y + dy + h * (1 - s) / 2, w * s, h * s]


def embed_records(recs: list[dict], emb: Embedder, jitter: float, seed: int, cache: Path | None) -> np.ndarray:
    if cache is not None and cache.exists():
        z = np.load(cache)
        if len(z["x"]) == len(recs):
            return z["x"]
    rng = random.Random(seed)
    vectors = []
    t0 = time.time()
    batch_imgs = []
    for i, r in enumerate(recs):
        im = Image.open(r["_dir"] / "images" / r["image"])
        box = jitter_box(rng, r["bbox_xywh"], jitter) if jitter > 0 else r["bbox_xywh"]
        batch_imgs.append(crop_with_margin(im, box))
        if len(batch_imgs) == 128 or i == len(recs) - 1:
            vectors.append(emb.embed(batch_imgs))
            batch_imgs = []
            if (i + 1) % 1280 == 0:
                print(f"  embedded {i + 1}/{len(recs)} ({(i + 1) / (time.time() - t0):.1f} img/s)")
    x = np.concatenate(vectors, axis=0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x)
    return x


def report(name: str, pred: list[str], labels: list[str]) -> dict:
    pred, labels = np.array(pred), np.array(labels)
    out = {"n": len(labels), "accuracy": float((pred == labels).mean())}
    is_err = np.array([error_component(l) is not None for l in labels])
    if is_err.any():
        out["accuracy_on_error_states"] = float((pred[is_err] == labels[is_err]).mean())
    if (~is_err).any():
        out["accuracy_on_correct_states"] = float((pred[~is_err] == labels[~is_err]).mean())
    # binary: did we call "some error" when there was one, and "correct" when it was correct
    pred_err = np.array([error_component(p) is not None for p in pred])
    tp = int((pred_err & is_err).sum())
    fp = int((pred_err & ~is_err).sum())
    fn = int((~pred_err & is_err).sum())
    tn = int((~pred_err & ~is_err).sum())
    out["error_detection"] = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
    }
    # same correct state, wrong error/correct decision
    same_state = np.array([correct_of(p) == correct_of(l) for p, l in zip(pred, labels)])
    out["correct_state_identified"] = float(same_state.mean())
    print(f"[{name}] n={out['n']} acc={out['accuracy']:.3f} state-id={out['correct_state_identified']:.3f} "
          f"err-acc={out.get('accuracy_on_error_states', float('nan')):.3f} "
          f"err-det P={out['error_detection']['precision']:.3f} R={out['error_detection']['recall']:.3f} FPR={out['error_detection']['false_positive_rate']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "dinov2-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--jitter", type=float, default=0.08)
    ap.add_argument("--per-class-prototypes", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N records (smoke test)")
    args = ap.parse_args()

    recs = load_manifests(args.renders)
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} records from {args.renders}")
    labels_all = [r["label_state"] for r in recs]
    counts = Counter(labels_all)
    print(f"{len(counts)} classes; smallest {min(counts.values())}, largest {max(counts.values())}")

    # split by sample index so views of the same state at the same seed never leak across the split
    rng = random.Random(args.seed)
    val_mask = np.array([rng.random() < args.val_fraction for _ in recs])
    emb = Embedder(args.model)
    cache = args.out / "embeddings_cache.npz"
    x = embed_records(recs, emb, args.jitter, args.seed, cache)
    x_tr, x_va = x[~val_mask], x[val_mask]
    y_tr = [l for l, m in zip(labels_all, val_mask) if not m]
    y_va = [l for l, m in zip(labels_all, val_mask) if m]
    classes = sorted(set(y_tr))
    print(f"train {len(y_tr)}, val {len(y_va)}, classes {len(classes)}")

    bank = PrototypeBank.build(x_tr, y_tr, per_class=args.per_class_prototypes, seed=args.seed)
    proto_only = Verifier(bank, None)
    results = {"n_train": len(y_tr), "n_val": len(y_va), "classes": classes}
    results["prototypes_only"] = report("prototypes only", proto_only.predict(x_va), y_va)

    head, info = train_head(x_tr, y_tr, classes, epochs=args.epochs, seed=args.seed, val=(x_va, y_va))
    results["head_history"] = info["history"]
    results["head_only"] = report("head only", [classes[i] for i in head_logits(head, x_va).argmax(1)], y_va)
    verifier = Verifier(bank, head)
    results["blended"] = report("blended", verifier.predict(x_va), y_va)

    # per correct state: how often an error at that state is caught (recall) and a correct crop is falsely flagged
    per_state = defaultdict(lambda: {"n_correct": 0, "n_error": 0, "false_alarm": 0, "missed": 0})
    for p, l in zip(verifier.predict(x_va), y_va):
        st = correct_of(l)
        if error_component(l) is None:
            per_state[st]["n_correct"] += 1
            if error_component(p) is not None:
                per_state[st]["false_alarm"] += 1
        else:
            per_state[st]["n_error"] += 1
            if error_component(p) is None or correct_of(p) != st:
                per_state[st]["missed"] += 1
    results["per_state"] = dict(per_state)

    args.out.mkdir(parents=True, exist_ok=True)
    verifier.save(args.out)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"saved verifier and results to {args.out}")


if __name__ == "__main__":
    main()
