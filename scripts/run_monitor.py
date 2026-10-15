"""Run the monitor on recordings: video -> localizer -> crop -> verifier -> recognizer -> events.

For each recording it writes ``<name>.events.json`` (every event with frame, step, line, confidence
and evidence) and ``<name>.yhat.json`` (completions in the IndustReal format for the metric code).

Usage:
    python scripts/run_monitor.py --videos data/industreal/videos --recordings data/industreal/recordings/test \
        --verifier runs/phase3/verifier_v2 --localizer runs/phase3/localizer_v2/train/weights/best.pt \
        --out runs/phase4/monitor_v2 [--limit 3] [--stride 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.features import Embedder, crop_with_margin  # noqa: E402
from stepwitness.procedure import Procedure  # noqa: E402
from stepwitness.recognizer import Recognizer, RecognizerConfig  # noqa: E402
from stepwitness.verifier import Verifier  # noqa: E402


def procedure_for(name: str) -> Procedure:
    return Procedure.builtin("industreal_assembly" if "assy" in name else "industreal_maintenance")


def run_recording(video: Path, verifier, embedder: Embedder, localizer, config: RecognizerConfig, stride: int, conf: float, out: Path, kind: str = "v2") -> dict:
    name = video.stem
    proc = procedure_for(name)
    rec = Recognizer(proc, verifier.classes, config)
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = time.time()
    frame_no = -1
    n_detected = 0
    last_box = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % stride:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = None
        if hasattr(localizer, "locate"):  # the fallback patch localizer
            box, score = localizer.locate(Image.fromarray(rgb))
            if box is not None and score < conf:
                box = None
        else:
            res = localizer.predict(rgb, verbose=False, conf=conf, device=0, imgsz=640)[0]
            if len(res.boxes):
                b = res.boxes
                i = int(b.conf.argmax())
                x0, y0, x1, y1 = b.xyxy[i].tolist()
                box = [x0, y0, x1 - x0, y1 - y0]
        if box is not None:
            last_box = box
            n_detected += 1
        if box is None:
            # no assembly found: nothing to accumulate; let the recognizer decay and count a low-confidence frame
            posterior = np.full(len(verifier.classes), 1.0 / len(verifier.classes))
            rec.update(posterior, frame_no, evidence={"box": None})
            continue
        crop = crop_with_margin(Image.fromarray(rgb), box)
        if kind == "v3":
            toks = embedder.embed_tokens([crop], image_size=verifier.image_size, grid=verifier.grid)
            state_probs, err_probs = verifier.posterior_parts(toks)
            state_probs, err_probs = state_probs[0], err_probs[0]
            top = int(state_probs.argmax())
            worst = int(err_probs.argmax())
            evidence = {"box": [round(v, 1) for v in box], "top": verifier.classes[top], "top_p": round(float(state_probs[top]), 3),
                        "err_component": worst, "err_p": round(float(err_probs[worst]), 3)}
            rec.update_parts(state_probs, verifier.classes, err_probs, frame_no, evidence=evidence)
        else:
            emb = embedder.embed([crop])[0]
            posterior = verifier.posterior(emb[None])[0]
            top = int(posterior.argmax())
            evidence = {"box": [round(v, 1) for v in box], "top": verifier.classes[top], "top_p": round(float(posterior[top]), 3)}
            rec.update(posterior, frame_no, evidence=evidence)
    rec.finalize(frame_no)
    cap.release()
    dt = time.time() - t0
    events = rec.events
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.events.json").write_text(json.dumps(events, indent=1))
    proc_info = json.loads((ROOT / "third_party" / "IndustReal" / "PSR" / "procedure_info.json").read_text())
    yhat = rec.y_hat(proc_info)
    (out / f"{name}.yhat.json").write_text(json.dumps(yhat, indent=1))
    summary = {
        "recording": name,
        "frames": n_frames,
        "processed": frame_no + 1,
        "detected_frames": n_detected,
        "seconds": round(dt, 1),
        "fps": round((frame_no + 1) / max(dt, 1e-6), 1),
        "completions": [e["step"] for e in events if e["type"] == "completion"],
        "flags": [(e["type"], e.get("component")) for e in events if e["type"] in ("omission", "execution_error") and not e.get("at_end")],
    }
    print(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "data" / "industreal" / "videos")
    ap.add_argument("--recordings", type=Path, default=ROOT / "data" / "industreal" / "recordings" / "test")
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--localizer", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "dinov2-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--decay", type=float, default=0.75)
    ap.add_argument("--min-posterior", type=float, default=0.30)
    ap.add_argument("--error-frames", type=int, default=6)
    ap.add_argument("--error-margin", type=float, default=0.10)
    ap.add_argument("--verifier-kind", choices=["v2", "v3"], default="v2")
    args = ap.parse_args()

    from ultralytics import YOLO

    names = sorted(p.name for p in args.recordings.iterdir() if p.is_dir())
    if args.limit:
        names = names[: args.limit]
    if args.verifier_kind == "v3":
        from stepwitness.verifier_v3 import MultiTaskVerifier

        verifier = MultiTaskVerifier.load(args.verifier)
    else:
            verifier = Verifier.load(args.verifier)
    embedder = Embedder(args.model)
    if args.localizer.is_dir() and (args.localizer / "patch_localizer.npz").exists():
        from stepwitness.localizer_patch import PatchLocalizer

        localizer = PatchLocalizer.load(args.localizer, embedder)
    else:
        localizer = YOLO(str(args.localizer))
    config = RecognizerConfig(threshold=args.threshold, decay=args.decay, min_posterior=args.min_posterior, error_frames=args.error_frames, error_margin=args.error_margin)
    summaries = []
    for name in names:
        video = args.videos / f"{name}.mp4"
        if not video.exists():
            print(f"missing video for {name}")
            continue
        summaries.append(run_recording(video, verifier, embedder, localizer, config, args.stride, args.conf, args.out, kind=args.verifier_kind))
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
