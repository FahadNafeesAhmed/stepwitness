"""Phase 6: measure what the deployed models cost on this machine and write the profile file.

Measures the localizer and the verifier backbone at FP32 and FP16 on the GPU (and INT8 through
TensorRT when the ``tensorrt`` package is installed), replaying real frames from a recording so the
numbers reflect the actual input, and samples GPU power from the driver while timing. Writes
``profile.json`` with hardware, runtime, latency percentiles, throughput, peak memory, power, and
the verifier's agreement with FP32 at each precision.

Usage:
    python scripts/profile_edge.py --video data/industreal/videos/03_assy_0_1.mp4 \
        --verifier runs/phase3/verifier_v2 --localizer runs/phase3/localizer_v2/train/weights/best.pt \
        --out runs/phase6/profile_v2 --frames 300
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.features import Embedder, crop_with_margin  # noqa: E402
from stepwitness.verifier import Verifier  # noqa: E402


class PowerSampler:
    """Samples GPU power draw from nvidia-smi at 10 Hz on a background thread."""

    def __init__(self):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.samples)) if self.samples else None


def hardware() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "vram_mb": int(p.total_memory / 2**20)})
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            info["driver"] = out.stdout.strip()
        except Exception:
            pass
    return info


def load_frames(video: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), num=min(n, total)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def percentiles(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


def time_localizer(localizer, frames, half: bool) -> tuple[dict, list]:
    boxes = []
    ms = []
    for f in frames[:10]:
        localizer.predict(f, verbose=False, device=0, imgsz=640, half=half)
    torch.cuda.synchronize()
    for f in frames:
        t0 = time.perf_counter()
        r = localizer.predict(f, verbose=False, device=0, imgsz=640, half=half)[0]
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        b = None
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            x0, y0, x1, y1 = r.boxes.xyxy[i].tolist()
            b = [x0, y0, x1 - x0, y1 - y0]
        boxes.append(b)
    return percentiles(ms), boxes


def time_verifier(embedder: Embedder, verifier: Verifier, crops, half: bool) -> tuple[dict, list[int]]:
    model = embedder.model
    if half:
        model.half()
    preds, ms = [], []
    for c in crops[:10]:
        embedder.embed([c])
    torch.cuda.synchronize()
    for c in crops:
        t0 = time.perf_counter()
        e = embedder.embed([c])
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        preds.append(int(verifier.posterior(e)[0].argmax()))
    if half:
        model.float()
    return percentiles(ms), preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--localizer", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "models" / "dinov2-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--version", default="0.1.0")
    args = ap.parse_args()

    from ultralytics import YOLO

    frames = load_frames(args.video, args.frames)
    localizer = YOLO(str(args.localizer))
    verifier = Verifier.load(args.verifier)
    embedder = Embedder(args.model)
    profile = {
        "model": "stepwitness",
        "version": args.version,
        "previous_version": None,
        "hardware": hardware(),
        "input": {"video": args.video.name, "frames": len(frames), "resolution": list(frames[0].shape[:2][::-1]) if frames else None},
        "runs": {},
    }

    torch.cuda.reset_peak_memory_stats()
    with PowerSampler() as ps:
        loc32, boxes = time_localizer(localizer, frames, half=False)
    crops = [crop_with_margin(Image.fromarray(f), b) for f, b in zip(frames, boxes) if b is not None]
    with PowerSampler() as pv:
        ver32, pred32 = time_verifier(embedder, verifier, crops, half=False)
    profile["runs"]["fp32"] = {
        "localizer_ms": loc32,
        "verifier_ms": ver32,
        "pipeline_ms_p50": loc32["p50"] + ver32["p50"],
        "pipeline_fps": 1000.0 / (loc32["p50"] + ver32["p50"]),
        "peak_memory_mb": int(torch.cuda.max_memory_allocated() / 2**20),
        "power_w_avg": {"localizer": ps.mean, "verifier": pv.mean},
        "detected_fraction": sum(b is not None for b in boxes) / max(len(boxes), 1),
    }

    torch.cuda.reset_peak_memory_stats()
    with PowerSampler() as ps:
        loc16, _ = time_localizer(localizer, frames, half=True)
    with PowerSampler() as pv:
        ver16, pred16 = time_verifier(embedder, verifier, crops, half=True)
    agree = float(np.mean(np.array(pred16) == np.array(pred32))) if pred32 else None
    profile["runs"]["fp16"] = {
        "localizer_ms": loc16,
        "verifier_ms": ver16,
        "pipeline_ms_p50": loc16["p50"] + ver16["p50"],
        "pipeline_fps": 1000.0 / (loc16["p50"] + ver16["p50"]),
        "peak_memory_mb": int(torch.cuda.max_memory_allocated() / 2**20),
        "power_w_avg": {"localizer": ps.mean, "verifier": pv.mean},
        "agreement_with_fp32": agree,
    }

    try:
        import tensorrt  # noqa: F401

        profile["runs"]["int8"] = {"status": "tensorrt available, INT8 build not yet implemented"}
    except ImportError:
        profile["runs"]["int8"] = {"status": "tensorrt package not installed on this machine"}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "profile.json").write_text(json.dumps(profile, indent=1))
    print(json.dumps({k: v for k, v in profile["runs"].items()}, indent=1))
    print(f"wrote {args.out / 'profile.json'}")


if __name__ == "__main__":
    main()
