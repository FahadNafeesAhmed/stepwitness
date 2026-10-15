"""Re-run the recognizer offline from cached verifier posteriors and score the result.

The monitor saves ``<name>.posteriors.npz`` per recording (the verifier's 22-way state posterior and
11-way component-error probabilities on every processed frame). Replaying those through the
recognizer takes seconds, so recognizer settings can be compared without another GPU pass.

Usage:
    python scripts/rescore.py --posteriors runs/phase4/monitor_v3 --out runs/phase4/rescore_a \
        [--threshold 8 --decay 0.75 --min-posterior 0.3 --error-frames 6 --error-margin 0.1 --reach 0 --stable-frames 0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import Procedure  # noqa: E402
from stepwitness.recognizer import Recognizer, RecognizerConfig  # noqa: E402


def procedure_for(name: str) -> Procedure:
    return Procedure.builtin("industreal_assembly" if "assy" in name else "industreal_maintenance")


def replay(npz: Path, config: RecognizerConfig, out: Path, proc_info: dict) -> dict:
    name = npz.name.replace(".posteriors.npz", "")
    d = np.load(npz, allow_pickle=False)
    frames, state, err, classes = d["frames"], d["state"], d["err"], [str(c) for c in d["classes"]]
    boxes = d["boxes"] if "boxes" in d else None
    rec = Recognizer(procedure_for(name), classes, config)
    uniform = np.full(len(classes), 1.0 / len(classes))
    for i, f in enumerate(frames):
        f = int(f)
        if np.isnan(state[i, 0]):
            rec.update(uniform, f, evidence={"box": None})
            continue
        top = int(state[i].argmax())
        worst = int(err[i].argmax())
        box = None if boxes is None or np.isnan(boxes[i, 0]) else [round(float(v), 1) for v in boxes[i]]
        evidence = {"box": box, "top": classes[top], "top_p": round(float(state[i, top]), 3), "err_component": worst, "err_p": round(float(err[i, worst]), 3)}
        rec.update_parts(state[i].astype(np.float64), classes, err[i].astype(np.float64), f, evidence=evidence)
    rec.finalize(int(frames[-1]))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.events.json").write_text(json.dumps(rec.events, indent=1))
    yhat = rec.y_hat(proc_info)
    (out / f"{name}.yhat.json").write_text(json.dumps(yhat, indent=1))
    trace = npz.with_name(f"{name}.trace.json")  # per-frame boxes and readings, so clips and the stepper can be made from this replay
    if trace.exists():
        (out / trace.name).write_bytes(trace.read_bytes())
    return {"recording": name, "completions": len(yhat), "flags": sum(1 for e in rec.events if e["type"] in ("omission", "execution_error") and not e.get("at_end"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posteriors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="rescore")
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--decay", type=float, default=0.75)
    ap.add_argument("--min-posterior", type=float, default=0.30)
    ap.add_argument("--error-frames", type=int, default=6)
    ap.add_argument("--error-margin", type=float, default=0.10)
    ap.add_argument("--reach", type=int, default=0, help="keep posterior mass only on states within this many transitions of the current state (0 = off)")
    ap.add_argument("--stable-frames", type=int, default=0, help="the top state must hold this many consecutive frames before it earns credit (0 = off)")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    config = RecognizerConfig(threshold=args.threshold, decay=args.decay, min_posterior=args.min_posterior, error_frames=args.error_frames,
                              error_margin=args.error_margin, reach=args.reach, stable_frames=args.stable_frames)
    proc_info = json.loads((ROOT / "third_party" / "IndustReal" / "PSR" / "procedure_info.json").read_text())
    files = sorted(args.posteriors.glob("*.posteriors.npz"))
    if not files:
        sys.exit(f"no posterior files under {args.posteriors}")
    summaries = [replay(f, config, args.out, proc_info) for f in files]
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=1))
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    print(f"replayed {len(files)} recordings into {args.out}")
    if args.no_score:
        return
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts" / "evaluate_psr.py"), "--predictions", str(args.out), "--out", str(args.out), "--label", args.label], check=False)
    subprocess.run([py, str(ROOT / "scripts" / "evaluate_flags.py"), "--events", str(args.out), "--out", str(args.out)], check=False)


if __name__ == "__main__":
    main()
