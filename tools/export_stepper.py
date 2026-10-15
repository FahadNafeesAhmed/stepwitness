"""Export frames and events of a monitored recording for the page's frame stepper.

Writes ``site/assets/stepper/<name>/`` with web-sized frames at a fixed stride plus the frames at
every event, and ``index.json`` describing them, then updates ``site/assets/stepper/index.json``.

Usage:
    python tools/export_stepper.py --events runs/phase4/monitor_v2 --video data/industreal/videos/03_assy_1_3.mp4 \
        --label "03_assy_1_3: assembly with two errors" --stride 25 --width 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import Procedure  # noqa: E402
from stepwitness.timeline import frame_evidence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--site", type=Path, default=ROOT / "site")
    args = ap.parse_args()

    name = args.video.stem
    events = json.loads((args.events / f"{name}.events.json").read_text())
    proc = Procedure.builtin("industreal_assembly" if "assy" in name else "industreal_maintenance")
    lines = {s.line: s.text for s in proc.steps.values()}
    out = args.site / "assets" / "stepper" / name
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted = set(range(0, n_frames, args.stride))
    for e in events:
        if e["type"] != "hold":
            wanted.add(e["frame"])
    by_frame = frame_evidence(args.events, name, events)
    frames = []
    last_ev = None
    fno = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fno += 1
        if fno in by_frame:
            last_ev = by_frame[fno] if by_frame[fno].get("box") else None
        if fno not in wanted:
            continue
        scale = args.width / src_w
        small = cv2.resize(img, (args.width, int(src_h * scale)))
        fname = f"f{fno:06d}.jpg"
        cv2.imwrite(str(out / fname), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rec = {"frame": fno, "file": fname}
        if last_ev:
            if last_ev.get("box"):
                rec["box"] = last_ev["box"]
            if last_ev.get("top"):
                rec["top"] = last_ev["top"].replace(",", "")
                rec["top_p"] = last_ev.get("top_p", 0.0)
            if last_ev.get("err_p") is not None and last_ev["err_p"] >= 0.5:
                rec["err"] = [last_ev.get("err_component"), round(float(last_ev["err_p"]), 2)]
        frames.append(rec)
    cap.release()

    ev_out = []
    for e in events:
        if e["type"] == "hold":
            continue
        if e["type"] == "completion":
            text = f"line {e.get('line')}: {e.get('text') or lines.get(e.get('line'), '')}"
        elif e["type"] == "execution_error":
            text = f"execution error on component {e.get('component')} (line {e.get('line')})"
        elif e["type"] == "omission":
            text = ("at end: never completed, " if e.get("at_end") else "skipped: ") + f"line {e.get('line')}"
        else:
            text = e["type"]
        ev_out.append({"type": e["type"], "frame": e["frame"], "text": text})

    (out / "index.json").write_text(json.dumps({"name": name, "size": [src_w, src_h], "n_frames": n_frames, "frames": frames, "events": ev_out}))
    index_path = args.site / "assets" / "stepper" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [i for i in index if i["path"] != f"assets/stepper/{name}"]
    index.append({"path": f"assets/stepper/{name}", "label": args.label or name})
    index_path.write_text(json.dumps(index, indent=1))
    size_mb = sum(p.stat().st_size for p in out.glob("*.jpg")) / 1e6
    print(f"exported {len(frames)} frames ({size_mb:.1f} MB) and {len(ev_out)} events for {name}")


if __name__ == "__main__":
    main()
