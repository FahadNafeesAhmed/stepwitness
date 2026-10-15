"""Make the clips for the project page.

Two kinds:

* ``monitor``: a real recording with the monitor's evidence drawn on every frame: the localizer box,
  the verifier's top state and confidence, a timeline bar along the bottom that fills with completions
  (green ticks) and flags (red ticks), and a text panel that shows each event when it happens.
  Reads ``<name>.events.json`` from scripts/run_monitor.py.
* ``renders``: a montage of the training renders: a sequence of randomized views, each shown with its
  state code and, for error renders, the error component and kind, so a viewer sees what the model
  learned from and that no real frame is among them.

Both are encoded with ffmpeg to H.264 mp4 that browsers play, sized for the web.

Usage:
    python scripts/make_videos.py monitor --events runs/phase4/monitor_v2 --video data/industreal/videos/03_assy_1_3.mp4 \
        --out site/assets/videos/03_assy_1_3.mp4 [--start 0 --end 1200 --speed 2]
    python scripts/make_videos.py renders --renders data/renders/v2 --out site/assets/videos/renders.mp4 --n 120
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import Procedure  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (42, 42, 42)
PAPER = (234, 239, 245)  # BGR of the page's cream
GREEN = (87, 139, 46)
RED = (43, 57, 192)
BLUE = (209, 111, 59)
WHITE = (255, 255, 255)


def encode(frames_dir: Path, out: Path, fps: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def put_text(img, text, org, scale=0.6, color=INK, thick=1, bg=None):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + base + 4), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_panel(frame, events_so_far: list[dict], lines: dict, state_text: str, frame_no: int, n_frames: int, all_events: list[dict]):
    h, w = frame.shape[:2]
    # bottom timeline bar
    bar_y0, bar_y1 = h - 54, h - 34
    cv2.rectangle(frame, (0, h - 120), (w, h), PAPER, -1)
    cv2.rectangle(frame, (24, bar_y0), (w - 24, bar_y1), WHITE, -1)
    cv2.rectangle(frame, (24, bar_y0), (w - 24, bar_y1), INK, 1)
    span = w - 48
    for e in all_events:
        if e["type"] not in ("completion", "omission", "execution_error"):
            continue
        if e.get("at_end"):
            continue
        x = 24 + int(span * e["frame"] / max(n_frames, 1))
        col = GREEN if e["type"] == "completion" else RED
        alpha_done = e["frame"] <= frame_no
        cv2.line(frame, (x, bar_y0 + 2), (x, bar_y1 - 2), col if alpha_done else (200, 200, 200), 3 if alpha_done else 1)
    xnow = 24 + int(span * frame_no / max(n_frames, 1))
    cv2.line(frame, (xnow, bar_y0 - 6), (xnow, bar_y1 + 6), INK, 2)
    put_text(frame, f"{frame_no / 10:6.1f} s", (24, h - 10), 0.55, INK)
    put_text(frame, state_text, (170, h - 10), 0.55, BLUE)
    # latest events, right side
    recent = [e for e in events_so_far if e["type"] in ("completion", "omission", "execution_error") and not e.get("at_end")][-3:]
    y = h - 96
    for e in recent:
        if e["type"] == "completion":
            txt = f"done  line {e.get('line')}: {e.get('text') or lines.get(e.get('line'), '')}"
            col = GREEN
        elif e["type"] == "execution_error":
            txt = f"FLAG  wrong part on component {e.get('component')}  (line {e.get('line')})"
            col = RED
        else:
            txt = f"FLAG  skipped: line {e.get('line')}"
            col = RED
        put_text(frame, txt[:90], (24, y), 0.52, col, 1)
        y += 20


def make_monitor(args) -> None:
    name = Path(args.video).stem
    events = json.loads((args.events / f"{name}.events.json").read_text())
    proc = Procedure.builtin("industreal_assembly" if "assy" in name else "industreal_maintenance")
    lines = {s.line: s.text for s in proc.steps.values()}
    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end or n_frames
    boxes = {}
    tops = {}
    for e in events:
        ev = e.get("evidence") or {}
        if ev.get("box"):
            boxes[e["frame"]] = ev["box"]
        if ev.get("top"):
            tops[e["frame"]] = (ev["top"], ev.get("top_p"))
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        out_i = 0
        last_box, last_top = None, None
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        fno = args.start
        while fno < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (fno - args.start) % args.speed == 0:
                frame = cv2.resize(frame, (args.width, int(frame.shape[0] * args.width / frame.shape[1])))
                h, w = frame.shape[:2]
                canvas = np.full((h + 120, w, 3), PAPER, dtype=np.uint8)
                canvas[:h] = frame
                b = boxes.get(fno, last_box)
                if b:
                    last_box = b
                    sx = w / 1280
                    x, y, bw, bh = [int(v * sx) for v in b]
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), GREEN, 2)
                t = tops.get(fno, last_top)
                if t:
                    last_top = t
                state_text = f"state {t[0].replace(',', '')}  p={t[1]:.2f}" if t and t[1] is not None else ""
                so_far = [e for e in events if e["frame"] <= fno]
                draw_panel(canvas, so_far, lines, state_text, fno, n_frames, events)
                put_text(canvas, f"{name}  frame {fno}", (12, 24), 0.55, WHITE, 1, bg=INK)
                cv2.imwrite(str(tdp / f"{out_i:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
                out_i += 1
            fno += 1
        cap.release()
        encode(tdp, args.out, fps=10.0 * args.speed / args.speed if args.hold_speed else 10.0)


def make_renders(args) -> None:
    recs = []
    for m in sorted(args.renders.glob("manifest_seed*.json")):
        recs.extend(json.loads(m.read_text()))
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    picked = recs[: args.n]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        i = 0
        for r in picked:
            img = cv2.imread(str(args.renders / "images" / r["image"]))
            if img is None:
                continue
            img = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])))
            h, w = img.shape[:2]
            sx = w / r["size"][0]
            x, y, bw, bh = [int(v * sx) for v in r["bbox_xywh"]]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), GREEN, 2)
            label = f"state {r['correct_state']}"
            if r["error_component"] is not None:
                label += f"   ERROR on component {r['error_component']}: {r['error_kind']}"
                col = RED
            else:
                col = INK
            put_text(img, label, (12, h - 14), 0.55, col, 1, bg=PAPER)
            put_text(img, "rendered from CAD, no real frame", (12, 24), 0.5, WHITE, 1, bg=INK)
            for _ in range(args.hold):
                cv2.imwrite(str(tdp / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                i += 1
        encode(tdp, args.out, fps=args.fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor")
    m.add_argument("--events", type=Path, required=True)
    m.add_argument("--video", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--end", type=int, default=0)
    m.add_argument("--speed", type=int, default=2, help="keep every Nth frame")
    m.add_argument("--width", type=int, default=960)
    m.add_argument("--hold-speed", action="store_true", help="play at real time instead of Nx")
    r = sub.add_parser("renders")
    r.add_argument("--renders", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--n", type=int, default=120)
    r.add_argument("--hold", type=int, default=3, help="frames to hold each render at the given fps")
    r.add_argument("--fps", type=float, default=12.0)
    r.add_argument("--width", type=int, default=960)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "monitor":
        make_monitor(args)
    else:
        make_renders(args)


if __name__ == "__main__":
    main()
