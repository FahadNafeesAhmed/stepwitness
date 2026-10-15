"""Draw the architecture diagram in a hand-sketched style (thin ink, double strokes, slight wobble,
white and pastel fills, handwritten labels) and splice it into site/index.html.

Usage:
    python tools/draw_sketch.py [--site site/index.html] [--seed 7]
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK = "#1e1e1e"
FONT = "'Patrick Hand', 'Comic Sans MS', cursive"


class Sketch:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.parts: list[str] = []

    # ----- primitives --------------------------------------------------------------------
    def _wobble(self, x1, y1, x2, y2, amount=1.6) -> str:
        """One hand-drawn stroke from (x1,y1) to (x2,y2): a cubic curve whose control points sit a
        little off the straight line, and endpoints that miss by a hair."""
        r = self.rng
        n = math.hypot(x2 - x1, y2 - y1)
        a = amount * min(1.0, n / 120) + 0.4
        ex = lambda: r.uniform(-a, a)  # noqa: E731
        sx, sy = x1 + ex() * 0.6, y1 + ex() * 0.6
        tx, ty = x2 + ex() * 0.6, y2 + ex() * 0.6
        c1x, c1y = x1 + (x2 - x1) * r.uniform(0.25, 0.4) + ex(), y1 + (y2 - y1) * r.uniform(0.25, 0.4) + ex()
        c2x, c2y = x1 + (x2 - x1) * r.uniform(0.6, 0.75) + ex(), y1 + (y2 - y1) * r.uniform(0.6, 0.75) + ex()
        return f"M{sx:.1f} {sy:.1f} C{c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {tx:.1f} {ty:.1f}"

    def line(self, x1, y1, x2, y2, dash: str | None = None, width=1.8, double=True):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        d = self._wobble(x1, y1, x2, y2)
        self.parts.append(f'<path d="{d}" stroke="{INK}" stroke-width="{width}" fill="none" stroke-linecap="round"{extra}/>')
        if double:
            d2 = self._wobble(x1, y1, x2, y2, amount=1.2)
            self.parts.append(f'<path d="{d2}" stroke="{INK}" stroke-width="{width * 0.7:.1f}" fill="none" stroke-linecap="round" opacity="0.8"{extra}/>')

    def rect(self, x, y, w, h, fill="#ffffff"):
        r = self.rng
        j = lambda: r.uniform(-1.2, 1.2)  # noqa: E731
        pts = [(x + j(), y + j()), (x + w + j(), y + j()), (x + w + j(), y + h + j()), (x + j(), y + h + j())]
        poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        self.parts.append(f'<polygon points="{poly}" fill="{fill}" stroke="none"/>')
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            self.line(ax, ay, bx, by)

    def arrow(self, pts: list[tuple[float, float]], dash: str | None = None):
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            self.line(ax, ay, bx, by, dash=dash)
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        ang = math.atan2(by - ay, bx - ax)
        for s in (1, -1):
            a2 = ang + s * math.radians(155)
            self.line(bx, by, bx + 11 * math.cos(a2), by + 11 * math.sin(a2), width=1.8, double=False)

    def text(self, x, y, s, size=14, anchor="start", color=INK, spacing=None, weight=None):
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        extra += f' font-weight="{weight}"' if weight else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{extra}>{s}</text>')

    def box(self, x, y, w, h, title, lines, fill="#ffffff", num=None):
        self.rect(x, y, w, h, fill)
        tx = x + 12
        if num is not None:
            self.text(tx, y + 22, f"{num}", size=13, color="#1971c2")
            tx += 16
        self.text(tx, y + 23, title, size=18)
        for i, s in enumerate(lines):
            self.text(x + 12, y + 44 + i * 18, s, size=14.5, color="#3a3a3a")

    def svg(self, w, h, title) -> str:
        body = "\n".join(self.parts)
        return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-labelledby="dgt" xmlns="http://www.w3.org/2000/svg">\n'
                f"<title id=\"dgt\">{title}</title>\n{body}\n</svg>")


def draw(seed: int) -> str:
    s = Sketch(seed)
    W, H = 1300, 600
    # lanes: a handwritten label and a dashed separator, no shading
    s.text(26, 34, "SETUP  ·  once per job", size=14, color="#6b6b6b", spacing="1.5")
    s.line(20, 312, 1280, 312, dash="7 7", width=1.3, double=False)
    s.text(26, 346, "RUNTIME  ·  every frame  ·  runs as the edge package on site hardware", size=14, color="#6b6b6b", spacing="1.5")

    # setup lane
    s.box(24, 60, 190, 56, "Part geometry", ["CAD files, FBX or 3MF"])
    s.box(24, 134, 190, 56, "Work instruction", ["plain text, the job's sheet"])
    s.box(270, 60, 240, 92, "Render factory", ["Blender; random light, camera, hands", "22 correct + generated error states", "→ labelled renders"], num=2)
    s.box(580, 60, 240, 92, "State verifier (train)", ["frozen vision foundation model, 448 px", "patch grid + attention-pool head", "22 states, 11 component-error outputs"], fill="#ffec99", num=3)
    s.box(880, 60, 200, 92, "Edge package", ["FP32 / FP16 timed on the GPU", "profile.json, never by hand", "→ deployable package"], fill="#a5d8ff", num=6)
    s.box(1130, 60, 156, 92, "Procedure", ["compiler", "LLM draft, one review", "→ procedure graph"], fill="#b2f2bb", num=1)

    s.arrow([(214, 88), (242, 88), (242, 106), (270, 106)])
    s.arrow([(214, 162), (1105, 162), (1105, 106), (1130, 106)])
    s.text(640, 156, "instruction text", size=14, color="#6b6b6b")
    s.arrow([(510, 106), (580, 106)])
    s.text(522, 99, "renders", size=14, color="#6b6b6b")
    s.arrow([(820, 106), (880, 106)])
    s.text(830, 99, "weights", size=14, color="#6b6b6b")

    # produced in setup, used at runtime
    s.arrow([(980, 152), (980, 396)], dash="6 5")
    s.text(990, 260, "deploy", size=14, color="#1971c2")
    s.arrow([(1208, 152), (1208, 290), (1030, 290), (1030, 396)], dash="6 5")
    s.text(1040, 283, "procedure graph", size=14, color="#1971c2")

    # runtime lane
    s.box(24, 398, 140, 92, "Camera or video", ["720p RGB, 10 fps"])
    s.box(204, 398, 306, 92, "Localize assembly", ["one-class detector trained on renders", "finds the box, nothing else", "→ crop"])
    s.box(580, 398, 240, 92, "State verifier (run)", ["state posterior over the 22 states", "error probability per component", "→ state · error · unknown, per frame"], fill="#ffec99", num=3)
    s.box(880, 398, 200, 92, "Step recognizer", ["accumulate and decay", "procedure guards: reach, hold", "→ completions, flags"], fill="#ffc9c9", num=4)
    s.box(1130, 398, 156, 92, "Evidence", ["timeline", "events and flags with", "frame and line links"], num=5)

    s.rect(580, 526, 240, 56, fill="#ffffff")
    s.text(592, 548, "7", size=13, color="#1971c2")
    s.text(608, 548, "Optional VLM check", size=16)
    s.text(592, 568, "yes/no with crop and line; evidence only", size=13, color="#3a3a3a")
    s.arrow([(700, 490), (700, 524)], dash="6 5")
    s.text(708, 512, "low confidence only", size=14, color="#6b6b6b")

    s.arrow([(164, 444), (204, 444)])
    s.arrow([(510, 444), (580, 444)])
    s.arrow([(820, 444), (880, 444)])
    s.arrow([(1080, 444), (1130, 444)])
    return s.svg(W, H, "Stepwitness architecture: setup lane and runtime lane")




# ----- results chart ---------------------------------------------------------------------------
PALETTE = {"real": "#e9ecef", "synth": "#ffffff", "on": "#ffec99", "off": "#ffc9c9"}
LABELS = {"real": "B3, real video", "synth": "B3, synthetic only", "on": "Stepwitness, guards on", "off": "Stepwitness, guards off"}


def load_results() -> dict:
    """Four systems x {all, with_errors} x {pos, f1, delay_s}, read from the score files."""
    import json

    b = json.loads((ROOT / "runs/phase0/baseline_results.json").read_text())
    g = json.loads((ROOT / "runs/phase4/monitor_v3_guarded/psr_results.json").read_text())
    q = json.loads((ROOT / "runs/phase4/monitor_v3_plain/psr_results.json").read_text())
    return {"real": b["real_trained/expected"], "synth": b["synthetic_only/expected"], "on": g, "off": q}


def draw_results(seed: int) -> str:
    res = load_results()
    s = Sketch(seed + 1)
    W, H = 760, 400
    systems = ["real", "synth", "on", "off"]
    panels = [("all", "All 32 recordings", 40), ("with_errors", "The 15 recordings with errors", 400)]
    y0, y1 = 300, 70  # baseline and top of the 0..1 axis
    for key, title, x0 in panels:
        s.text(x0 + 2, 38, title, size=17)
        # axis with ticks at 0, 0.25 ... 1
        s.line(x0, y1 - 6, x0, y0)
        s.line(x0, y0, x0 + 300, y0)
        for t in range(5):
            yy = y0 - (y0 - y1) * t / 4
            s.line(x0 - 5, yy, x0, yy, width=1.3, double=False)
            s.text(x0 - 9, yy + 4, f"{t / 4:.2f}".rstrip("0").rstrip(".") if t else "0", size=11.5, anchor="end", color="#6b6b6b")
        for gi, (metric, mlabel) in enumerate([("pos", "order similarity (POS)"), ("f1", "F1")]):
            gx = x0 + 24 + gi * 150
            for si, sysk in enumerate(systems):
                v = res[sysk][key][metric]
                bx = gx + si * 30
                bh = (y0 - y1) * v
                s.rect(bx, y0 - bh, 24, bh, fill=PALETTE[sysk])
                s.text(bx + 12, y0 - bh - 6, f"{v:.2f}", size=11.5, anchor="middle")
            s.text(gx + 60, y0 + 20, mlabel, size=13, anchor="middle", color="#3a3a3a")
    # legend
    lx = 60
    for sysk in systems:
        s.rect(lx, 348, 18, 14, fill=PALETTE[sysk])
        s.text(lx + 26, 360, LABELS[sysk], size=13, color="#3a3a3a")
        lx += 26 + 8 * len(LABELS[sysk]) + 14
    s.text(W / 2, 390, "same 32 test recordings, the authors' metric code; higher is better", size=12.5, anchor="middle", color="#6b6b6b")
    return s.svg(W, H, "Order similarity and F1 for the two baselines and the two Stepwitness settings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=ROOT / "site" / "index.html")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-results", action="store_true", help="skip the results chart (score files absent)")
    args = ap.parse_args()
    svg = draw(args.seed)
    html = args.site.read_text(encoding="utf-8")
    pattern = re.compile(r'(<div class="diagram">\n)<svg viewBox="0 0 1300 \d+".*?</svg>', re.S)
    assert len(pattern.findall(html)) == 1, "diagram block not found exactly once"
    html = pattern.sub(lambda m: m.group(1) + svg, html)
    print(f"diagram written ({len(svg)} chars)")
    if not args.no_results:
        fig = ('<figure class="card sketch" id="results-sketch">\n' + draw_results(args.seed) + "\n"
               '<figcaption>Order similarity and F1 on the test split. Grey and white are the authors\u2019 baseline trained on real video and on their synthetic images; yellow and pink are Stepwitness, trained on renders only, with the procedure guards on and off. Drawn from the score files.</figcaption>\n</figure>\n')
        existing = re.compile(r'<figure class="card sketch" id="results-sketch">.*?</figure>\n', re.S)
        if existing.search(html):
            html = existing.sub(lambda m: fig, html)
        else:
            anchor = '<div class="tbl"><table>\n<thead><tr><th>Baseline</th><th>Trained on</th>'
            assert html.count(anchor) == 1, "results table anchor not found exactly once"
            html = html.replace(anchor, fig + anchor)
        print("results chart written")
    args.site.write_text(html, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
