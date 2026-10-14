"""Compose derived/original render pairs into one sheet: one row per state, derived left, original right.

Usage:
    python tools/compose_pairs.py runs/phase2/derive --out runs/phase2/derive/sheet.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=2, help="state pairs per row")
    ap.add_argument("--thumb", type=int, nargs=2, default=[360, 225])
    args = ap.parse_args()

    derived = sorted(args.folder.glob("state*_derived.jpg"))
    tw, th = args.thumb
    pair_w = 2 * tw
    rows = (len(derived) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * pair_w, rows * (th + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, d in enumerate(derived):
        o = d.with_name(d.name.replace("_derived", "_original"))
        x = (i % args.cols) * pair_w
        y = (i // args.cols) * (th + 18)
        draw.text((x + 6, y + 2), d.stem.replace("_derived", "") + "   derived | original", fill="black")
        sheet.paste(Image.open(d).convert("RGB").resize((tw, th)), (x, y + 18))
        if o.exists():
            sheet.paste(Image.open(o).convert("RGB").resize((tw, th)), (x + tw, y + 18))
    sheet.save(args.out, quality=90)
    print(f"wrote {args.out} ({len(derived)} pairs)")


if __name__ == "__main__":
    main()
