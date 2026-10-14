"""Compose numbered thumbnails into one contact sheet image.

Usage:
    python tools/compose_sheet.py runs/phase2/shells_state13 --cols 6 --out runs/phase2/shells_state13/sheet.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thumb", type=int, nargs=2, default=[320, 200])
    args = ap.parse_args()

    files = sorted(args.folder.glob("shell_*.jpg"))
    tw, th = args.thumb
    rows = (len(files) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * tw, rows * th), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((tw, th))
        x, y = (i % args.cols) * tw, (i // args.cols) * th
        sheet.paste(im, (x, y))
        draw.rectangle([x + 4, y + 4, x + 54, y + 28], fill="black")
        draw.text((x + 10, y + 8), f.stem.split("_")[1], fill="white")
    sheet.save(args.out, quality=90)
    print(f"wrote {args.out} ({len(files)} thumbnails)")


if __name__ == "__main__":
    main()
