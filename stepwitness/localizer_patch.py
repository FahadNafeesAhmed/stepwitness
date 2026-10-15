"""A fallback localizer that needs no detector training: patch tokens of the frozen backbone over
the whole frame, a per-patch logistic classifier (assembly or not) fitted on renders, and the box
of the largest blob of assembly patches.

It exists because detector training needs more memory than this machine has spare. It is coarser
than a detector (one cell is about 40 pixels of a 1280-wide frame), which the verifier tolerates
because it crops with a margin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stepwitness.features import Embedder


class PatchLocalizer:
    def __init__(self, embedder: Embedder, w: np.ndarray, b: float, image_size: int = 448, grid: int = 16, threshold: float = 0.5):
        self.embedder = embedder
        self.w = w.astype(np.float32)
        self.b = float(b)
        self.image_size = image_size
        self.grid = grid
        self.threshold = threshold

    def heatmap(self, image: Image.Image) -> np.ndarray:
        toks = self.embedder.embed_tokens([image], image_size=self.image_size, grid=self.grid)[0, 1:].astype(np.float32)
        logits = toks @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-logits))
        return p.reshape(self.grid, self.grid)

    def locate(self, image: Image.Image) -> tuple[list[float] | None, float]:
        """Box (x, y, w, h) in image pixels of the largest connected blob of assembly cells, and its mean score."""
        hm = self.heatmap(image)
        mask = hm > self.threshold
        if not mask.any():
            return None, 0.0
        # largest 4-connected component
        seen = np.zeros_like(mask)
        best, best_score = None, 0.0
        for i in range(self.grid):
            for j in range(self.grid):
                if mask[i, j] and not seen[i, j]:
                    stack, cells = [(i, j)], []
                    seen[i, j] = True
                    while stack:
                        a, c = stack.pop()
                        cells.append((a, c))
                        for da, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            na, nc = a + da, c + dc
                            if 0 <= na < self.grid and 0 <= nc < self.grid and mask[na, nc] and not seen[na, nc]:
                                seen[na, nc] = True
                                stack.append((na, nc))
                    if len(cells) > (len(best) if best else 0):
                        best = cells
                        best_score = float(np.mean([hm[a, c] for a, c in cells]))
        rows = [a for a, _ in best]
        cols = [c for _, c in best]
        W, H = image.size
        cw, ch = W / self.grid, H / self.grid
        x0, x1 = min(cols) * cw, (max(cols) + 1) * cw
        y0, y1 = min(rows) * ch, (max(rows) + 1) * ch
        return [x0, y0, x1 - x0, y1 - y0], best_score

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "patch_localizer.npz", w=self.w, b=np.array([self.b]))
        (folder / "meta.json").write_text(json.dumps({"image_size": self.image_size, "grid": self.grid, "threshold": self.threshold}))

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> "PatchLocalizer":
        z = np.load(folder / "patch_localizer.npz")
        meta = json.loads((folder / "meta.json").read_text())
        return cls(embedder, z["w"], float(z["b"][0]), meta["image_size"], meta["grid"], meta["threshold"])


def fit_patch_classifier(tokens: np.ndarray, labels: np.ndarray, epochs: int = 200, lr: float = 0.05, l2: float = 1e-3, device: str | None = None) -> tuple[np.ndarray, float]:
    """Logistic regression on (n_patches, dim) tokens with 0/1 labels; returns weights and bias."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(tokens, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pos = y.mean().clamp(1e-3, 1 - 1e-3)
    pw = ((1 - pos) / pos).item()
    for _ in range(epochs):
        logits = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(pw, device=device)) + l2 * (w * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().cpu().numpy(), float(b.item())
