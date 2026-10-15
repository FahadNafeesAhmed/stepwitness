"""Frozen vision-foundation-model embeddings for assembly crops.

The verifier never trains the backbone. A crop goes in, a unit-length vector comes out, and everything
downstream (prototype banks, the small head, template contrast) works on that vector. DINOv2 is the
default because its weights are Apache-2.0 and it transfers well from renders to photographs; DINOv3
is a drop-in replacement once its gated weights are available locally.

Usage:
    from stepwitness.features import Embedder
    emb = Embedder("data/models/dinov2-base")
    vectors = emb.embed([pil_image, ...])   # (n, 768) float32, L2-normalised
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Embedder:
    def __init__(self, model_dir: str | Path, device: str | None = None, pooling: str = "cls+mean", image_size: int = 224):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir))
        # load on the CPU first: a direct load onto the GPU maps the safetensors file through the
        # Windows page file and fails with "paging file too small" when memory is tight
        self.model = AutoModel.from_pretrained(str(model_dir), device_map="cpu", low_cpu_mem_usage=True).to(self.device).eval()
        self.autocast = True  # mixed precision on CUDA; the profiler turns it off to measure true FP32
        self.pooling = pooling
        self.image_size = image_size
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self) -> int:
        d = self.model.config.hidden_size
        return 2 * d if self.pooling == "cls+mean" else d

    @torch.no_grad()
    def embed(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                size={"height": self.image_size, "width": self.image_size},
                do_center_crop=False,
            ).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda" and self.autocast):
                hidden = self.model(**inputs).last_hidden_state  # (b, 1 + patches, d)
            cls = hidden[:, 0]
            mean = hidden[:, 1:].mean(dim=1)
            if self.pooling == "cls":
                vec = cls
            elif self.pooling == "mean":
                vec = mean
            else:
                vec = torch.cat([cls, mean], dim=1)
            vec = torch.nn.functional.normalize(vec.float(), dim=1)
            out.append(vec.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)


    @torch.no_grad()
    def embed_tokens(self, images: list[Image.Image], image_size: int = 448, grid: int = 8, batch_size: int = 32) -> np.ndarray:
        """Patch tokens kept as a coarse grid instead of one pooled vector.

        Returns (n, 1 + grid*grid, dim) float16: the CLS token followed by the patch grid average-pooled
        to grid x grid cells. At 448 pixels a ViT-B/14 sees a 32 x 32 patch grid, so an 8 x 8 cell is
        4 x 4 patches, about 56 pixels of the crop; a clip or a pin head survives that pooling where it
        did not survive pooling the whole image to one vector.
        """
        out = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                size={"height": image_size, "width": image_size},
                do_center_crop=False,
            ).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda" and self.autocast):
                hidden = self.model(**inputs).last_hidden_state  # (b, 1 + P, d)
            cls = hidden[:, :1]
            patches = hidden[:, 1:]
            side = int(round(patches.shape[1] ** 0.5))
            g = patches.reshape(patches.shape[0], side, side, -1).permute(0, 3, 1, 2)  # (b, d, side, side)
            pooled = torch.nn.functional.adaptive_avg_pool2d(g.float(), grid).permute(0, 2, 3, 1).reshape(patches.shape[0], grid * grid, -1)
            toks = torch.cat([cls.float(), pooled], dim=1)
            out.append(toks.to(torch.float16).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1 + grid * grid, self.model.config.hidden_size), dtype=np.float16)


def crop_with_margin(image: Image.Image, bbox_xywh: list[float], margin: float = 0.15, square: bool = True) -> Image.Image:
    """Crop a box with a relative margin on each side; optionally pad to a square so the backbone sees
    the assembly at a stable scale regardless of its aspect ratio."""
    x, y, w, h = bbox_xywh
    mx, my = w * margin, h * margin
    x0, y0, x1, y1 = x - mx, y - my, x + w + mx, y + h + my
    if square:
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(image.width, int(x1)), min(image.height, int(y1))
    return image.crop((x0, y0, x1, y1))
