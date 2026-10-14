"""The state verifier: frozen embeddings, a prototype bank per state, and a small trained head.

Classes are label states as comma-joined 11-digit codes with -1 allowed, for example
"1,1,1,0,0,0,0,0,0,0,0" (correct state 5) or "1,1,-1,0,0,0,0,0,0,0,0" (front chassis pin wrong).
Two views of the verifier's output are kept side by side because they fail differently:

* prototype similarity: nearest-centroid over a bank of k-means centroids per class, cosine metric.
  It needs no training, degrades gracefully, and adding real photos later is just adding prototypes.
* head: a two-layer MLP on the embedding trained with cross-entropy. It learns the fine differences
  the prototypes blur, at the cost of needing enough renders per class.

``predict`` blends the two. ``error_contrast`` compares a crop against the correct state and its
error variants for one component, which is what the recognizer asks when a step is about to complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


def code_to_tuple(code: str) -> tuple[int, ...]:
    return tuple(int(v) for v in code.split(","))


def tuple_to_code(state) -> str:
    return ",".join(str(int(v)) for v in state)


def correct_of(code: str) -> str:
    """The correct state a label belongs to: -1 digits are the component that is wrong, so the
    correct code has the same digits with -1 read as 1."""
    return tuple_to_code(1 if v == -1 else v for v in code_to_tuple(code))


def error_component(code: str) -> int | None:
    t = code_to_tuple(code)
    bad = [i for i, v in enumerate(t) if v == -1]
    return bad[0] if bad else None


# ----------------------------------------------------------------------------------------------
# prototype bank
# ----------------------------------------------------------------------------------------------

@dataclass
class PrototypeBank:
    classes: list[str] = field(default_factory=list)
    prototypes: np.ndarray | None = None  # (n_protos, dim), unit length
    owner: np.ndarray | None = None  # (n_protos,) index into classes

    @classmethod
    def build(cls, embeddings: np.ndarray, labels: list[str], per_class: int = 16, seed: int = 0) -> "PrototypeBank":
        rng = np.random.default_rng(seed)
        classes = sorted(set(labels))
        protos, owners = [], []
        for ci, c in enumerate(classes):
            x = embeddings[[i for i, l in enumerate(labels) if l == c]]
            k = min(per_class, len(x))
            centres = _kmeans(x, k, rng)
            protos.append(centres)
            owners.extend([ci] * len(centres))
        bank = cls(classes, np.concatenate(protos, axis=0), np.array(owners))
        return bank

    def similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """(n, n_classes): best cosine similarity to any prototype of each class."""
        sims = embeddings @ self.prototypes.T  # (n, n_protos)
        out = np.full((len(embeddings), len(self.classes)), -1.0, dtype=np.float32)
        for ci in range(len(self.classes)):
            mask = self.owner == ci
            out[:, ci] = sims[:, mask].max(axis=1)
        return out

    def save(self, path: Path) -> None:
        np.savez(path, prototypes=self.prototypes, owner=self.owner, classes=np.array(self.classes))

    @classmethod
    def load(cls, path: Path) -> "PrototypeBank":
        z = np.load(path, allow_pickle=False)
        return cls(list(z["classes"]), z["prototypes"], z["owner"])


def _kmeans(x: np.ndarray, k: int, rng, iters: int = 25) -> np.ndarray:
    if k >= len(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    centres = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        sims = x @ centres.T
        assign = sims.argmax(axis=1)
        for j in range(k):
            members = x[assign == j]
            if len(members):
                centres[j] = members.mean(axis=0)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True) + 1e-9
    return centres


# ----------------------------------------------------------------------------------------------
# head
# ----------------------------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def train_head(
    embeddings: np.ndarray,
    labels: list[str],
    classes: list[str],
    epochs: int = 40,
    lr: float = 1e-3,
    batch: int = 256,
    seed: int = 0,
    device: str | None = None,
    val: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[Head, dict]:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    idx = {c: i for i, c in enumerate(classes)}
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    y = torch.tensor([idx[l] for l in labels], dtype=torch.long, device=device)
    head = Head(x.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(x), device=device)
        total, correct_n = 0.0, 0
        for i in range(0, len(x), batch):
            b = perm[i : i + batch]
            xb = x[b] + 0.02 * torch.randn_like(x[b])  # light feature noise as augmentation
            logits = head(xb)
            loss = nn.functional.cross_entropy(logits, y[b], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
            correct_n += int((logits.argmax(1) == y[b]).sum())
        sched.step()
        entry = {"epoch": ep, "loss": total / len(x), "train_acc": correct_n / len(x)}
        if val is not None:
            entry["val_acc"] = evaluate_head(head, val[0], val[1], classes, device)
        history.append(entry)
    return head, {"history": history, "classes": classes}


@torch.no_grad()
def head_logits(head: Head, embeddings: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or next(head.parameters()).device
    head.eval()
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    return head(x).cpu().numpy()


def evaluate_head(head: Head, embeddings: np.ndarray, labels: list[str], classes: list[str], device=None) -> float:
    pred = head_logits(head, embeddings, device).argmax(1)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx.get(l, -1) for l in labels])
    return float((pred == y).mean())


# ----------------------------------------------------------------------------------------------
# verifier
# ----------------------------------------------------------------------------------------------

class Verifier:
    def __init__(self, bank: PrototypeBank, head: Head | None = None, temperature: float = 0.05, blend: float = 0.5):
        self.bank = bank
        self.head = head
        self.temperature = temperature
        self.blend = blend
        self.classes = bank.classes

    def posterior(self, embeddings: np.ndarray) -> np.ndarray:
        """(n, n_classes) probabilities blending prototype similarity and the head."""
        sims = self.bank.similarities(embeddings)
        logits = sims / self.temperature
        if self.head is not None:
            h = head_logits(self.head, embeddings)
            logits = self.blend * h + (1 - self.blend) * logits
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, embeddings: np.ndarray) -> list[str]:
        return [self.classes[i] for i in self.posterior(embeddings).argmax(axis=1)]

    def error_contrast(self, embedding: np.ndarray, correct_code: str, component: int) -> dict:
        """Compare one crop against the correct state and the error variants of ``component``.
        Returns best similarities and the margin (error minus correct); positive favours an error."""
        sims = self.bank.similarities(embedding[None])[0]
        correct_idx = self.classes.index(correct_code) if correct_code in self.classes else None
        err_codes = [c for c in self.classes if correct_of(c) == correct_code and error_component(c) == component]
        best_correct = float(sims[correct_idx]) if correct_idx is not None else float("nan")
        best_error = max((float(sims[self.classes.index(c)]) for c in err_codes), default=float("nan"))
        return {"best_correct": best_correct, "best_error": best_error, "margin": best_error - best_correct, "n_error_classes": len(err_codes)}

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.bank.save(folder / "bank.npz")
        meta = {"temperature": self.temperature, "blend": self.blend, "classes": self.classes, "has_head": self.head is not None}
        (folder / "meta.json").write_text(json.dumps(meta))
        if self.head is not None:
            torch.save({"state_dict": self.head.state_dict(), "dim": self.head.net[0].in_features, "n_classes": len(self.classes)}, folder / "head.pt")

    @classmethod
    def load(cls, folder: Path, device: str | None = None) -> "Verifier":
        meta = json.loads((folder / "meta.json").read_text())
        bank = PrototypeBank.load(folder / "bank.npz")
        head = None
        if meta.get("has_head") and (folder / "head.pt").exists():
            ck = torch.load(folder / "head.pt", map_location="cpu")
            head = Head(ck["dim"], ck["n_classes"])
            head.load_state_dict(ck["state_dict"])
            head.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        return cls(bank, head, meta["temperature"], meta["blend"])
