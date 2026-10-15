"""Verifier version 3: patch-token features and two heads.

Version 2 pooled the whole crop into one vector and learned 144 classes; it could not see a 3 mm
clip and had only a few dozen examples of each error class. Version 3 keeps the backbone's patch
tokens as a coarse grid, reads them with a small attention-pooling head, and predicts two things
the recognizer actually needs:

* a 22-way posterior over the correct states, and
* an 11-way set of probabilities that component k is installed wrongly.

"Component k is wrong" is the same target across every state that contains k, so each of the
eleven outputs is trained on hundreds of renders instead of a few dozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from stepwitness.procedure import N_COMPONENTS, labelled_states_industreal
from stepwitness.verifier import code_to_tuple, correct_of, tuple_to_code

STATE_CODES = [tuple_to_code(s) for _, s in sorted(labelled_states_industreal().items())]  # 22, in dataset order


class AttnPoolHead(nn.Module):
    def __init__(self, dim: int, n_states: int, n_components: int, n_queries: int = 4, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(dim * (n_queries + 1), hidden), nn.GELU(), nn.Dropout(dropout))
        self.state_out = nn.Linear(hidden, n_states)
        self.err_out = nn.Linear(hidden, n_components)

    def forward(self, tokens: torch.Tensor):
        x = self.ln(tokens.float())
        cls, patches = x[:, 0], x[:, 1:]
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(q, patches, patches, need_weights=False)
        feat = self.mlp(torch.cat([cls, pooled.flatten(1)], dim=1))
        return self.state_out(feat), self.err_out(feat)


def to_device_chunked(tokens: np.ndarray, device: str, chunk: int = 512) -> torch.Tensor:
    """Copy a (possibly memory-mapped) token array to the device a chunk at a time, so no full-size
    copy is ever made in RAM."""
    out = torch.empty(tuple(tokens.shape), dtype=torch.float16, device=device)
    for i in range(0, len(tokens), chunk):
        out[i : i + chunk] = torch.from_numpy(np.ascontiguousarray(tokens[i : i + chunk])).to(device)
    return out


def labels_from_codes(codes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(STATE_CODES)}
    y_state = np.array([idx[correct_of(c)] for c in codes], dtype=np.int64)
    y_err = np.zeros((len(codes), N_COMPONENTS), dtype=np.float32)
    for i, c in enumerate(codes):
        for k, v in enumerate(code_to_tuple(c)):
            if v == -1:
                y_err[i, k] = 1.0
    return y_state, y_err


def train_v3(
    tokens: np.ndarray,
    codes: list[str],
    epochs: int = 30,
    lr: float = 5e-4,
    batch: int = 128,
    seed: int = 0,
    pos_weight: float = 3.0,
    device: str | None = None,
    val: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[AttnPoolHead, dict]:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = to_device_chunked(tokens, device)
    y_state, y_err = labels_from_codes(codes)
    ys = torch.tensor(y_state, device=device)
    ye = torch.tensor(y_err, device=device)
    head = AttnPoolHead(x.shape[2], len(STATE_CODES), N_COMPONENTS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(x) + batch - 1) // batch))
    pw = torch.full((N_COMPONENTS,), pos_weight, device=device)
    history = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(x), device=device)
        tot, n = 0.0, 0
        for i in range(0, len(x), batch):
            b = perm[i : i + batch]
            xb = x[b].float()
            xb = xb + 0.02 * torch.randn_like(xb)
            ls, le = head(xb)
            loss = nn.functional.cross_entropy(ls, ys[b], label_smoothing=0.05) + nn.functional.binary_cross_entropy_with_logits(le, ye[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
            n += len(b)
        entry = {"epoch": ep, "loss": tot / n}
        if val is not None:
            entry.update(evaluate_v3(head, val[0], val[1], device))
        history.append(entry)
        print(f"  epoch {ep:2d} loss {entry['loss']:.3f}" + (f"  val state-acc {entry['state_acc']:.3f}  err-recall {entry['err_recall']:.3f}  err-fpr {entry['err_fpr']:.3f}" if val is not None else ""))
    return head, {"history": history}


@torch.no_grad()
def predict_v3(head: AttnPoolHead, tokens: np.ndarray, device=None, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    device = device or next(head.parameters()).device
    head.eval()
    ps, pe = [], []
    for i in range(0, len(tokens), batch):
        xb = torch.from_numpy(np.ascontiguousarray(tokens[i : i + batch])).to(device).float()
        ls, le = head(xb)
        ps.append(torch.softmax(ls, dim=1).cpu().numpy())
        pe.append(torch.sigmoid(le).cpu().numpy())
    return np.concatenate(ps), np.concatenate(pe)


def evaluate_v3(head: AttnPoolHead, tokens: np.ndarray, codes: list[str], device=None, threshold: float = 0.5) -> dict:
    ps, pe = predict_v3(head, tokens, device)
    y_state, y_err = labels_from_codes(codes)
    state_acc = float((ps.argmax(1) == y_state).mean())
    pred_err = pe > threshold
    true_err = y_err > 0.5
    tp = int((pred_err & true_err).sum())
    fp = int((pred_err & ~true_err).sum())
    fn = int((~pred_err & true_err).sum())
    tn = int((~pred_err & ~true_err).sum())
    # per crop: did we say "some error" when there was one
    crop_pred = pred_err.any(1)
    crop_true = true_err.any(1)
    ctp = int((crop_pred & crop_true).sum())
    cfp = int((crop_pred & ~crop_true).sum())
    cfn = int((~crop_pred & crop_true).sum())
    ctn = int((~crop_pred & ~crop_true).sum())
    # exact code: state right and the error components right
    pred_codes = []
    for i in range(len(codes)):
        t = list(code_to_tuple(STATE_CODES[int(ps[i].argmax())]))
        for k in range(N_COMPONENTS):
            if pred_err[i, k] and t[k] == 1:
                t[k] = -1
        pred_codes.append(tuple_to_code(t))
    exact = float(np.mean([p == c for p, c in zip(pred_codes, codes)]))
    return {
        "state_acc": state_acc,
        "exact_code_acc": exact,
        "err_precision": tp / max(tp + fp, 1),
        "err_recall": tp / max(tp + fn, 1),
        "err_fpr": fp / max(fp + tn, 1),
        "crop_err_precision": ctp / max(ctp + cfp, 1),
        "crop_err_recall": ctp / max(ctp + cfn, 1),
        "crop_err_fpr": cfp / max(cfp + ctn, 1),
    }


class MultiTaskVerifier:
    """Runtime wrapper: tokens in, (state posterior over STATE_CODES, component error probs) out."""

    def __init__(self, head: AttnPoolHead, image_size: int = 448, grid: int = 8):
        self.head = head
        self.classes = STATE_CODES
        self.image_size = image_size
        self.grid = grid

    def posterior_parts(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return predict_v3(self.head, tokens)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.head.state_dict(), "dim": self.head.ln.normalized_shape[0], "n_queries": self.head.queries.shape[0]}, folder / "head_v3.pt")
        (folder / "meta.json").write_text(json.dumps({"kind": "v3", "image_size": self.image_size, "grid": self.grid, "classes": self.classes}))

    @classmethod
    def load(cls, folder: Path, device: str | None = None) -> "MultiTaskVerifier":
        meta = json.loads((folder / "meta.json").read_text())
        ck = torch.load(folder / "head_v3.pt", map_location="cpu")
        head = AttnPoolHead(ck["dim"], len(STATE_CODES), N_COMPONENTS, n_queries=ck.get("n_queries", 4))
        head.load_state_dict(ck["state_dict"])
        head.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        return cls(head, meta.get("image_size", 448), meta.get("grid", 8))
