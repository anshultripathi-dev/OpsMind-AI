"""PyTorch training worker for the early-warning networks.

Runs as its own process: on macOS, PyTorch and XGBoost ship different OpenMP runtimes
that crash or deadlock when loaded into the same interpreter. The trained LSTM is
exported as plain NumPy weights, so serving needs neither torch nor this file.

    python -m opsmind.models.ew_worker <in.npz> <out.npz>
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np


def _build(kind: str, d_in: int, hidden: int, dropout: float = 0.25):
    import torch.nn as nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind](d_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1]).squeeze(-1)

    return SeqNet()


def _proba(model, P_t, idx, rows, batch=4096):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            b = rows[s:s + batch]
            out.append(torch.sigmoid(model(P_t[torch.from_numpy(idx[b])])).numpy())
    return np.concatenate(out)


def train(kind, P, idx, y, mask, tr_rows, va_rows, epochs=30, hidden=64, batch=512, lr=2e-3, seed=0, patience=6):
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = tr_rows[mask[tr_rows] == 1]
    va = va_rows[mask[va_rows] == 1]
    model = _build(kind, P.shape[1], hidden)
    pos = float(y[tr].sum())
    neg = float(len(tr) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((neg / max(pos, 1)) ** 0.5))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    P_t = torch.from_numpy(P.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        total = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            x = P_t[torch.from_numpy(idx[b])]
            x = x + 0.03 * torch.randn_like(x)                  # light noise augmentation
            opt.zero_grad()
            loss = loss_fn(model(x), y_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss) * len(b)
        ap = float(average_precision_score(y[va], _proba(model, P_t, idx, va)))
        history.append(dict(epoch=ep + 1, loss=round(total / len(perm), 4), val_ap=round(ap, 4)))
        print(f"      [{kind}] epoch {ep + 1:02d}  loss {total / len(perm):.4f}  val AP {ap:.4f}", flush=True)
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, _proba(model, P_t, idx, np.arange(len(idx)))


def main(inp: str, out: str) -> None:
    d = np.load(inp)
    P, idx, y, mask, tr, va = (d[k] for k in ("P", "idx", "y", "mask", "tr", "va"))
    seed = int(d["seed"])
    result: dict = {}
    hist = {}
    for kind in ("lstm", "gru"):
        model, h, probs = train(kind, P, idx, y, mask, tr, va, seed=seed)
        hist[kind] = h
        result[f"probs_{kind}"] = probs
        if kind == "lstm":
            sd = {k: v.numpy() for k, v in model.state_dict().items()}
            result.update(w_ih=sd["rnn.weight_ih_l0"], w_hh=sd["rnn.weight_hh_l0"],
                          b_ih=sd["rnn.bias_ih_l0"], b_hh=sd["rnn.bias_hh_l0"],
                          h1_w=sd["head.1.weight"], h1_b=sd["head.1.bias"],
                          h2_w=sd["head.3.weight"], h2_b=sd["head.3.bias"])
    np.savez(out, history=json.dumps(hist), **result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
