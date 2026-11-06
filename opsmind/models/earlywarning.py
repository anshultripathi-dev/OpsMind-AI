"""Module 7 — early warning: LSTM / GRU over the last 5 minutes, plus a gradient-boosting fallback.

Training of the recurrent networks happens in :mod:`opsmind.models.ew_worker` (separate
process, see its docstring). Inference here is a dependency-free NumPy LSTM, which is
what the API serves.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from ..config import ROOT


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTM:
    """Forward pass of the trained single-layer LSTM + MLP head (gate order i, f, g, o)."""

    KEYS = ("w_ih", "w_hh", "b_ih", "b_hh", "h1_w", "h1_b", "h2_w", "h2_b")

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(weights[k], dtype=np.float32) for k in self.KEYS}
        self.hidden = self.w["w_hh"].shape[1]

    @classmethod
    def load(cls, path: Path) -> "NumpyLSTM":
        d = np.load(path)
        return cls({k: d[k] for k in cls.KEYS})

    def save(self, path: Path) -> None:
        np.savez(path, **self.w)

    def forward(self, x: np.ndarray) -> np.ndarray:               # x: (B, L, d) -> logits (B,)
        w, H = self.w, self.hidden
        h = np.zeros((x.shape[0], H), np.float32)
        c = np.zeros_like(h)
        bias = w["b_ih"] + w["b_hh"]
        xi = x @ w["w_ih"].T + bias                                 # precompute input projections (B, L, 4H)
        for t in range(x.shape[1]):
            g = xi[:, t] + h @ w["w_hh"].T
            i, f, gg, o = (g[:, k * H:(k + 1) * H] for k in range(4))
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(gg)
            h = _sigmoid(o) * np.tanh(c)
        z = np.maximum(h @ w["h1_w"].T + w["h1_b"], 0.0)
        return (z @ w["h2_w"].T + w["h2_b"])[:, 0]

    def predict_proba(self, P: np.ndarray, idx: np.ndarray, rows: np.ndarray | None = None, batch: int = 8192) -> np.ndarray:
        rows = np.arange(len(idx)) if rows is None else rows
        out = [_sigmoid(self.forward(P[idx[rows[s:s + batch]]].astype(np.float32))) for s in range(0, len(rows), batch)]
        return np.concatenate(out) if out else np.zeros(0)


def train_sequence_models(P, idx, y, mask, tr, va, seed: int = 0, log=print) -> dict:
    """Train LSTM and GRU in a worker process; return NumPy LSTM + probabilities for all rows."""
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, P=P.astype(np.float32), idx=idx, y=y, mask=mask, tr=tr, va=va, seed=seed)
        proc = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "opsmind.models.ew_worker", str(inp), str(out)],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if "[lstm]" in line:
                log(line.rstrip())
        if proc.wait() != 0 or not out.exists():
            raise RuntimeError("early-warning worker failed")
        d = np.load(out)
        hist = json.loads(str(d["history"]))
        lstm = NumpyLSTM({k: d[k] for k in NumpyLSTM.KEYS})
        return dict(lstm=lstm, probs_lstm=d["probs_lstm"], probs_gru=d["probs_gru"],
                    history_lstm=hist["lstm"], history_gru=hist["gru"])


class GBForecaster:
    """Gradient-boosting fallback on the same sequence (as listed in the risk table)."""

    def __init__(self, seed: int = 0):
        self.m = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8,
                               colsample_bytree=0.7, tree_method="hist", n_jobs=-1, random_state=seed,
                               eval_metric="logloss")

    @staticmethod
    def features(P: np.ndarray, idx: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(len(idx)) if rows is None else rows
        S = P[idx[rows]]                                            # (n, L, d)
        last = S[:, -1]
        return np.concatenate([last, last - S[:, -4], last - S[:, 0], S.std(1)], axis=1)

    def fit(self, P, idx, y, rows):
        pos = float(y[rows].sum())
        self.m.set_params(scale_pos_weight=(len(rows) - pos) / max(pos, 1) * 0.5)
        self.m.fit(self.features(P, idx, rows), y[rows])
        return self

    def predict_proba(self, P, idx, rows=None):
        return self.m.predict_proba(self.features(P, idx, rows))[:, 1]
