"""Preprocessing and feature engineering (module 1).

Raw 5 s Prometheus samples -> cleaned series -> one feature vector per 30 s
window (every 10 s) made of rolling statistics, plus log-template counts from
Drain. The same functions are used offline (training) and online (serving).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import (CALLS, COMPONENTS, HORIZON_SEC, RAW_COLUMNS, RECOVERY_SEC, SAMPLE_SEC, SEQ_LEN, SEQ_STEP_WINDOWS,
                     STRIDE_SAMPLES, WINDOW_SAMPLES, WINDOW_SEC)

STATS = ("mean", "std", "min", "max", "delta")
METRIC_FEATURES = [f"{c}__{s}" for c in RAW_COLUMNS for s in STATS]

# Load-normalised block (optional): a traffic burst raises load, a fault changes the cost of each request.
#   <c>.cpu_per_req   CPU % per request/s       <b>.rps_from_<a>   callee throughput per caller request
EXTRA_FEATURES = ([f"{c}.cpu_per_req__mean" for c in COMPONENTS]
                  + [f"{b}.rps_from_{a}__mean" for a, d in CALLS.items() for b in d])
_MEAN_IDX = {c: METRIC_FEATURES.index(f"{c}__mean") for c in RAW_COLUMNS}


def extra_window_features(m: np.ndarray) -> np.ndarray:
    """(windows, len(METRIC_FEATURES)) -> (windows, len(EXTRA_FEATURES))."""
    mean = lambda col: m[:, _MEAN_IDX[col]]
    cols = [mean(f"{c}.cpu") / (mean(f"{c}.rps") + 5.0) for c in COMPONENTS]
    cols += [mean(f"{b}.rps") / (mean(f"{a}.rps") + 5.0) for a, d in CALLS.items() for b in d]
    return np.stack(cols, axis=1).astype(np.float32)


# ------------------------------------------------------------------------ cleaning
def ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill missing scrapes (NaN) per column; leading gaps get the column median."""
    out = a.astype(np.float32, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        nan = np.isnan(col)
        if not nan.any():
            continue
        if nan.all():
            col[:] = 0.0
            continue
        idx = np.where(~nan, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        if np.isnan(col[0]):
            first = np.flatnonzero(~nan)[0]
            col[:first] = col[first]
    return out


# ------------------------------------------------------------------------ windowing
def window_ends(n_steps: int) -> np.ndarray:
    """Index of the last sample of every window."""
    return np.arange(WINDOW_SAMPLES - 1, n_steps, STRIDE_SAMPLES)


def metric_window_features(arr: np.ndarray) -> np.ndarray:
    """(steps, cols) -> (windows, cols*len(STATS)); ordering matches METRIC_FEATURES."""
    sw = sliding_window_view(arr, WINDOW_SAMPLES, axis=0)[::STRIDE_SAMPLES]   # (w, cols, 6)
    half = WINDOW_SAMPLES // 2
    feats = np.stack([
        sw.mean(-1), sw.std(-1), sw.min(-1), sw.max(-1),
        sw[..., half:].mean(-1) - sw[..., :half].mean(-1),
    ], axis=-1)
    return feats.reshape(sw.shape[0], -1).astype(np.float32)


def log_window_features(step_counts: np.ndarray) -> np.ndarray:
    """Rolling sum of per-step log counts over the same windows."""
    cs = np.concatenate([np.zeros((1, step_counts.shape[1]), step_counts.dtype), np.cumsum(step_counts, 0)])
    starts = np.arange(0, step_counts.shape[0] - WINDOW_SAMPLES + 1, STRIDE_SAMPLES)
    return (cs[starts + WINDOW_SAMPLES] - cs[starts]).astype(np.float32)


def single_window_features(buf: np.ndarray, log_counts: np.ndarray, load_norm: bool = False) -> np.ndarray:
    """Online variant: ``buf`` = last WINDOW_SAMPLES cleaned samples, ``log_counts`` = their log counts."""
    m = metric_window_features(buf)
    parts = [m[0]] + ([extra_window_features(m)[0]] if load_norm else []) + [log_counts.sum(0).astype(np.float32)]
    return np.concatenate(parts)


# ------------------------------------------------------------------------ labelling
def label_windows(truth: dict, ends: np.ndarray) -> dict[str, np.ndarray]:
    """Labels derived exclusively from the injector log (no manual labelling).

    * ``anomaly``   window overlaps the fault interval by >= 50 %
    * ``grace``     onset/recovery windows whose label is ambiguous -> excluded from scoring
    * ``warn``      an incident (user impact) starts within HORIZON_SEC or is ongoing
    * ``warn_mask`` windows usable for the early-warning task
    * ``active``    user-visible incident ongoing
    """
    tw = (ends * SAMPLE_SEC + SAMPLE_SEC).astype(np.float64)       # time the window becomes available
    a = tw - WINDOW_SEC
    n = len(ends)
    anomaly = np.zeros(n, bool)
    grace = np.zeros(n, bool)
    warn = np.zeros(n, bool)
    warn_mask = np.ones(n, bool)
    active = np.zeros(n, bool)
    if truth.get("fault_type"):
        fs, fe, imp = truth["fault_start"], truth["fault_end"], truth.get("impact_start")
        overlap = np.clip(np.minimum(tw, fe) - np.maximum(a, fs), 0, None) / WINDOW_SEC
        anomaly = overlap >= 0.5
        grace = (~anomaly) & (((tw > fs) & (a < fs)) | ((tw > fe) & (a < fe + RECOVERY_SEC)))
        tail = (tw > fe) & (a < fe + RECOVERY_SEC)
        if imp is None:
            warn_mask = ~(((tw > fs) & (a < fe + RECOVERY_SEC)))
        else:
            warn = (tw >= imp - HORIZON_SEC) & (tw <= fe)
            active = (tw >= imp) & (tw <= fe)
            warn_mask = ~(((tw > fs) & (tw < imp - HORIZON_SEC)) | tail)
    return dict(tw=tw, anomaly=anomaly, grace=grace, warn=warn, warn_mask=warn_mask, active=active)


# ------------------------------------------------------------------------ scaling
class NormalScaler:
    """z-scores relative to *normal* operation, with a variance floor.

    Features that never move during normal operation (e.g. error counts) would
    otherwise explode to huge z-values at the first deviation.
    """

    def __init__(self, floor_frac: float = 0.02, clip: float = 25.0):
        self.floor_frac, self.clip = floor_frac, clip

    def fit(self, X_normal: np.ndarray, X_all: np.ndarray) -> "NormalScaler":
        self.mean_ = X_normal.mean(0)
        std = X_normal.std(0)
        rng = np.percentile(X_all, 99, axis=0) - np.percentile(X_all, 1, axis=0)
        self.scale_ = np.maximum(std, self.floor_frac * rng + 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean_) / self.scale_, -self.clip, self.clip).astype(np.float32)

    def fit_transform(self, X_normal, X_all):
        return self.fit(X_normal, X_all).transform(X_all)


# ------------------------------------------------------------------------ sequences
def sequence_index(run_start: np.ndarray, n: int, seq_len: int = SEQ_LEN, step: int = SEQ_STEP_WINDOWS) -> np.ndarray:
    """Row indices of the ``seq_len`` windows (oldest first) ending at every row.

    ``run_start[i]`` is the first row of the run that row ``i`` belongs to; history
    before the start of a run is padded by repeating the first window.
    """
    i = np.arange(n)[:, None]
    lags = (seq_len - 1 - np.arange(seq_len))[None, :] * step
    return np.maximum(i - lags, run_start[:, None])
