"""Static-threshold baseline = Prometheus-style alert rules (the baseline of the proposal)."""
from __future__ import annotations

import numpy as np

from .config import COMPONENTS, METRICS, THRESHOLD_RULES

_OPS = {">": np.greater, "<": np.less}


class ThresholdBaseline:
    """Fires when any rule is violated on the window mean of a metric."""

    def __init__(self, feature_names: list[str]):
        idx = {n: i for i, n in enumerate(feature_names)}
        self.rules: list[tuple[str, str, int, float, str]] = []     # (rule, component, col, thr, op)
        for rule, (metric, op, thr) in THRESHOLD_RULES.items():
            for c in COMPONENTS:
                if metric in METRICS[c]:
                    self.rules.append((rule, c, idx[f"{c}.{metric}__mean"], thr, op))

    def flags(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X), bool)
        for _, _, j, thr, op in self.rules:
            out |= _OPS[op](X[:, j], thr)
        return out

    def firing(self, x: np.ndarray) -> list[dict]:
        """Rules firing for one feature vector (used by the live dashboard)."""
        res = []
        for rule, c, j, thr, op in self.rules:
            if _OPS[op](x[j], thr):
                res.append(dict(rule=rule, component=c, value=round(float(x[j]), 2), threshold=thr))
        return res
