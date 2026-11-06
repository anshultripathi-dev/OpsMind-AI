"""Log parsing with Drain (He et al., 2017) via Drain3, plus log-derived features.

Drain is trained once on raw messages to discover event templates, then used in
match mode (fast, deterministic template ids) for the rest of the data and at
serving time.
"""
from __future__ import annotations

import io
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from .config import COMPONENTS, SAMPLE_SEC

LEVELS = ("INFO", "WARN", "ERROR", "FATAL")


def _make_miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.5
    cfg.drain_depth = 4
    cfg.drain_max_children = 100
    cfg.profiling_enabled = False
    cfg.masking_instructions = [
        MaskingInstruction(r"\b[\w\-\.]*\d[\w\-\.]*\b", "NUM"),
    ]
    return TemplateMiner(config=cfg)


class LogParser:
    def __init__(self) -> None:
        self.miner = _make_miner()

    def fit(self, messages) -> "LogParser":
        for m in messages:
            self.miner.add_log_message(m)
        return self

    def match(self, msg: str) -> int:
        c = self.miner.match(msg, full_search_strategy="fallback")
        return c.cluster_id if c is not None else -1

    def templates(self) -> dict[int, str]:
        return {c.cluster_id: c.get_template() for c in self.miner.drain.clusters}

    # joblib/pickle friendly
    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "LogParser":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class LogVocab:
    """Which (service, template) pairs become features."""

    def __init__(self, pairs: list[tuple[str, int]], templates: dict[int, str]):
        self.pairs = list(pairs)
        self.templates = templates
        self.pair_index = {p: i for i, p in enumerate(self.pairs)}
        self.n_pairs = len(self.pairs)
        self.names = (
            [f"log.{s}.t{t}" for s, t in self.pairs]
            + [f"log.{s}.n_{lv}" for s in COMPONENTS for lv in ("warn", "error")]
        )

    @staticmethod
    def build(service_template_counts: Counter, templates: dict[int, str], min_count: int = 40) -> "LogVocab":
        pairs = sorted(p for p, n in service_template_counts.items() if n >= min_count and p[1] >= 0)
        return LogVocab(pairs, templates)

    def step_counts(self, lines, n_steps: int) -> np.ndarray:
        """Per 5 s step counts. ``lines`` = iterable of (t, service, level, template_id)."""
        out = np.zeros((n_steps, len(self.names)), dtype=np.float32)
        base = self.n_pairs
        sidx = {s: i for i, s in enumerate(COMPONENTS)}
        for t, svc, level, tid in lines:
            i = int(t // SAMPLE_SEC)
            if not 0 <= i < n_steps:
                continue
            j = self.pair_index.get((svc, tid))
            if j is not None:
                out[i, j] += 1
            if level in ("WARN", "ERROR", "FATAL") and svc in sidx:
                k = 0 if level == "WARN" else 1
                out[i, base + sidx[svc] * 2 + k] += 1
        return out

    def step_vector(self, lines) -> np.ndarray:
        """Counts for a single 5 s step (online counterpart of :meth:`step_counts`)."""
        out = np.zeros(len(self.names), dtype=np.float32)
        base = self.n_pairs
        sidx = {s: i for i, s in enumerate(COMPONENTS)}
        for _, svc, level, tid in lines:
            j = self.pair_index.get((svc, tid))
            if j is not None:
                out[j] += 1
            if level in ("WARN", "ERROR", "FATAL") and svc in sidx:
                out[base + sidx[svc] * 2 + (0 if level == "WARN" else 1)] += 1
        return out
