"""Telemetry sources feeding the runtime: live simulator (with fault injection) and replay of saved runs.

A Prometheus-backed source for the real Docker testbed lives in ``testbed/collector``;
it yields the same ``(t, row, logs, truth)`` tuples.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from .config import DATA_DIR, FAULTS, RAW_COLUMNS, RUN_SEC, SAMPLE_SEC
from .sim.engine import Engine


REAL_PREFIXES = ("real", "rb")                     # campaign 1 (abrupt faults) and campaign 2 (ramped faults)
REAL_DIRS = [DATA_DIR / "real", DATA_DIR / "real2"]


def is_real(run_id: str) -> bool:
    return run_id.startswith(REAL_PREFIXES)


def real_dir(run_id: str):
    for d in REAL_DIRS:
        p = d / "runs.json"
        if p.exists() and any(r["run_id"] == run_id for r in json.loads(p.read_text())):
            return d
    raise KeyError(run_id)


class SimulatorSource:
    """Endless simulated deployment; faults can be injected at any moment."""

    kind = "simulator"

    def __init__(self, seed: int | None = None):
        seed = random.randrange(1 << 30) if seed is None else seed
        self.engine = Engine(seed=seed)
        self.seed = seed

    @property
    def now(self) -> float:
        return self.engine.t

    def step(self):
        t = self.engine.t
        row, logs = self.engine.step()
        return t, row, logs, self._truth(t)

    def _truth(self, t: float) -> dict:
        e = self.engine
        faults = [dict(f.to_dict(), severity=round(f.severity(t), 3)) for f in e.faults if f.severity(t) > 0 or f.start - 30 <= t < f.start]
        spike = max([s.multiplier(t) for s in e.spikes] or [1.0])
        return dict(faults=[f for f in faults if f["severity"] > 0], spike=round(spike, 2),
                    batch=any(b.level(t) > 0 for b in e.batches))

    # -- operator actions -------------------------------------------------------------------------------
    def inject(self, ftype: str, target: str | None, duration: float, intensity: float) -> dict:
        cfg = FAULTS[ftype]
        target = target or cfg["primary"] or random.choice(cfg["targets"])
        if target not in cfg["targets"]:
            raise ValueError(f"{ftype} cannot target {target}; valid: {cfg['targets']}")
        f = self.engine.add_fault(ftype, target, self.engine.t + SAMPLE_SEC, duration, intensity)
        return f.to_dict()

    def spike(self, factor: float, hold: float) -> dict:
        s = self.engine.add_spike(self.engine.t + SAMPLE_SEC, hold, factor)
        return dict(start=s.start, hold=s.hold, factor=s.factor)

    def clear(self) -> None:
        self.engine.clear_faults()
        for s in self.engine.spikes:
            if s.start <= self.engine.t < s.start + 2 * s.ramp + s.hold:
                s.hold = max(0.0, self.engine.t - s.start - s.ramp)


class ReplaySource:
    """Streams a saved experiment run (the 'replay mode' that makes the demo robust)."""

    kind = "replay"

    def __init__(self, run_id: str):
        self.run_id = run_id
        base = real_dir(run_id) if is_real(run_id) else DATA_DIR                   # real = Docker testbed runs
        m = pd.read_parquet(base / "raw_metrics.parquet", filters=[("run_id", "==", run_id)])
        l = pd.read_parquet(base / "raw_logs.parquet", filters=[("run_id", "==", run_id)])
        if m.empty:
            raise KeyError(run_id)
        self.metrics = m.sort_values("t")[RAW_COLUMNS].to_numpy(np.float32)
        self.logs = [[] for _ in range(len(self.metrics))]
        for t, svc, lv, msg in zip(l["t"], l["service"], l["level"], l["msg"]):
            i = int(t // SAMPLE_SEC)
            if 0 <= i < len(self.logs):
                self.logs[i].append((float(t), svc, lv, msg))
        with open(base / "runs.json") as fh:
            self.truth_run = next(r for r in json.load(fh) if r["run_id"] == run_id)
        self.i = 0

    @property
    def now(self) -> float:
        return self.i * SAMPLE_SEC

    @property
    def done(self) -> bool:
        return self.i >= len(self.metrics)

    def step(self):
        if self.done:
            return None
        t = self.i * SAMPLE_SEC
        row = dict(zip(RAW_COLUMNS, self.metrics[self.i].tolist()))
        logs = self.logs[self.i]
        self.i += 1
        tr = self.truth_run
        faults = []
        if tr["fault_type"] and tr["fault_start"] <= t < tr["fault_end"]:
            faults.append(dict(type=tr["fault_type"], target=tr["target"], intensity=tr["intensity"], severity=1.0))
        spike = 1.0
        for s in tr.get("spikes", []):
            if s["start"] <= t < s["start"] + 2 * s.get("ramp", 20) + s["hold"]:
                spike = max(spike, s["factor"])
        return t, row, logs, dict(faults=faults, spike=spike, batch=False)
