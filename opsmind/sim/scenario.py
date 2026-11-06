"""Experiment campaign: schedules of 30-minute runs and their simulation.

A run mirrors one overnight experiment of the proposal: normal traffic, one
injected fault, recovery. The "injector log" (start/end of every fault) is
returned as ``truth`` and is the only source of labels.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

import numpy as np

from ..config import (COMPONENTS, FAULTS, FAULT_TYPES, RAW_COLUMNS, RUN_SEC, SAMPLE_SEC,
                      SLO_ERR_RATE, SLO_LATENCY_MS)
from ..features import ffill
from .engine import Engine

CAMPAIGN_EPOCH = 1_788_220_800.0   # 2026-09-01 00:00 UTC, purely cosmetic


@dataclass
class RunSpec:
    run_id: str
    seed: int
    kind: str                 # "normal" | "fault"
    index: int = 0            # chronological position in the campaign
    wall_start: float = 0.0
    base_load: float = 120.0
    fault: dict | None = None
    spikes: list[dict] = field(default_factory=list)
    batches: list[dict] = field(default_factory=list)


@dataclass
class RunData:
    spec: RunSpec
    metrics: np.ndarray               # (steps, len(RAW_COLUMNS)) float32, NaN = missing scrape
    logs: list[tuple[float, str, str, str]]
    truth: dict


def _pick_target(rng: random.Random, ftype: str) -> str:
    cfg = FAULTS[ftype]
    if cfg["primary"] is None or rng.random() > 0.7:
        return rng.choice(cfg["targets"])
    return cfg["primary"]


def make_campaign(n_per_fault: int = 35, n_normal: int = 44, seed: int = 7) -> list[RunSpec]:
    rng = random.Random(seed)
    specs: list[RunSpec] = []

    def spike(start_lo: float, start_hi: float, base: float) -> dict:
        factor = min(rng.uniform(1.8, 4.8), max(1.5, 480.0 / base))
        return dict(start=round(rng.uniform(start_lo, start_hi), 0), hold=round(rng.uniform(60, 240), 0),
                    factor=round(factor, 2), ramp=round(rng.uniform(15, 40), 0))

    def batches() -> list[dict]:
        out = []
        if rng.random() < 0.3:      # nightly backup: DB CPU + page-cache memory (benign, but trips static alerts)
            out.append(dict(start=round(rng.uniform(100, 1500)), duration=round(rng.uniform(60, 200)),
                            cpu=round(rng.uniform(15, 30), 1), target="postgres", mem=round(rng.uniform(25, 42), 1)))
        if rng.random() < 0.35:     # noisy neighbour blip on a random service
            out.append(dict(start=round(rng.uniform(100, 1500)), duration=round(rng.uniform(40, 120)),
                            cpu=round(rng.uniform(25, 45), 1), target=rng.choice(["gateway", "auth", "order", "payment", "redis"]),
                            mem=0.0))
        return out

    for i in range(n_normal):
        base = rng.uniform(85, 165)
        sp = [spike(60, 1500, base) for _ in range(rng.choice([0, 1, 1, 2, 3]))]
        bt = batches()
        specs.append(RunSpec(f"n{i:03d}", rng.randrange(1 << 30), "normal", base_load=base, spikes=sp, batches=bt))

    for ftype in FAULT_TYPES:
        for i in range(n_per_fault):
            base = rng.uniform(85, 165)
            start = round(rng.uniform(360, 720) / SAMPLE_SEC) * SAMPLE_SEC
            duration = round(rng.uniform(240, 540) / SAMPLE_SEC) * SAMPLE_SEC
            intensity = 0.25 + 0.75 * rng.betavariate(1.8, 1.5)
            fault = dict(type=ftype, target=_pick_target(rng, ftype), start=float(start),
                         duration=float(duration), intensity=round(intensity, 3))
            sp = []
            if rng.random() < 0.45:   # healthy load burst overlapping the fault
                sp.append(spike(start - 30, start + duration - 60, base))
            if rng.random() < 0.3:
                sp.append(spike(60, 1500, base))
            bt = batches()
            specs.append(RunSpec(f"{ftype[:3]}{i:03d}", rng.randrange(1 << 30), "fault",
                                 base_load=base, fault=fault, spikes=sp, batches=bt))

    rng.shuffle(specs)           # overnight runs are executed in random order
    for k, s in enumerate(specs):
        s.index = k
        s.wall_start = CAMPAIGN_EPOCH + k * (RUN_SEC + 120)
    return specs


def build_engine(spec: RunSpec, with_logs: bool = True) -> Engine:
    eng = Engine(seed=spec.seed, base_load=spec.base_load, with_logs=with_logs)
    for s in spec.spikes:
        eng.add_spike(s["start"], s["hold"], s["factor"], s.get("ramp"))
    for b in spec.batches:
        eng.add_batch(b["start"], b["duration"], b["cpu"], b.get("target", "postgres"), b.get("mem", 0.0))
    if spec.fault:
        f = spec.fault
        eng.add_fault(f["type"], f["target"], f["start"], f["duration"], f["intensity"])
    return eng


def find_impact(metrics: np.ndarray, fault_start: float, fault_end: float, strict: bool = False) -> float | None:
    """First time the user-facing SLO is violated (gateway error rate or p95).

    ``strict`` (used for real telemetry): the breach must *begin* at or after the fault start. A breach
    already in progress (e.g. a healthy traffic burst) is not credited to the fault.
    """
    ci = {c: i for i, c in enumerate(RAW_COLUMNS)}
    m = ffill(metrics)
    err, lat = m[:, ci["gateway.err_rate"]], m[:, ci["gateway.lat_p95"]]
    breach = (err > SLO_ERR_RATE) | (lat > SLO_LATENCY_MS)
    lo, hi = int(math.ceil(fault_start / SAMPLE_SEC)) if strict else int(fault_start // SAMPLE_SEC), int(fault_end // SAMPLE_SEC)
    for i in range(lo, min(hi, len(breach) - 1)):
        if breach[i] and breach[i + 1] and not (strict and i > 0 and breach[i - 1]):
            return float(i * SAMPLE_SEC)
    return None


def simulate_run(spec: RunSpec, with_logs: bool = True) -> RunData:
    eng = build_engine(spec, with_logs)
    n = RUN_SEC // SAMPLE_SEC
    metrics = np.empty((n, len(RAW_COLUMNS)), dtype=np.float32)
    logs: list = []
    for i in range(n):
        row, ls = eng.step()
        metrics[i] = [row[c] for c in RAW_COLUMNS]
        logs.extend(ls)
    truth: dict = dict(run_id=spec.run_id, kind=spec.kind, index=spec.index, wall_start=spec.wall_start,
                       fault_type=None, target=None, fault_start=None, fault_end=None,
                       impact_start=None, intensity=None, ramp=None,
                       spikes=spec.spikes, batches=spec.batches)
    if spec.fault:
        f = eng.faults[0]
        truth.update(fault_type=f.type, target=f.target, fault_start=f.start, fault_end=f.end,
                     intensity=f.intensity, ramp=f.ramp,
                     impact_start=find_impact(metrics, f.start, f.end))
    return RunData(spec, metrics, logs, truth)
