"""Live session: paces a telemetry source through the OpsMind runtime and fans results out to clients."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque

import numpy as np

from opsmind.config import ARTIFACT_DIR, COMPONENT_LABEL, FAULTS
from opsmind.naming import fault_label
from opsmind.runtime import OpsMindRuntime, StreamState
from opsmind.sources import ReplaySource, SimulatorSource, is_real

PREWARM_TICKS = 90          # 15 simulated minutes of healthy history before the first client connects

SCENARIOS = {
    "healthy_surge": dict(
        title="Healthy traffic surge",
        description="A 4× Locust burst. Heavy but healthy: static CPU alerts may fire, OpsMind should not.",
        steps=[(0, "spike", dict(factor=4.2, hold=170))]),
    "db_incident": dict(
        title="Database slowdown",
        description="Slow queries and lock contention on PostgreSQL ripple up to every caller.",
        steps=[(0, "fault", dict(type="db_slowdown", target="postgres", duration=360, intensity=0.85))]),
    "memory_leak": dict(
        title="Memory leak in Order service",
        description="A leaking endpoint: slow ramp, GC thrash, finally errors. The early-warning case.",
        steps=[(0, "fault", dict(type="memory_leak", target="order", duration=420, intensity=0.85))]),
    "payment_network": dict(
        title="Payment network degradation",
        description="tc netem latency and packet loss on the payment service.",
        steps=[(0, "fault", dict(type="network_latency", target="payment", duration=360, intensity=0.8))]),
    "auth_crash": dict(
        title="Auth service crash",
        description="docker kill on auth: instant outage with a crash-loop prelude.",
        steps=[(0, "fault", dict(type="service_crash", target="auth", duration=240, intensity=1.0))]),
    "cache_outage": dict(
        title="Redis outage",
        description="Cache stops; callers fall back to PostgreSQL and the database feels it.",
        steps=[(0, "fault", dict(type="cache_failure", target="redis", duration=240, intensity=1.0))]),
    "chaos_playlist": dict(
        title="Chaos playlist (≈ 25 min)",
        description="Surge → network → crash → DB slowdown. Hands-free demo of the whole pipeline.",
        steps=[(0, "spike", dict(factor=3.8, hold=140)),
               (30, "fault", dict(type="network_latency", target="payment", duration=300, intensity=0.8)),
               (95, "fault", dict(type="service_crash", target="auth", duration=200, intensity=1.0)),
               (150, "fault", dict(type="db_slowdown", target="postgres", duration=300, intensity=0.85))]),
}


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, default=_default, separators=(",", ":"))


class LiveSession:
    def __init__(self, runtime: OpsMindRuntime):
        self.rt = runtime
        self.sim_rt = runtime
        self.real_rt: OpsMindRuntime | None = None
        self.lock = threading.RLock()
        self.clients: set = set()
        self.history: deque = deque(maxlen=720)       # 2 simulated hours
        self.paused = False
        self.speed = 1.0                               # ticks per real second (1 tick = 10 simulated seconds)
        self.mode = "sim"
        self.replay_done = False
        self.script: list[tuple[int, str, dict]] = []
        self.tick_no = 0
        self.scenario: str | None = None
        self.source = None
        self.state: StreamState = runtime.new_state()
        self.prev: dict | None = None
        self.stats = dict(ticks=0, avg_ms=0.0, max_ms=0.0)
        self._pending: list[dict] = []
        self.reset(prewarm=True)

    # -- lifecycle ---------------------------------------------------------------------------------
    def reset(self, prewarm: bool = True) -> None:
        with self.lock:
            self.rt = self.sim_rt
            self.mode, self.replay_done, self.paused = "sim", False, False
            self.source = SimulatorSource()
            self.state = self.rt.new_state()
            self.history.clear()
            self.script, self.scenario, self.tick_no, self.prev = [], None, 0, None
            if prewarm:
                for _ in range(PREWARM_TICKS):
                    self._advance()

    def runtime_for(self, run_id: str) -> OpsMindRuntime:
        """Real runs are scored by the models trained on real data when they exist."""
        real_art = ARTIFACT_DIR / "real"
        if is_real(run_id) and (real_art / "bundle.joblib").exists():
            if self.real_rt is None or (real_art / "bundle.joblib").stat().st_mtime > getattr(self, "_real_mtime", 0):
                self.real_rt = OpsMindRuntime(real_art)
                self._real_mtime = (real_art / "bundle.joblib").stat().st_mtime
            return self.real_rt
        return self.sim_rt

    def start_replay(self, run_id: str) -> None:
        with self.lock:
            self.source = ReplaySource(run_id)
            self.rt = self.runtime_for(run_id)
            self.state = self.rt.new_state()
            self.history.clear()
            self.mode, self.replay_done, self.paused, self.prev = "replay", False, False, None
            self.script, self.scenario, self.tick_no = [], None, 0

    def status(self) -> dict:
        src = self.source
        info = dict(kind=src.kind)
        if self.mode == "replay":
            info.update(run_id=src.run_id, truth=_replay_truth(src.truth_run), progress=round(src.i / len(src.metrics), 3))
        return dict(type="status", models="real" if self.rt is not self.sim_rt else "simulator", mode=self.mode, paused=self.paused, speed=self.speed, source=info,
                    replay_done=self.replay_done, scenario=self.scenario, stats=self.stats,
                    scenarios=[dict(id=k, title=v["title"], description=v["description"]) for k, v in SCENARIOS.items()],
                    clients=len(self.clients))

    # -- operator actions -----------------------------------------------------------------------------
    def inject(self, ftype: str, target: str | None, duration: float, intensity: float) -> dict:
        with self.lock:
            if self.mode != "sim":
                raise ValueError("fault injection is only available in live-simulator mode")
            f = self.source.inject(ftype, target, duration, intensity)
            self._emit_event(dict(t=self.source.now, kind="injected", severity="info",
                                  title=f"Injected {fault_label(ftype)}",
                                  detail=f"{COMPONENT_LABEL[f['target']]} · intensity {intensity:.2f} · {int(duration)} s"))
            return f

    def spike(self, factor: float, hold: float) -> dict:
        with self.lock:
            if self.mode != "sim":
                raise ValueError("traffic bursts are only available in live-simulator mode")
            s = self.source.spike(factor, hold)
            self._emit_event(dict(t=self.source.now, kind="injected", severity="info",
                                  title="Traffic burst started", detail=f"{factor:.1f}× load for {int(hold)} s (healthy)"))
            return s

    def clear(self) -> None:
        with self.lock:
            if self.mode == "sim":
                self.source.clear()
                self.script.clear()
                self._emit_event(dict(t=self.source.now, kind="cleared", severity="info",
                                      title="Faults cleared", detail="Operator ended all injections"))

    def run_scenario(self, name: str) -> None:
        sc = SCENARIOS[name]
        with self.lock:
            if self.mode != "sim":
                raise ValueError("scenarios need the live simulator")
            self.scenario = name
            base = self.tick_no
            self.script = [(base + at, kind, params) for at, kind, params in sc["steps"]]

    # -- stepping ------------------------------------------------------------------------------------------
    def _run_script(self) -> None:
        due = [s for s in self.script if s[0] <= self.tick_no]
        for s in due:
            self.script.remove(s)
            _, kind, p = s
            if kind == "fault":
                self.inject(p["type"], p.get("target"), p["duration"], p["intensity"])
            else:
                self.spike(p["factor"], p["hold"])

    def _advance(self) -> dict | None:
        """Advance one 10 s window; returns the tick (or None when a replay has ended)."""
        with self.lock:
            if self.mode == "sim":
                self._run_script()
            tick = None
            for _ in range(2):
                s = self.source.step()
                if s is None:
                    self.replay_done = True
                    return None
                t, row, logs, truth = s
                t0 = time.perf_counter()
                out = self.rt.ingest(self.state, t, row, logs, truth)
                if out is not None:
                    dt = (time.perf_counter() - t0) * 1000
                    self.stats["ticks"] += 1
                    self.stats["avg_ms"] = round(self.stats["avg_ms"] * 0.95 + dt * 0.05, 2)
                    self.stats["max_ms"] = round(max(self.stats["max_ms"] * 0.99, dt), 2)
                    tick = out
            if tick is None:
                return None
            self.tick_no += 1
            tick["events"] = self._derive_events(self.prev, tick)
            tick["i"] = self.tick_no
            self.prev = tick
            self.history.append(tick)
            return tick

    def _emit_event(self, ev: dict) -> None:
        self._pending.append(ev)

    def _derive_events(self, prev: dict | None, k: dict) -> list[dict]:
        evs = list(self._pending)
        self._pending = []
        t = k["t"]
        was = lambda *path: _get(prev, *path) if prev else None
        if k["risk"]["alert"] and not was("risk", "alert"):
            evs.append(dict(t=t, kind="early_warning", severity="warn", title="Early warning: incident likely",
                            detail=f"Incident probability {k['risk']['prob'] * 100:.0f}% within 2 min"))
        if k["anomaly"]["sustained"] and not was("anomaly", "sustained"):
            top = k["rca"][0] if k.get("rca") else None
            evs.append(dict(t=t, kind="detected", severity="critical", title=f"Anomaly detected · {k['classifier']['likely_fault_name']}",
                            detail=(f"Most likely cause: {COMPONENT_LABEL[top['component']]}" if top else "Investigating")))
        if k["baseline"]["alert"] and not was("baseline", "alert"):
            f = k["baseline"]["firing"][0]
            evs.append(dict(t=t, kind="threshold_alert", severity="info", title=f"Static alert: {f['rule']}",
                            detail=f"{COMPONENT_LABEL[f['component']]} = {f['value']}"))
        if k["slo"]["impacted"] and not was("slo", "impacted"):
            evs.append(dict(t=t, kind="impact", severity="critical", title="User-facing SLO breached",
                            detail=f"Gateway p95 {k['slo']['lat']:.0f} ms · errors {k['slo']['err']:.1f}%"))
        if k["pattern"]["novel"] and not was("pattern", "novel"):
            evs.append(dict(t=t, kind="novel_pattern", severity="warn", title="Unseen failure pattern",
                            detail="Anomaly does not match any known cluster"))
        cur_prev, cur = _get(prev, "incident", "current") if prev else None, k["incident"]["current"]
        if cur_prev and not cur:
            hist = k["incident"]["history"]
            if hist:
                h = hist[0]
                evs.append(dict(t=t, kind="resolved", severity="info", title=f"{h['id']} resolved",
                                detail=f"{h.get('type_name') or 'Incident'} · lasted {int(h['resolved_at'] - h['started_at'])} s"))
        return evs

    async def run(self) -> None:
        while True:
            if self.paused or (self.mode == "replay" and self.replay_done):
                await asyncio.sleep(0.15)
                continue
            t0 = time.perf_counter()
            tick = await asyncio.to_thread(self._advance)
            if tick is not None:
                await self.broadcast(dumps(tick))
            elif self.replay_done:
                await self.broadcast(dumps(self.status()))
            await asyncio.sleep(max(0.0, 1.0 / self.speed - (time.perf_counter() - t0)))

    async def broadcast(self, payload: str) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def snapshot(self, n: int = 120) -> list[dict]:
        with self.lock:
            return list(self.history)[-n:]


def _get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def _replay_truth(tr: dict) -> dict:
    return dict(run_id=tr["run_id"], fault_type=tr["fault_type"], fault_label=fault_label(tr["fault_type"]) if tr["fault_type"] else "Normal",
                target=tr["target"], intensity=tr["intensity"], fault_start=tr["fault_start"], fault_end=tr["fault_end"],
                impact_start=tr["impact_start"])
