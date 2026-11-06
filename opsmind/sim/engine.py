"""Stateful telemetry simulator of the OpsMind e-commerce testbed.

The Docker testbed (``testbed/``) is the ground truth for the project; this
engine reproduces its behaviour step by step (one step = one 5 s Prometheus
scrape) so that datasets can be generated in seconds, the live dashboard can
inject faults interactively, and everything runs on a laptop without Docker.

Modelled mechanisms (all visible in the telemetry the ML models see):

* an exogenous request load with slow oscillation, AR(1) noise and benign
  traffic spikes (labelled *normal* on purpose),
* a synchronous call graph: throughput flows top-down, latency and errors flow
  bottom-up, timeouts convert slow callees into errors in their callers,
* queueing latency that explodes as CPU saturates,
* six fault types with gradual onset (ramp), intensity and recovery,
* cache fallback to PostgreSQL, cold cache after a Redis restart,
* structured log lines whose mix changes with the system state.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..config import CALLS, COMPONENTS, METRICS, SAMPLE_SEC

TIMEOUT_MS = 1000.0

PROFILE = {
    #            idle cpu, cpu per req/s, mem base %, p95 base ms
    "gateway": dict(idle=6.0, cost=0.16, mem=24.0, lat=9.0),
    "auth": dict(idle=4.0, cost=0.15, mem=30.0, lat=18.0),
    "order": dict(idle=6.0, cost=0.40, mem=38.0, lat=40.0),
    "payment": dict(idle=5.0, cost=0.90, mem=33.0, lat=65.0),
    "postgres": dict(idle=8.0, cost=0.12, mem=55.0, lat=6.0),
    "redis": dict(idle=2.0, cost=0.012, mem=22.0, lat=1.0),
}

TOP_DOWN = ["gateway", "auth", "order", "payment", "redis", "postgres"]
BOTTOM_UP = ["postgres", "redis", "payment", "auth", "order", "gateway"]


# ----------------------------------------------------------------------------- events
@dataclass
class Fault:
    fid: str
    type: str
    target: str
    start: float
    duration: float
    intensity: float
    ramp: float
    recover: float = 45.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    def severity(self, t: float) -> float:
        if t < self.start:
            return 0.0
        if t < self.start + self.ramp:
            return (t - self.start) / max(self.ramp, 1e-6)
        if t < self.end:
            return 1.0
        if t < self.end + self.recover:
            return 1.0 - (t - self.end) / self.recover
        return 0.0

    def active(self, t: float) -> bool:
        return self.start <= t < self.end

    def outage(self, t: float) -> bool:
        return self.start + self.ramp <= t < self.end

    def to_dict(self) -> dict:
        return dict(fid=self.fid, type=self.type, target=self.target, start=self.start,
                    end=self.end, intensity=round(self.intensity, 3), ramp=round(self.ramp, 1))


@dataclass
class Spike:
    start: float
    ramp: float
    hold: float
    factor: float

    def multiplier(self, t: float) -> float:
        if t < self.start:
            return 1.0
        dt = t - self.start
        if dt < self.ramp:
            return 1 + (self.factor - 1) * dt / self.ramp
        if dt < self.ramp + self.hold:
            return self.factor
        dt -= self.ramp + self.hold
        if dt < self.ramp:
            return self.factor - (self.factor - 1) * dt / self.ramp
        return 1.0


@dataclass
class Batch:
    """Benign background job (nightly backup / report) that raises DB CPU."""
    start: float
    duration: float
    cpu: float
    target: str = "postgres"
    mem: float = 0.0

    def level(self, t: float) -> float:
        if self.start <= t < self.start + self.duration:
            return self.cpu
        return 0.0

    def mem_level(self, t: float) -> float:
        """Page-cache style memory growth that lingers briefly after the job ends."""
        if self.start <= t < self.start + self.duration:
            return self.mem * min(1.0, (t - self.start) / 40.0)
        if self.start + self.duration <= t < self.start + self.duration + 60:
            return self.mem * (1 - (t - self.start - self.duration) / 60.0)
        return 0.0


# ------------------------------------------------------------------------ helpers
class AR1:
    def __init__(self, rng: random.Random, phi: float, sigma: float):
        self.rng, self.phi, self.sigma, self.x = rng, phi, sigma, 0.0

    def step(self) -> float:
        self.x = self.phi * self.x + self.rng.gauss(0.0, self.sigma)
        return self.x


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# -------------------------------------------------------------------- log catalogue
INFO_LINES = {
    "gateway": ["GET /api/products/{id} 200 {ms}ms", "POST /api/orders 201 {ms}ms",
                "POST /auth/login 200 {ms}ms", "GET /api/cart/{id} 200 {ms}ms"],
    "auth": ["token issued user={uid} ttl=3600", "session cache hit user={uid}",
             "password hash verified in {ms}ms"],
    "order": ["order {oid} created total={amt}", "cart {cid} loaded items={n}",
              "order {oid} payment authorised"],
    "payment": ["payment {pid} authorised amount={amt}", "charge captured txn={txn}",
                "ledger entry written txn={txn}"],
    "postgres": ["checkpoint complete: wrote {n} buffers", "connection received host=10.0.0.{n}",
                 "statement: SELECT * FROM orders WHERE id = {id}"],
    "redis": ["client connected 10.0.0.{n}", "Background saving started by pid {pid}",
              "DB saved on disk keys={n}"],
}


def _fmt(rng: random.Random, tpl: str, **kw) -> str:
    vals = dict(
        id=rng.randint(1, 99999), ms=rng.randint(2, 60), uid=rng.randint(1000, 9999),
        oid=f"o-{rng.randint(10000, 99999)}", cid=f"c-{rng.randint(1000, 9999)}",
        pid=rng.randint(100, 9999), txn=f"t{rng.randint(100000, 999999)}",
        amt=round(rng.uniform(5, 400), 2), n=rng.randint(2, 250), mb=rng.randint(16, 512),
    )
    vals.update(kw)
    return tpl.format(**vals)


# ------------------------------------------------------------------------- engine
class Engine:
    """One simulated deployment of the testbed. Call :meth:`step` once per scrape."""

    def __init__(self, seed: int = 0, base_load: float | None = None, with_logs: bool = True):
        self.rng = rng = random.Random(seed)
        self.t = 0.0
        self.with_logs = with_logs
        self.base_load = base_load if base_load is not None else rng.uniform(85, 165)
        self.period = rng.uniform(1500, 6000)
        self.phase = rng.random()
        self.faults: list[Fault] = []
        self.spikes: list[Spike] = []
        self.batches: list[Batch] = []
        self._fid = 0

        self.load_noise = AR1(rng, 0.92, 0.035)
        self.cpu_noise = {c: AR1(rng, 0.85, 1.4) for c in COMPONENTS}
        self.mem_noise = {c: AR1(rng, 0.98, 0.25) for c in COMPONENTS}
        self.lat_noise = {c: AR1(rng, 0.7, 0.10) for c in COMPONENTS}
        # every deployment differs a little (instance sizes, data volume, versions)
        self.mem_base = {c: PROFILE[c]["mem"] * rng.uniform(0.78, 1.3) for c in COMPONENTS}
        self.prof = {c: {k: v * (rng.uniform(0.8, 1.25) if k in ("cost", "idle") else
                                 rng.uniform(0.85, 1.3) if k == "lat" else 1.0)
                         for k, v in PROFILE[c].items()} for c in COMPONENTS}
        self.restart_t: dict[str, float] = {}
        self.was_up = {c: True for c in COMPONENTS}
        self.last: dict[str, dict] = {}

    # -- scheduling -------------------------------------------------------------
    def add_fault(self, ftype: str, target: str, start: float, duration: float,
                  intensity: float, ramp: float | None = None) -> Fault:
        rng = self.rng
        if ramp is None:
            ramp = {
                "cpu_saturation": rng.uniform(40, 240),
                "memory_leak": rng.uniform(150, 420),
                "db_slowdown": rng.uniform(30, 180),
                "network_latency": rng.uniform(20, 130),
                "service_crash": 0.0 if rng.random() < 0.35 else rng.uniform(40, 160),
                "cache_failure": rng.uniform(0, 15),
            }[ftype]
        self._fid += 1
        f = Fault(f"F{self._fid}", ftype, target, start, duration, intensity, ramp)
        self.faults.append(f)
        return f

    def add_spike(self, start: float, hold: float, factor: float, ramp: float | None = None) -> Spike:
        s = Spike(start, ramp if ramp is not None else self.rng.uniform(15, 40), hold, factor)
        self.spikes.append(s)
        return s

    def add_batch(self, start: float, duration: float, cpu: float, target: str = "postgres",
                  mem: float = 0.0) -> Batch:
        b = Batch(start, duration, cpu, target, mem)
        self.batches.append(b)
        return b

    def clear_faults(self, now: float | None = None) -> None:
        """End every active fault immediately (operator pressed 'stop')."""
        now = self.t if now is None else now
        for f in self.faults:
            if f.start <= now < f.end:
                f.duration = max(now - f.start, 1.0)

    # -- helpers ------------------------------------------------------------------
    def _poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= L:
                return k
            k += 1
            if k > 40:
                return k

    def active_faults(self, t: float | None = None) -> list[Fault]:
        t = self.t if t is None else t
        return [f for f in self.faults if f.active(t)]

    # -- main step ----------------------------------------------------------------
    def step(self) -> tuple[dict[str, float], list[tuple[float, str, str, str]]]:
        rng, t = self.rng, self.t
        P = self.prof

        spike = 1.0
        for s in self.spikes:
            spike = max(spike, s.multiplier(t))
        diurnal = 1.0 + 0.22 * math.sin(2 * math.pi * (t / self.period + self.phase))
        load = max(5.0, self.base_load * diurnal * (1.0 + self.load_noise.step()) * spike)

        # ---- fault effects ----------------------------------------------------------
        up = {c: True for c in COMPONENTS}
        cpu_add = {c: 0.0 for c in COMPONENTS}
        mem_add = {c: 0.0 for c in COMPONENTS}
        lat_mult = {c: 1.0 for c in COMPONENTS}
        lat_add = {c: 0.0 for c in COMPONENTS}
        err_add = {c: 0.0 for c in COMPONENTS}
        drop = {c: 0.0 for c in COMPONENTS}
        retrans_add = {c: 0.0 for c in COMPONENTS}
        locks_add = conns_add = 0.0
        cache_warm = 1.0

        for f in self.faults:
            s, I, c = f.severity(t), f.intensity, f.target
            if f.type == "cpu_saturation" and s > 0:
                cpu_add[c] += s * (28 + 68 * I)
            elif f.type == "memory_leak" and s > 0:
                mem_add[c] += s * (14 + 52 * I)
            elif f.type == "db_slowdown" and s > 0:
                lat_mult["postgres"] *= 1 + s * (4 + 60 * I * I)
                locks_add += s * I * (10 + 30 * I)
                conns_add += s * I * (40 + 80 * I)
                cpu_add["postgres"] += s * I * 18
            elif f.type == "network_latency" and s > 0:
                lat_add[c] += s * (60 + 900 * I)
                retrans_add[c] += s * I * (10 + 50 * I)
                drop[c] += s * I * 0.06
                err_add[c] += s * I * 4.8
            elif f.type in ("service_crash", "cache_failure"):
                if f.outage(t):
                    up[c] = False
                elif f.start <= t < f.start + f.ramp and rng.random() < 0.7 * s:
                    up[c] = False        # crash-loop prelude / flapping
                if t >= f.end:
                    self.restart_t[c] = max(self.restart_t.get(c, -1e9), f.end)
                    if f.type == "cache_failure":
                        cache_warm = min(cache_warm, min(1.0, (t - f.end) / 150.0))
                elif f.type == "cache_failure" and not up[c]:
                    cache_warm = 0.0

        for b in self.batches:
            cpu_add[b.target] += b.level(t)
            mem_add[b.target] += b.mem_level(t)

        # OOM-kill when memory is exhausted
        for c in COMPONENTS:
            m = self.last.get(c, {}).get("mem", 0.0)
            if m > 96.0 and rng.random() < 0.4:
                up[c] = False

        # ---- throughput, top-down -----------------------------------------------------
        attempts = {c: 0.0 for c in COMPONENTS}
        served = {c: 0.0 for c in COMPONENTS}
        attempts["gateway"] = load
        redis_attempt = 0.0
        for c in TOP_DOWN:
            if c == "postgres":
                miss_extra = redis_attempt * max(0.0, (1.0 - 0.93 * cache_warm) - 0.07) * 1.0
                attempts[c] += miss_extra
            served[c] = attempts[c] * (1 - min(drop[c], 0.9)) if up[c] else 0.0
            for callee, fan in CALLS.get(c, {}).items():
                attempts[callee] += fan * served[c]
                if callee == "redis":
                    redis_attempt += fan * served[c]

        # ---- cpu / mem / misc ---------------------------------------------------------
        st: dict[str, dict] = {}
        for c in COMPONENTS:
            p = P[c]
            since = t - self.restart_t.get(c, -1e9)
            boost = 28.0 * math.exp(-since / 25.0) if 0 <= since < 120 else 0.0
            if up[c]:
                cpu = (p["idle"] + p["cost"] * served[c] + self.cpu_noise[c].step()
                       + rng.gauss(0, 0.5) + cpu_add[c] + boost)
                mem = (self.mem_base[c] + 0.012 * served[c] + self.mem_noise[c].step()
                       + mem_add[c])
                if 0 <= since < 90:
                    mem *= 0.65 + 0.35 * (1 - math.exp(-since / 35.0))
                # garbage-collection pressure when memory runs hot
                if mem > 75:
                    gc = (mem - 75.0)
                    lat_mult[c] *= 1 + gc / 6.0
                    cpu += gc * 0.9
                    if mem > 90:
                        err_add[c] += (mem - 90.0) * 1.8   # allocation failures
            else:
                cpu = mem = 0.0
            cpu = min(max(cpu, 0.0), 100.0)
            mem = min(max(mem, 0.0), 100.0)
            st[c] = dict(up=up[c], cpu=cpu, mem=mem, served=served[c])

        # ---- latency & errors, bottom-up ---------------------------------------------
        for c in BOTTOM_UP:
            p, s = P[c], st[c]
            timeouts: dict[str, float] = {}
            refused: list[str] = []
            if not s["up"]:
                s.update(lat=0.0, err=100.0, own_lat=0.0, timeouts=timeouts, refused=refused)
                continue
            rho = min(s["cpu"] / 100.0, 0.985)
            q = min(1.0 + 0.45 * rho * rho / (1.0 - rho), 30.0)
            own = p["lat"] * q * lat_mult[c] * math.exp(self.lat_noise[c].step()) + lat_add[c]
            err = 0.12 + abs(rng.gauss(0, 0.06)) + err_add[c]
            if c == "postgres":
                conns = 6 + 0.08 * served[c] + conns_add
                locks = max(0.0, 0.4 + abs(rng.gauss(0, 0.5)) + locks_add)
                if conns > 95:
                    err += (conns - 95) * 0.25
                if locks > 15:
                    err += (locks - 15) * 0.2
                s.update(conns=conns, locks=locks)
            if s["cpu"] > 97:
                err += (s["cpu"] - 97) * 1.2
            lat, e_acc = own, err
            for callee, fan in CALLS.get(c, {}).items():
                w = min(fan, 1.0)
                cs = st[callee]
                if not cs["up"]:
                    refused.append(callee)
                    if callee == "redis":              # graceful fallback to the database
                        lat += w * (st["postgres"]["lat"] * 1.5 + 25.0)
                        e_acc += w * 6.0
                    else:
                        lat += w * 3.0                 # connection refused fails fast
                        e_acc += w * 100.0
                else:
                    p_to = _sigmoid((cs["lat"] - 1.15 * TIMEOUT_MS) / (0.11 * TIMEOUT_MS))
                    timeouts[callee] = p_to
                    lat += w * cs["lat"]
                    e_acc += w * (cs["err"] * 0.9 + p_to * 100.0)
            s.update(lat=min(lat, 3 * TIMEOUT_MS), own_lat=own, err=min(e_acc, 100.0),
                     timeouts=timeouts, refused=refused)

        # ---- emit metrics ------------------------------------------------------------
        row: dict[str, float] = {}
        for c in COMPONENTS:
            s = st[c]
            alive = s["up"]
            row[f"{c}.cpu"] = round(s["cpu"], 3)
            row[f"{c}.mem"] = round(s["mem"], 3)
            row[f"{c}.rps"] = round(max(0.0, s["served"] + rng.gauss(0, math.sqrt(s["served"] + 1))), 3) if alive else 0.0
            row[f"{c}.lat_p95"] = round(s["lat"] * math.exp(rng.gauss(0, 0.08)), 3) if alive else 0.0
            row[f"{c}.err_rate"] = round(min(100.0, s["err"] * math.exp(rng.gauss(0, 0.08))), 3) if alive else 0.0
            row[f"{c}.up"] = 1.0 if alive else 0.0
            if "net_retrans" in METRICS[c]:
                row[f"{c}.net_retrans"] = round(
                    (0.3 + 0.004 * s["served"] + abs(rng.gauss(0, 0.25)) + retrans_add[c]) if alive else 0.0, 3)
        pg = st["postgres"]
        row["postgres.conns"] = round(pg.get("conns", 0.0) + rng.gauss(0, 0.8), 3) if pg["up"] else 0.0
        row["postgres.locks"] = round(pg.get("locks", 0.0), 3) if pg["up"] else 0.0
        hit = (0.93 * cache_warm + rng.gauss(0, 0.012)) if st["redis"]["up"] else 0.0
        row["redis.hit_ratio"] = round(min(max(hit, 0.0), 1.0), 4)

        # missing scrapes (≈0.4 %) show up as NaN and are cleaned in preprocessing
        for k in list(row):
            if not k.endswith(".up") and rng.random() < 0.004:
                row[k] = float("nan")

        logs = self._emit_logs(t, st, cache_warm) if self.with_logs else []
        for c in COMPONENTS:
            self.was_up[c] = st[c]["up"]
        self.last = st
        self.t += SAMPLE_SEC
        return row, logs

    # -- logs --------------------------------------------------------------------------
    def _emit_logs(self, t: float, st: dict, cache_warm: float) -> list[tuple[float, str, str, str]]:
        rng, out = self.rng, []

        def add(svc: str, level: str, msg: str) -> None:
            out.append((t + rng.random() * SAMPLE_SEC * 0.99, svc, level, msg))

        for c in COMPONENTS:
            s = st[c]
            if not s["up"]:
                continue
            # routine (sampled) access / housekeeping logs
            for _ in range(min(3, self._poisson(0.55 + 0.004 * s["served"]))):
                add(c, "INFO", _fmt(rng, rng.choice(INFO_LINES[c])))
            if not self.was_up[c]:
                add(c, "INFO", "Ready to accept connections" if c in ("redis", "postgres")
                    else _fmt(rng, "service started version=1.4.2 port=8000 pid={pid}"))

            # generic symptoms
            if s["cpu"] > 85 and rng.random() < 0.6:
                add(c, "WARN", _fmt(rng, "cpu throttling detected usage={n}%", n=int(s["cpu"])))
            if s["mem"] > 82 and rng.random() < 0.7:
                add(c, "WARN", _fmt(rng, "GC pause {ms}ms heap={n}%", ms=int((s["mem"] - 70) * 12), n=int(s["mem"])))
            if s["mem"] > 92 and rng.random() < 0.6:
                add(c, "ERROR", _fmt(rng, "MemoryError: unable to allocate {mb} MB"))
            if s["lat"] > 5 * self.prof[c]["lat"] + 150 and rng.random() < 0.7:
                add(c, "WARN", _fmt(rng, "slow request {ms}ms exceeded slo", ms=int(s["lat"])))
            for _ in range(min(3, self._poisson(min(3.0, s["err"] * 0.08)))):
                add(c, "ERROR", _fmt(rng, rng.choice(["request failed status=500 id={id}",
                                                       "unhandled exception while serving request id={id}"])))

            # dependency-specific symptoms
            for callee, p_to in s["timeouts"].items():
                if p_to > 0.05 and rng.random() < min(1.0, 0.3 + 3 * p_to):
                    add(c, "ERROR", _fmt(rng, "upstream {svc} timed out after 1000ms", svc=callee))
                    if c == "order" and callee == "payment":
                        add(c, "WARN", _fmt(rng, "retrying payment request attempt={n}", n=rng.randint(1, 3)))
                        if rng.random() < 0.4:
                            add(c, "WARN", "circuit breaker open for payment")
            for callee in s["refused"]:
                if callee == "redis":
                    if rng.random() < 0.8:
                        add(c, "WARN", "redis connection refused falling back to postgres")
                else:
                    add(c, "ERROR", _fmt(rng, "connection refused to {svc}:8000", svc=callee))
                    if c == "gateway":
                        add(c, "ERROR", _fmt(rng, "502 bad gateway upstream {svc}", svc=callee))

            if c == "postgres":
                if s["lat"] > 40 and rng.random() < 0.8:
                    add(c, "WARN", _fmt(rng, "duration: {ms} ms slow statement", ms=int(s["lat"] * 10)))
                if s.get("locks", 0) > 8 and rng.random() < 0.8:
                    add(c, "ERROR", _fmt(rng, "process {pid} still waiting for ShareLock on transaction {n}"))
                    if s["locks"] > 18 and rng.random() < 0.5:
                        add(c, "ERROR", "canceling statement due to lock timeout")
                if s.get("conns", 0) > 90 and rng.random() < 0.7:
                    add(c, "FATAL", "remaining connection slots are reserved for non-replication superuser connections")
        # network faults leave fingerprints in the logs of their target
        for f in self.faults:
            if f.type == "network_latency" and f.severity(t) > 0.15 and st[f.target]["up"]:
                if rng.random() < 0.9:
                    add(f.target, "WARN", _fmt(rng, "tcp retransmits detected count={n}", n=rng.randint(5, 80)))
                if rng.random() < 0.5 * f.intensity + 0.1:
                    add(f.target, "ERROR", _fmt(rng, "connect timeout to bank-gateway after {ms}ms", ms=rng.randint(800, 3000)))
        return out
