"""OpsMind AI — FastAPI service.

Serves the trained models (REST + WebSocket), a fault-injection console for the live
simulator, replay of saved experiment runs, and the built dashboard.

    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from opsmind import __version__
from opsmind.config import (ARTIFACT_DIR, CALLS, CLASSES, COMPONENT_KIND, COMPONENT_LABEL, COMPONENTS, DATA_DIR, EDGES,
                            FAULTS, METRIC_INFO, METRICS, RAW_COLUMNS, SLO_ERR_RATE, SLO_LATENCY_MS, THRESHOLD_RULES)
from opsmind.runtime import OpsMindRuntime

from .session import SCENARIOS, LiveSession, dumps

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dashboard" / "dist"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    rt = await asyncio.to_thread(OpsMindRuntime)
    session = await asyncio.to_thread(LiveSession, rt)
    state.update(rt=rt, session=session)
    state["task"] = asyncio.create_task(session.run())
    yield
    state["task"].cancel()


app = FastAPI(title="OpsMind AI", version=__version__, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def S() -> LiveSession:
    return state["session"]


# ------------------------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    ok = "session" in state
    return dict(status="ok" if ok else "loading", version=__version__,
                models_loaded=ok, ticks=S().stats["ticks"] if ok else 0)


@app.get("/api/config")
def config():
    return dict(
        components=[dict(id=c, label=COMPONENT_LABEL[c], kind=COMPONENT_KIND[c], metrics=METRICS[c],
                         calls=list(CALLS.get(c, {}))) for c in COMPONENTS],
        edges=[dict(source=a, target=b, fan=CALLS[a][b]) for a, b in EDGES],
        metrics={k: dict(label=v[0], unit=v[1]) for k, v in METRIC_INFO.items()},
        faults={k: dict(label=v["label"], tool=v["tool"], targets=v["targets"], primary=v["primary"])
                for k, v in FAULTS.items()},
        classes=CLASSES,
        thresholds={k: dict(metric=m, op=o, value=v) for k, (m, o, v) in THRESHOLD_RULES.items()},
        slo=dict(err_rate=SLO_ERR_RATE, latency_ms=SLO_LATENCY_MS),
        scenarios=[dict(id=k, title=v["title"], description=v["description"]) for k, v in SCENARIOS.items()],
        window=dict(seconds=30, stride=10, horizon=120),
    )


@app.get("/api/status")
def status():
    return S().status()


# ------------------------------------------------------------------------------ live control
class Control(BaseModel):
    action: str = Field(pattern="^(pause|resume|reset)$")


class Speed(BaseModel):
    speed: float = Field(ge=0.25, le=10)


class Inject(BaseModel):
    type: str
    target: str | None = None
    duration: float = Field(300, ge=60, le=900)
    intensity: float = Field(0.8, ge=0.2, le=1.0)


class SpikeReq(BaseModel):
    factor: float = Field(3.5, ge=1.5, le=5.0)
    hold: float = Field(150, ge=30, le=600)


class ReplayReq(BaseModel):
    run_id: str


class ScenarioReq(BaseModel):
    id: str


@app.post("/api/session/control")
def control(c: Control):
    s = S()
    if c.action == "pause":
        s.paused = True
    elif c.action == "resume":
        s.paused = False
    else:
        s.reset()
    return s.status()


@app.post("/api/session/speed")
def speed(sp: Speed):
    S().speed = sp.speed
    return S().status()


@app.post("/api/session/replay")
def replay(r: ReplayReq):
    try:
        S().start_replay(r.run_id)
    except (KeyError, StopIteration):
        raise HTTPException(404, f"unknown run {r.run_id}")
    return S().status()


@app.post("/api/faults/inject")
def inject(r: Inject):
    if r.type not in FAULTS:
        raise HTTPException(422, f"unknown fault type {r.type}")
    try:
        return S().inject(r.type, r.target, r.duration, r.intensity)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/faults/spike")
def spike(r: SpikeReq):
    try:
        return S().spike(r.factor, r.hold)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/faults/clear")
def clear():
    S().clear()
    return S().status()


@app.post("/api/scenarios/run")
def scenario(r: ScenarioReq):
    if r.id not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    try:
        S().run_scenario(r.id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return S().status()


# --------------------------------------------------------------------------- stateless predict
class PredictReq(BaseModel):
    samples: list[dict[str, float | None]] = Field(min_length=6, description="consecutive 5 s scrapes, oldest first")
    logs: list[list[dict]] | None = Field(None, description="optional per-sample log lines [{service, level, msg}]")


@app.post("/api/predict")
def predict(req: PredictReq):
    """Score a sequence of raw scrapes (≥ 6 × 5 s) with the full pipeline; returns the last window's verdict."""
    rt: OpsMindRuntime = state["rt"]
    st, tick = rt.new_state(), None
    missing = [c for c in RAW_COLUMNS if c not in req.samples[0]]
    if missing:
        raise HTTPException(422, f"missing metric columns, e.g. {missing[:3]}")
    with S().lock:
        for i, row in enumerate(req.samples):
            row = {c: (float("nan") if row.get(c) is None else row[c]) for c in RAW_COLUMNS}
            lines = [(i * 5.0, l.get("service", ""), l.get("level", "INFO"), l.get("msg", ""))
                     for l in ((req.logs[i] if req.logs and i < len(req.logs) else []))]
            out = rt.ingest(st, i * 5.0, row, lines)
            tick = out or tick
    if tick is None:
        raise HTTPException(422, "need an even number (≥ 6) of samples to complete a window")
    tick.pop("truth", None)
    return json.loads(dumps(tick))


# ------------------------------------------------------------------------------ benchmarks
_cache: dict = {}


def _results() -> dict:
    p = ARTIFACT_DIR / "results.json"
    if "results" not in _cache or _cache.get("mtime") != p.stat().st_mtime:
        _cache["results"] = json.loads(p.read_text())
        _cache["mtime"] = p.stat().st_mtime
    return _cache["results"]


@app.get("/api/benchmarks/sources")
def benchmark_sources():
    return dict(simulator=True, real=(ARTIFACT_DIR / "real" / "results.json").exists())


@app.get("/api/benchmarks")
def benchmarks(source: str = "simulator"):
    if source == "real":
        p = ARTIFACT_DIR / "real" / "results.json"
        if not p.exists():
            raise HTTPException(404, "no real-data results yet — run scripts/train_real.py")
        r = json.loads(p.read_text())
        extra = ARTIFACT_DIR / "real" / "sim_vs_real.json"
        r["sim_vs_real"] = json.loads(extra.read_text()) if extra.exists() else None
        return {k: v for k, v in r.items() if k != "episodes"}
    r = _results()
    return {k: v for k, v in r.items() if k not in ("episodes",)}


@app.get("/api/dataset")
def dataset():
    stats = json.loads((DATA_DIR / "dataset_stats.json").read_text())
    lh = ARTIFACT_DIR / "loghub_validation.json"
    return dict(stats=stats, templates=json.loads((DATA_DIR / "log_templates.json").read_text()),
                episodes=_results()["episodes"], loghub=json.loads(lh.read_text()) if lh.exists() else None)


@app.get("/api/replay/runs")
def replay_runs():
    runs = json.loads((DATA_DIR / "runs.json").read_text())
    for d in ("real", "real2"):
        real = DATA_DIR / d / "runs.json"
        if real.exists():
            runs += json.loads(real.read_text())
    return [dict(run_id=r["run_id"], kind=r["kind"], fault_type=r["fault_type"], target=r["target"],
                 intensity=r["intensity"], impact_start=r["impact_start"], split=r["split"])
            for r in runs if r["split"] in ("test", "real")]


@app.get("/api/real")
def real_eval():
    """Real Docker-testbed campaign: static thresholds vs simulator-trained (zero-shot) vs trained on real runs."""
    d = ARTIFACT_DIR / "real"
    if not (d / "results.json").exists() or not (d / "sim_vs_real.json").exists():
        return None
    r, sv = json.loads((d / "results.json").read_text()), json.loads((d / "sim_vs_real.json").read_text())
    ds = json.loads((DATA_DIR / "real_features" / "dataset_stats.json").read_text())
    key = ("f1", "precision", "recall", "detected", "false_alarms_per_hour", "mean_delay_s")
    prim = next(m for m in r["anomaly"]["models"] if m["key"] == r["anomaly"]["primary"])
    ew = next(m for m in r["earlywarning"]["models"] if m["model"] == "LSTM")
    ek = ("warned", "late", "missed", "eligible_episodes", "mean_lead_s", "false_alarms_per_hour")
    cls = r["classification"]
    rca = {m["method"]: m["episode"] for m in r["rootcause"]["methods"]}
    best = lambda fam: max((m for m in cls["models"] if m["family"] == fam), key=lambda m: m["macro_f1"])
    return dict(
        dataset=ds, test_runs=sv["test_runs"], test_fault_episodes=sv["test_fault_episodes"],
        detection=dict(static=sv["detection"]["static_thresholds"], simulator=sv["detection"]["simulator_zero_shot"],
                       real={**{k: prim[k] for k in key}, "model": prim["model"]}),
        early_warning=dict(static={k: sv["early_warning"]["static_thresholds"][k] for k in ek},
                           simulator={k: sv["early_warning"]["simulator_zero_shot"][k] for k in ek},
                           real={k: ew[k] for k in ek}),
        classification=dict(best_single=dict(model=best("single")["model"], f1=best("single")["macro_f1"]),
                            best_ensemble=dict(model=best("ensemble")["model"], f1=best("ensemble")["macro_f1"]),
                            pipeline=cls["pipeline"]),
        rootcause=dict(real=rca["SHAP + dependency graph"], baseline=rca["Highest raw CPU (baseline)"], zscore=rca["Largest z-score deviation"]),
        clustering=next(a for a in r["clustering"]["algorithms"] if a["model"] == "K-means"),
        episodes=r["episodes"],
    )


# ------------------------------------------------------------------------------- websocket
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    s = S()
    s.clients.add(ws)
    try:
        await ws.send_text(dumps(dict(type="hello", status=s.status(), snapshot=s.snapshot(120))))
        while True:
            msg = await ws.receive_text()           # keep-alive / client pings
            if msg == "status":
                await ws.send_text(dumps(s.status()))
            elif msg == "snapshot":
                await ws.send_text(dumps(dict(type="hello", status=s.status(), snapshot=s.snapshot(120))))
    except WebSocketDisconnect:
        pass
    finally:
        s.clients.discard(ws)


# ------------------------------------------------------------------------------- dashboard
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
