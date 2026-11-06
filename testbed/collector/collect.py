"""Pull a finished experiment from Prometheus into the OpsMind raw-metrics schema.

    python testbed/collector/collect.py --episodes data/real/episodes.jsonl --out data/real

Writes ``raw_metrics.parquet`` (same columns as the simulator: ``<component>.<metric>``, 5 s
steps) and ``runs.json`` (labels from the injector log), so the feature pipeline and models
run unchanged on real telemetry. Several PromQL expressions are approximations (noted inline);
untested on the authoring machine because Docker was unavailable.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opsmind.config import APP_SERVICES, COMPONENTS, METRICS, RAW_COLUMNS, RUN_SEC, SAMPLE_SEC  # noqa: E402

CNT = {c: f'container="opsmind-{c}"' for c in COMPONENTS}
R = "[15s]"
Z = " or vector(0)"        # series are absent when nothing happens (no errors, no locks): treat as 0


def promql(comp: str, metric: str) -> str:
    n = CNT[comp]
    app = comp in APP_SERVICES
    if metric == "cpu":
        return f"opsmind_container_cpu_percent{{{n}}}"
    if metric == "mem":
        return f"opsmind_container_mem_percent{{{n}}}"
    if metric == "net_retrans":
        return f"rate(opsmind_container_tcp_retrans_total{{{n}}}{R}){Z}"
    if metric == "up":
        # app services: scrape success x container running. Databases: container running only — the
        # exporters lose their connection under a database *slowdown*, which must not look like a crash.
        if app:
            return f'(min(up{{job="{comp}"}}) * min(opsmind_container_up{{{n}}})){Z}'
        return f"(min(opsmind_container_up{{{n}}})){Z}"
    if app:
        s = f'service="{comp}"'
        return {
            "rps": f"(sum(rate(http_requests_total{{{s}}}{R}))){Z}",
            "lat_p95": f"(1000 * histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{{s}}}[30s])) by (le))){Z}",
            "err_rate": f'(100 * sum(rate(http_requests_total{{{s},status=~"5.."}}{R})) / clamp_min(sum(rate(http_requests_total{{{s}}}{R})), 1e-9)){Z}',
        }[metric]
    if comp == "postgres":
        return {
            "rps": f"(sum(rate(pg_stat_database_xact_commit{R}))){Z}",
            # client-side database latency (histogram exported by the services)
            "lat_p95": f"(1000 * histogram_quantile(0.95, sum(rate(db_query_duration_seconds_bucket[30s])) by (le))){Z}",
            "err_rate": f"(100 * sum(rate(pg_stat_database_xact_rollback{R})) / clamp_min(sum(rate(pg_stat_database_xact_commit{R})), 1e-9)){Z}",
            "conns": f"(sum(last_over_time(pg_stat_database_numbackends[30s]))){Z}",
            "locks": f'(sum(pg_locks_count{{mode=~".*exclusive.*"}})){Z}',
        }[metric]
    return {   # redis
        "rps": f"(rate(redis_commands_processed_total{R})){Z}",
        "lat_p95": f"(1000 * rate(redis_commands_duration_seconds_total{R}) / clamp_min(rate(redis_commands_processed_total{R}), 1e-9)){Z}",
        "err_rate": f"(100 * rate(redis_rejected_connections_total{R})){Z}",
        "hit_ratio": f"(rate(redis_keyspace_hits_total{R}) / clamp_min(rate(redis_keyspace_hits_total{R}) + rate(redis_keyspace_misses_total{R}), 1e-9)){Z}",
    }[metric]


def query_range(prom: str, q: str, start: float, end: float) -> np.ndarray:
    r = httpx.get(f"{prom}/api/v1/query_range", params=dict(query=q, start=start, end=end, step=SAMPLE_SEC), timeout=60)
    r.raise_for_status()
    n = int((end - start) // SAMPLE_SEC)
    out = np.full(n, np.nan, np.float32)
    res = r.json()["data"]["result"]
    if res:
        for ts, v in res[0]["values"]:
            i = int((float(ts) - start) // SAMPLE_SEC)
            if 0 <= i < n:
                out[i] = float(v)
    return out


# ---------------------------------------------------------------------------------- logs
DOCKER = shutil.which("docker") or "/Applications/Docker.app/Contents/Resources/bin/docker"
INFO_KEEP = 20          # keep 1 in 20 INFO lines (sampled access log); every WARN/ERROR/FATAL is kept
PG_LEVEL = re.compile(r"\b(PANIC|FATAL|ERROR|WARNING):")
PG_MSG = re.compile(r"\[\d+\]\s+(?:[A-Z]+:\s+)?(.*)$")
REDIS_MSG = re.compile(r"^\d+:[A-Z]\s+\d+ \w+ \d+ [\d:.]+ ([*#.-]) (.*)$")


def _parse(service: str, ts: float, body: str) -> tuple[float, str, str, str] | None:
    body = body.strip()
    if not body:
        return None
    if service in ("gateway", "auth", "order", "payment"):
        try:
            j = json.loads(body)
            return (ts, service, j["level"], j["msg"])
        except Exception:
            return (ts, service, "INFO", body)
    if service == "postgres":
        m = PG_LEVEL.search(body)
        level = {"WARNING": "WARN", "PANIC": "FATAL"}.get(m.group(1), m.group(1)) if m else "INFO"
        mm = PG_MSG.search(body)
        return (ts, service, level, mm.group(1) if mm else body)
    m = REDIS_MSG.match(body)
    return (ts, service, "WARN" if (m and m.group(1) == "#") else "INFO", m.group(2) if m else body)


def collect_logs(t0: float, t1: float) -> list[tuple[float, str, str, str]]:
    """Container logs between t0 and t1 as (t_seconds_since_run_start, service, level, message)."""
    out = []
    for comp in COMPONENTS:
        r = subprocess.run([DOCKER, "logs", "--timestamps", "--since", str(t0), "--until", str(t1), f"opsmind-{comp}"],
                           capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            stamp, _, body = line.partition(" ")
            try:
                ts = datetime.fromisoformat(stamp[:26].rstrip("Z") + "+00:00").timestamp()
            except ValueError:
                continue
            rec = _parse(comp, ts - t0, body)
            if rec and (rec[2] != "INFO" or zlib.crc32(f"{comp}{ts}".encode()) % INFO_KEEP == 0):
                out.append(rec)
    return sorted(out)


def main(prom: str, episodes: Path, out: Path, prefix: str = "real") -> None:
    out.mkdir(parents=True, exist_ok=True)
    frames, runs, logs = [], [], []
    for i, line in enumerate(episodes.read_text().splitlines()):
        ep = json.loads(line)
        t0, t1 = ep["run_start"], ep["run_end"]
        run_id = f"{prefix}{i:03d}"
        cols = {f"{c}.{m}": query_range(prom, promql(c, m), t0, t1) for c in COMPONENTS for m in METRICS[c]}
        df = pd.DataFrame(cols)[RAW_COLUMNS]
        # the host sleeping / Prometheus gaps leave long all-NaN stretches: keep only the valid prefix
        gap = df["gateway.cpu"].isna().rolling(6).sum().to_numpy() >= 6
        if gap.any():
            cut = int(np.argmax(gap)) - 5
            print(f"  run {i}: {len(df) - cut} steps lost to a monitoring gap (host asleep?) — trimmed to {cut} steps", flush=True)
            df, t1 = df.iloc[:cut].reset_index(drop=True), t0 + cut * SAMPLE_SEC
        df.insert(0, "t", np.arange(len(df)) * SAMPLE_SEC)
        df.insert(0, "run_id", run_id)
        frames.append(df)
        ld = pd.DataFrame(collect_logs(t0, t0 + len(df) * SAMPLE_SEC), columns=["t", "service", "level", "msg"])
        ld.insert(0, "run_id", run_id)
        logs.append(ld)
        fs = (ep["fault_start"] - t0) if ep.get("fault_type") else None
        fe = (ep["fault_end"] - t0) if ep.get("fault_type") else None
        impact = None
        if fs is not None:
            from opsmind.sim.scenario import find_impact
            impact = find_impact(df[RAW_COLUMNS].to_numpy(np.float32), fs, fe, strict=True)
        runs.append(dict(run_id=run_id, kind="fault" if fs is not None else "normal", index=i, fault_type=ep.get("fault_type"),
                         target=ep.get("target"), fault_start=fs, fault_end=fe, impact_start=impact,
                         intensity=ep.get("intensity"), ramp=ep.get("ramp"), spikes=[], batches=[], wall_start=t0, split="real",
                         duration=t1 - t0))
        print(f"{run_id}: {ep.get('fault_type') or 'normal':16s} {len(df)} steps, {len(ld)} log lines, impact_start={impact}", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(out / "raw_metrics.parquet", index=False)
    pd.concat(logs, ignore_index=True).to_parquet(out / "raw_logs.parquet", index=False)
    (out / "runs.json").write_text(json.dumps(runs, indent=1))
    print(f"wrote {len(runs)} runs to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--episodes", type=Path, default=Path("data/real/episodes.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/real"))
    ap.add_argument("--prefix", default="real", help="run id prefix; campaigns must use distinct prefixes")
    a = ap.parse_args()
    main(a.prometheus, a.episodes, a.out, a.prefix)
