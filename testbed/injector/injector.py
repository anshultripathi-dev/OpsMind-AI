"""Scripted experiment runner: normal traffic -> one injected fault -> recovery (≈30 min per run).

    python testbed/injector/injector.py --runs 6 --out data/real/episodes.jsonl

Every episode appends one JSON line with the exact fault start/end — the only source of
labels (no manual labelling). Requires Docker, the compose stack, and for network faults the
`gaiaadm/pumba` image. Untested on the authoring machine (no Docker there).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import threading
import time
from pathlib import Path

FAULTS = {
    "cpu_saturation": ["gateway", "auth", "order", "payment", "postgres", "redis"],
    "memory_leak": ["order", "payment", "auth"],
    "db_slowdown": ["postgres"],
    "network_latency": ["payment", "order", "auth"],
    "service_crash": ["auth", "payment", "order"],
    "cache_failure": ["redis"],
}
PRIMARY = {"memory_leak": "order", "db_slowdown": "postgres", "network_latency": "payment",
           "service_crash": "auth", "cache_failure": "redis"}


def sh(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def c(name: str) -> str:
    return f"opsmind-{name}"


def start(ftype: str, target: str, intensity: float, duration: float):
    """Begin the fault; returns a callable that ends it."""
    if ftype == "cpu_saturation":
        if target in ("postgres", "redis"):      # Alpine images: no stress-ng, burn CPU with `yes`
            sh("docker", "exec", "-d", c(target), "sh", "-c", f"timeout {int(duration)} yes > /dev/null")
            return lambda: sh("docker", "exec", c(target), "pkill", "yes", check=False)
        sh("docker", "exec", "-d", c(target), "sh", "-c", f"stress-ng --cpu 1 --cpu-load {int(30 + 70 * intensity)} --timeout {int(duration)}s")
        return lambda: sh("docker", "exec", c(target), "pkill", "stress-ng", check=False)
    if ftype == "memory_leak":
        # grow toward ~90 % of the 512 MB limit * intensity over the fault window (never blindly past the OOM limit)
        mb = max(1, int(0.9 * intensity * 480 / max(duration / 4, 1)))
        procs = subprocess.Popen(["sh", "-c", f"for i in $(seq 1 {int(duration // 4)}); do docker exec {c(target)} python -c \"import urllib.request as u;u.urlopen('http://localhost:8000/leak?mb={mb}')\"; sleep 4; done"])
        return lambda: (procs.terminate(), sh("docker", "exec", c(target), "python", "-c", "import urllib.request as u;u.urlopen('http://localhost:8000/leak/clear')", check=False))
    if ftype == "db_slowdown":
        sql = f"SELECT pg_sleep({0.05 + intensity * 0.6}); LOCK TABLE orders IN ACCESS EXCLUSIVE MODE; SELECT pg_sleep({duration});"
        p = subprocess.Popen(["docker", "exec", c("postgres"), "psql", "-U", "opsmind", "-d", "shop", "-c", sql])
        sh("docker", "exec", "-d", c("postgres"), "sh", "-c", f"for i in $(seq 1 {int(20 * intensity) + 2}); do psql -U opsmind -d shop -c 'SELECT count(*) FROM orders' & done")
        return lambda: p.terminate()
    if ftype == "network_latency":
        # tc netem (what Pumba drives under the hood) applied directly in the target's network namespace.
        # Pumba's CLI cannot combine delay and loss, so we call tc ourselves for a single combined qdisc.
        delay, loss = int(60 + 900 * intensity), round(intensity * 6, 1)
        tc = ["docker", "run", "--rm", "--net", f"container:{c(target)}", "--cap-add", "NET_ADMIN",
              "--entrypoint", "tc", "ghcr.io/alexei-led/pumba-alpine-nettools:latest", "qdisc"]
        sh(*tc, "replace", "dev", "eth0", "root", "netem", "delay", f"{delay}ms", "loss", f"{loss}%")
        return lambda: sh(*tc, "del", "dev", "eth0", "root", check=False)
    if ftype in ("service_crash", "cache_failure"):
        sh("docker", "kill" if ftype == "service_crash" else "stop", c(target))
        return lambda: sh("docker", "start", c(target))
    raise ValueError(ftype)


def start_ramped(ftype: str, target: str, intensity: float, duration: float, ramp: float):
    """Gradual onset: the fault builds over ``ramp`` seconds and then holds, like a real degradation.

    Abrupt faults (a table lock, an instant kill) leave no precursor, so no model could warn about them.
    Returns a callable that ends the fault. Falls back to the abrupt version for fault types without a ramp.
    """
    stop_evt = threading.Event()
    threads: list[threading.Thread] = []

    def spawn(fn) -> None:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        threads.append(t)

    def finish(cleanup) -> "callable":
        def stop() -> None:
            stop_evt.set()
            for t in threads:
                t.join(timeout=30)
            cleanup()
        return stop

    if ftype == "network_latency":
        dmax, lmax = int(60 + 900 * intensity), intensity * 6
        tc = ["docker", "run", "--rm", "--net", f"container:{c(target)}", "--cap-add", "NET_ADMIN",
              "--entrypoint", "tc", "ghcr.io/alexei-led/pumba-alpine-nettools:latest", "qdisc"]
        steps = max(int(ramp // 10), 1)

        def run() -> None:
            for k in range(1, steps + 1):
                f = k / steps
                sh(*tc, "replace", "dev", "eth0", "root", "netem", "delay", f"{int(20 + (dmax - 20) * f)}ms",
                   "loss", f"{lmax * f:.2f}%", check=False)
                if stop_evt.wait(10):
                    return
        spawn(run)
        return finish(lambda: sh(*tc, "del", "dev", "eth0", "root", check=False))

    if ftype == "db_slowdown":
        clients = 2 + round(3 * intensity)            # heavy queries pile up on the single-CPU database
        loop = "while :; do psql -U opsmind -d shop -c 'SELECT sum(i) FROM generate_series(1,25000000) i' > /dev/null; done"

        holder: dict = {}

        def run() -> None:
            for k in range(clients):
                if k and stop_evt.wait(ramp / clients):
                    return
                sh("docker", "exec", "-d", c("postgres"), "sh", "-c", f"timeout {int(duration)} sh -c \"{loop}\"", check=False)
            # the cascade ends in lock contention: that is when users are hurt, after a visible precursor
            if not stop_evt.wait(max(ramp / clients, 1)):
                sql = f"BEGIN; LOCK TABLE orders IN ACCESS EXCLUSIVE MODE; SELECT pg_sleep({max(duration - ramp, 10)}); COMMIT;"
                holder["p"] = subprocess.Popen(["docker", "exec", c("postgres"), "psql", "-U", "opsmind", "-d", "shop", "-c", sql],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def cleanup() -> None:
            if "p" in holder:
                holder["p"].terminate()
            sh("docker", "exec", c("postgres"), "pkill", "-f", "generate_series", check=False)
            sh("docker", "exec", c("postgres"), "psql", "-U", "opsmind", "-d", "shop", "-c",
               "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query LIKE 'LOCK TABLE%' OR query LIKE '%generate_series%'", check=False)
        spawn(run)
        return finish(cleanup)

    if ftype == "service_crash":
        flaps = 2 + int(2 * intensity)                # a crash-loop prelude, then a sustained outage

        def run() -> None:
            gap = ramp / (flaps + 1)
            for _ in range(flaps):
                if stop_evt.wait(gap):
                    return
                sh("docker", "kill", c(target), check=False)
                if stop_evt.wait(8):
                    break
                sh("docker", "start", c(target), check=False)
            if not stop_evt.wait(gap):
                sh("docker", "kill", c(target), check=False)
        spawn(run)
        return finish(lambda: sh("docker", "start", c(target), check=False))

    if ftype == "cpu_saturation" and target not in ("postgres", "redis"):
        steps = max(int(ramp // 20), 1)
        top = 30 + 70 * intensity

        def run() -> None:
            for k in range(1, steps + 1):
                sh("docker", "exec", c(target), "pkill", "stress-ng", check=False)
                load = int(15 + (top - 15) * k / steps)
                sh("docker", "exec", "-d", c(target), "sh", "-c", f"stress-ng --cpu 1 --cpu-load {load} --timeout {int(duration)}s", check=False)
                if stop_evt.wait(20):
                    return
        spawn(run)
        return finish(lambda: sh("docker", "exec", c(target), "pkill", "stress-ng", check=False))

    return start(ftype, target, intensity, duration)       # memory leak already ramps; cache stop is inherently sudden


RAMPED = ("network_latency", "db_slowdown", "service_crash", "cpu_saturation")


APP_CONTAINERS = ["gateway", "auth", "order", "payment"]


def reset_stack(settle: float = 20.0, timeout: float = 120.0) -> None:
    """Start every run from a known-good state.

    Long-lived services can end up degraded after injected faults (a gateway whose upstream calls all time out
    until it is restarted), which would silently contaminate later runs. So: clear leftover faults, make sure
    everything is up, restart the application services, and wait until a real request works end to end.
    """
    import urllib.request
    for name in APP_CONTAINERS:
        sh("docker", "run", "--rm", "--net", f"container:{c(name)}", "--cap-add", "NET_ADMIN", "--entrypoint", "tc",
           "ghcr.io/alexei-led/pumba-alpine-nettools:latest", "qdisc", "del", "dev", "eth0", "root", check=False)
        sh("docker", "exec", c(name), "pkill", "stress-ng", check=False)
    sh("docker", "exec", c("postgres"), "pkill", "-f", "generate_series", check=False)
    for name in APP_CONTAINERS + ["postgres", "redis"]:
        sh("docker", "start", c(name), check=False)
    sh("docker", "restart", *[c(n) for n in APP_CONTAINERS], check=False)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ok = [urllib.request.urlopen(urllib.request.Request(f"http://localhost:8080{p}", method=m, data=b"" if m == "POST" else None), timeout=3).status
                  for p, m in (("/api/products/1", "GET"), ("/auth/login", "POST"), ("/api/orders", "POST"))]
            if all(x == 200 for x in ok):
                time.sleep(settle)
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("stack did not become healthy; refusing to start a run on a broken system")


def one_run(rng: random.Random, ftype: str | None, out: Path, warmup=(360, 720), run_sec=1800,
            dur=(240, 540), imin=0.25, ramp_frac: float = 0.0, reset: bool = False) -> None:
    if reset:
        reset_stack()
    t_run = time.time()
    rec = dict(run_start=t_run, fault_type=ftype)
    time.sleep(rng.uniform(*warmup))
    if ftype:
        target = PRIMARY.get(ftype) if rng.random() < 0.7 and ftype in PRIMARY else rng.choice(FAULTS[ftype])
        duration, intensity = rng.uniform(*dur), imin + (1 - imin) * rng.betavariate(1.8, 1.2)
        rec.update(target=target, duration=duration, intensity=round(intensity, 3), fault_start=time.time())
        ramp = duration * ramp_frac if ftype in RAMPED else 0.0
        rec["ramp"] = round(ramp, 1)
        stop = start_ramped(ftype, target, intensity, duration, ramp) if ramp > 0 else start(ftype, target, intensity, duration)
        time.sleep(duration)
        stop()
        rec["fault_end"] = time.time()
    time.sleep(max(0, run_sec - (time.time() - t_run)))
    rec["run_end"] = time.time()
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


def smoke(ftype: str, target: str, seconds: int, intensity: float, prom: str = "http://localhost:9090", ramp_frac: float = 0.0) -> None:
    """Inject one fault briefly and print the target's metrics before / during / after."""
    import sys
    import httpx
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from testbed.collector.collect import promql
    from opsmind.config import COMPONENTS, METRICS
    def snap(label: str) -> None:
        vals = []
        for comp in COMPONENTS:
            if comp not in (target, "gateway", "postgres", "redis"):
                continue
            for m in METRICS[comp]:
                if m in ("cpu", "mem", "lat_p95", "err_rate", "up", "locks", "conns", "hit_ratio", "net_retrans"):
                    r = httpx.get(f"{prom}/api/v1/query", params=dict(query=promql(comp, m))).json()["data"]["result"]
                    vals.append(f"{comp}.{m}={float(r[0]['value'][1]):.2f}" if r else f"{comp}.{m}=∅")
        print(f"[{label}]", "  ".join(vals), flush=True)
    snap("before")
    stop = start_ramped(ftype, target, intensity, seconds, seconds * ramp_frac) if ramp_frac > 0 else start(ftype, target, intensity, seconds)
    time.sleep(seconds * 0.35)
    snap("mid-ramp")
    time.sleep(seconds * 0.35)
    snap("during")
    time.sleep(seconds * 0.3)
    stop()
    time.sleep(25)
    snap("after")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", nargs=4, metavar=("FAULT", "TARGET", "SECONDS", "INTENSITY"), help="inject one fault briefly and show its metric effect")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--normal-every", type=int, default=5, help="every n-th run is a fault-free control run")
    ap.add_argument("--out", type=Path, default=Path("data/real/episodes.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plan", help="comma-separated run types, e.g. normal,db_slowdown,normal (overrides --runs)")
    ap.add_argument("--run-sec", type=int, default=1800)
    ap.add_argument("--warmup", type=float, nargs=2, default=[360, 720])
    ap.add_argument("--dur", type=float, nargs=2, default=[240, 540], help="fault duration range (s)")
    ap.add_argument("--imin", type=float, default=0.25, help="minimum fault intensity")
    ap.add_argument("--reset", action="store_true", help="restart the application services and verify health before every run")
    ap.add_argument("--ramp-frac", type=float, default=0.0, help="share of the fault window spent ramping up (0 = abrupt faults)")
    a = ap.parse_args()
    if a.smoke:
        smoke(a.smoke[0], a.smoke[1], int(a.smoke[2]), float(a.smoke[3]), ramp_frac=a.ramp_frac)
        raise SystemExit
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    if a.plan:
        for item in a.plan.split(","):
            one_run(rng, None if item == "normal" else item, a.out, tuple(a.warmup), a.run_sec, tuple(a.dur), a.imin, a.ramp_frac, a.reset)
        raise SystemExit
    types = list(FAULTS)
    for i in range(a.runs):
        one_run(rng, None if (i + 1) % a.normal_every == 0 else types[i % len(types)], a.out,
                tuple(a.warmup), a.run_sec, tuple(a.dur), a.imin, a.ramp_frac, a.reset)
