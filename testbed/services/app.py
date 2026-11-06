"""One FastAPI codebase for all four testbed services; behaviour is selected with SERVICE.

    gateway  -> auth, order          (public entry point)
    auth     -> redis (sessions), postgres (users on cache miss)
    order    -> payment, postgres (orders), redis (carts)
    payment  -> postgres (ledger)

The call graph and fan-out match opsmind/config.py::CALLS, which the ML side
reuses for root-cause ranking. Metrics are exposed for Prometheus; logs are
structured JSON on stdout (parsed with Drain downstream).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid

import asyncpg
import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app

SERVICE = os.environ["SERVICE"]
AUTH_URL = os.getenv("AUTH_URL", "http://auth:8000")
ORDER_URL = os.getenv("ORDER_URL", "http://order:8000")
PAYMENT_URL = os.getenv("PAYMENT_URL", "http://payment:8000")
PG_DSN = os.getenv("PG_DSN", "postgresql://opsmind:opsmind@postgres:5432/shop")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TIMEOUT = float(os.getenv("DOWNSTREAM_TIMEOUT", "1.0"))


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        return json.dumps({"ts": time.time(), "service": SERVICE, "level": r.levelname, "msg": r.getMessage()})


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("svc")
log.setLevel(logging.INFO)
log.addHandler(handler)

REQS = Counter("http_requests_total", "HTTP requests", ["service", "route", "status"])
LAT = Histogram("http_request_duration_seconds", "Request latency", ["service", "route"],
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))

DBLAT = Histogram("db_query_duration_seconds", "Database call latency as seen by the client", ["service"],
                  buckets=(.002, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))

app = FastAPI(title=f"opsmind-{SERVICE}")
app.mount("/metrics", make_asgi_app())
_leak: list[bytes] = []
pg: asyncpg.Pool | None = None
rd: aioredis.Redis | None = None
http = httpx.AsyncClient(timeout=TIMEOUT)


@app.on_event("startup")
async def startup() -> None:
    global pg, rd
    if SERVICE in ("auth", "order", "payment"):
        pg = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=int(os.getenv("PG_POOL", "10")))
        async with pg.acquire() as c:
            await c.execute("CREATE TABLE IF NOT EXISTS users(id int primary key, name text)")
            await c.execute("CREATE TABLE IF NOT EXISTS orders(id text primary key, total numeric, ts timestamptz default now())")
            await c.execute("CREATE TABLE IF NOT EXISTS ledger(txn text primary key, amount numeric, ts timestamptz default now())")
    if SERVICE in ("auth", "order"):
        rd = aioredis.from_url(REDIS_URL, socket_timeout=TIMEOUT)
    log.info("service started version=1.4.2 port=8000 pid=%d", os.getpid())


@app.middleware("http")
async def observe(request: Request, call_next):
    if request.url.path.startswith("/metrics"):
        return await call_next(request)
    route = request.scope.get("route").path if request.scope.get("route") else request.url.path
    t0 = time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
        return resp
    finally:
        LAT.labels(SERVICE, route).observe(time.perf_counter() - t0)
        REQS.labels(SERVICE, route, str(status)).inc()


async def call(url: str, method: str = "GET", **kw) -> httpx.Response:
    try:
        r = await http.request(method, url, **kw)
        r.raise_for_status()
        return r
    except httpx.TimeoutException:
        log.error("upstream %s timed out after %dms", url.split("/")[2].split(":")[0], int(TIMEOUT * 1000))
        raise HTTPException(504, "upstream timeout")
    except httpx.ConnectError:
        log.error("connection refused to %s", url.split("/")[2])
        raise HTTPException(502, "upstream refused")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"upstream {e.response.status_code}")


@app.get("/health")
async def health():
    return {"service": SERVICE, "ok": True}


@app.get("/leak")
async def leak(mb: int = 8):
    """Fault-injection hook: grow resident memory (memory-leak fault)."""
    _leak.append(b"x" * mb * 1024 * 1024)
    log.warning("leaked %d MB total=%d MB", mb, len(_leak) * mb)
    return {"leaked_mb": len(_leak) * mb}


@app.get("/leak/clear")
async def leak_clear():
    _leak.clear()
    return {"leaked_mb": 0}


# ------------------------------------------------------------------------------- gateway
if SERVICE == "gateway":
    @app.get("/api/products/{pid}")
    async def product(pid: int):
        await call(f"{AUTH_URL}/validate")
        return {"id": pid, "name": f"product-{pid}"}

    @app.post("/api/orders")
    async def create_order():
        await call(f"{AUTH_URL}/validate")
        return (await call(f"{ORDER_URL}/orders", "POST", json={"cart": random.randint(1, 5000)})).json()

    @app.post("/auth/login")
    async def login():
        return (await call(f"{AUTH_URL}/login", "POST")).json()

# ----------------------------------------------------------------------------------- auth
if SERVICE == "auth":
    @app.post("/login")
    @app.get("/validate")
    async def session():
        uid = random.randint(1, 5000)
        try:
            hit = await rd.get(f"session:{uid}")
        except Exception:
            log.warning("redis connection refused falling back to postgres")
            hit = None
        if not hit:
            with DBLAT.labels(SERVICE).time():
                async with pg.acquire() as c:
                    await c.fetchval("SELECT 1")
            try:
                await rd.set(f"session:{uid}", "1", ex=300)
            except Exception:
                pass
        log.info("token issued user=%d ttl=3600", uid)
        return {"user": uid}

# ---------------------------------------------------------------------------------- order
if SERVICE == "order":
    @app.post("/orders")
    async def orders():
        oid = f"o-{uuid.uuid4().hex[:8]}"
        try:
            await rd.get("cart:1")
            await rd.get("cart:2")
        except Exception:
            log.warning("redis connection refused falling back to postgres")
        with DBLAT.labels(SERVICE).time():
            async with pg.acquire() as c:
                await c.execute("INSERT INTO orders(id, total) VALUES($1, $2)", oid, random.uniform(5, 400))
        if random.random() < 0.35:                       # not every order needs a charge
            await call(f"{PAYMENT_URL}/charge", "POST", json={"order": oid})
        log.info("order %s created total=%.2f", oid, random.uniform(5, 400))
        return {"order": oid}

# -------------------------------------------------------------------------------- payment
if SERVICE == "payment":
    @app.post("/charge")
    async def charge():
        txn = f"t{random.randint(100000, 999999)}{uuid.uuid4().hex[:4]}"
        await asyncio.sleep(random.uniform(0.02, 0.05))      # bank gateway
        with DBLAT.labels(SERVICE).time():
            async with pg.acquire() as c:
                await c.execute("INSERT INTO ledger(txn, amount) VALUES($1, $2)", txn, random.uniform(5, 400))
        log.info("charge captured txn=%s", txn)
        return {"txn": txn}
