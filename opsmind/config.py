"""Central configuration: components, metrics, topology, windowing and fault catalogue.

Everything that must stay identical between data generation, training, the API
runtime and the dashboard lives here.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# --------------------------------------------------------------------------- time
SAMPLE_SEC = 5            # Prometheus scrape interval
WINDOW_SAMPLES = 6        # 30 s feature window
STRIDE_SAMPLES = 2        # a new window every 10 s
WINDOW_SEC = WINDOW_SAMPLES * SAMPLE_SEC
STRIDE_SEC = STRIDE_SAMPLES * SAMPLE_SEC
RUN_SEC = 1800            # one experiment run = 30 min
RUN_STEPS = RUN_SEC // SAMPLE_SEC

SEQ_LEN = 10              # early-warning model sees the last 5 minutes ...
SEQ_STEP_WINDOWS = 3      # ... as 10 windows spaced 30 s apart
HORIZON_SEC = 120         # "incident within the next 2 minutes"
RECOVERY_SEC = 90         # post-fault grace period excluded from scoring

# ------------------------------------------------------------------ components
COMPONENTS = ["gateway", "auth", "order", "payment", "postgres", "redis"]
APP_SERVICES = ["gateway", "auth", "order", "payment"]

COMPONENT_LABEL = {
    "gateway": "API Gateway",
    "auth": "Auth Service",
    "order": "Order Service",
    "payment": "Payment Service",
    "postgres": "PostgreSQL",
    "redis": "Redis Cache",
}
COMPONENT_KIND = {
    "gateway": "service", "auth": "service", "order": "service",
    "payment": "service", "postgres": "database", "redis": "cache",
}

_APP_METRICS = ["cpu", "mem", "rps", "lat_p95", "err_rate", "up", "net_retrans"]
METRICS: dict[str, list[str]] = {
    **{c: list(_APP_METRICS) for c in APP_SERVICES},
    "postgres": ["cpu", "mem", "rps", "lat_p95", "err_rate", "up", "conns", "locks"],
    "redis": ["cpu", "mem", "rps", "lat_p95", "err_rate", "up", "hit_ratio"],
}

METRIC_INFO = {
    "cpu": ("CPU", "%"),
    "mem": ("Memory", "%"),
    "rps": ("Throughput", "req/s"),
    "lat_p95": ("p95 latency", "ms"),
    "err_rate": ("Error rate", "%"),
    "up": ("Up", ""),
    "net_retrans": ("TCP retransmits", "/s"),
    "conns": ("DB connections", ""),
    "locks": ("Lock waits", ""),
    "hit_ratio": ("Cache hit ratio", ""),
}

RAW_COLUMNS: list[str] = [f"{c}.{m}" for c in COMPONENTS for m in METRICS[c]]

# Caller -> callee with the mean number of calls the callee receives per
# request handled by the caller. Defined once, reused by the simulator and
# by root-cause ranking (as stated in the proposal).
CALLS: dict[str, dict[str, float]] = {
    "gateway": {"auth": 0.6, "order": 0.45},
    "auth": {"redis": 1.0, "postgres": 0.15},
    "order": {"payment": 0.35, "postgres": 0.9, "redis": 1.2},
    "payment": {"postgres": 0.8},
}
EDGES: list[tuple[str, str]] = [(a, b) for a, d in CALLS.items() for b in d]

# Static thresholds = the "Prometheus alert rules" baseline of the proposal.
THRESHOLD_RULES = {
    "HighCPU": ("cpu", ">", 90.0),
    "HighMemory": ("mem", ">", 90.0),
    "HighErrorRate": ("err_rate", ">", 5.0),
    "HighLatency": ("lat_p95", ">", 500.0),
    "InstanceDown": ("up", "<", 0.5),
}

# User-facing SLO used to define when an incident actually "impacts" users.
SLO_ERR_RATE = 3.0       # %
SLO_LATENCY_MS = 200.0   # gateway p95

# ------------------------------------------------------------------ fault catalogue
FAULTS: dict[str, dict] = {
    "cpu_saturation": {
        "label": "CPU saturation",
        "tool": "stress-ng --cpu 0 (inside container)",
        "targets": ["gateway", "auth", "order", "payment", "postgres", "redis"],
        "primary": None,
    },
    "memory_leak": {
        "label": "Memory pressure / leak",
        "tool": "leaking endpoint / stress-ng --vm",
        "targets": ["order", "payment", "auth", "redis"],
        "primary": "order",
    },
    "db_slowdown": {
        "label": "Database slowdown",
        "tool": "pg_sleep queries + table locks",
        "targets": ["postgres"],
        "primary": "postgres",
    },
    "network_latency": {
        "label": "Network latency / loss",
        "tool": "tc netem via Pumba",
        "targets": ["payment", "order", "auth", "postgres", "redis"],
        "primary": "payment",
    },
    "service_crash": {
        "label": "Service crash",
        "tool": "docker kill / stop",
        "targets": ["auth", "payment", "order", "gateway", "postgres"],
        "primary": "auth",
    },
    "cache_failure": {
        "label": "Cache failure",
        "tool": "docker stop redis",
        "targets": ["redis"],
        "primary": "redis",
    },
}
FAULT_TYPES = list(FAULTS)
CLASSES = ["normal"] + FAULT_TYPES


def column_component(col: str) -> str | None:
    """Component a feature/column belongs to (``order.cpu__mean`` -> ``order``)."""
    head = col.split(".", 1)[0]
    if head == "log":
        return col.split(".")[1]
    return head if head in COMPONENTS else None
