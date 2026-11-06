"""Per-container resource metrics straight from the Docker API (works on Linux and Docker Desktop).

cAdvisor cannot see container names on Docker Desktop for Mac, so this tiny exporter provides the
same signals the ML features need: CPU % of the container's CPU limit, memory % of its limit,
liveness, and TCP retransmits read from /proc/net/snmp inside the container.
"""
import re
import threading
import time

import docker
from prometheus_client import Gauge, start_http_server

PREFIX = "opsmind-"
G = {k: Gauge(f"opsmind_container_{k}", k, ["container"]) for k in ("cpu_percent", "mem_percent", "up", "tcp_retrans_total")}
client = docker.from_env()
streams: dict[str, threading.Thread] = {}


def cpu_pct(s: dict, limit_cpus: float) -> float:
    cpu = s["cpu_stats"]["cpu_usage"]["total_usage"] - s["precpu_stats"]["cpu_usage"].get("total_usage", 0)
    sys_ = s["cpu_stats"].get("system_cpu_usage", 0) - s["precpu_stats"].get("system_cpu_usage", 0)
    ncpu = s["cpu_stats"].get("online_cpus") or 1
    return 100.0 * cpu / sys_ * ncpu / limit_cpus if sys_ > 0 and cpu >= 0 else 0.0


def follow(name: str) -> None:
    try:
        c = client.containers.get(name)
        limit = (c.attrs["HostConfig"].get("NanoCpus") or 0) / 1e9 or 1.0
        for s in c.stats(stream=True, decode=True):
            mem = s["memory_stats"]
            used = mem.get("usage", 0) - mem.get("stats", {}).get("inactive_file", 0)
            G["cpu_percent"].labels(name).set(cpu_pct(s, limit))
            G["mem_percent"].labels(name).set(100.0 * used / mem["limit"] if mem.get("limit") else 0.0)
            G["up"].labels(name).set(1)
    except Exception:
        pass
    G["up"].labels(name).set(0)
    G["cpu_percent"].labels(name).set(0)
    G["mem_percent"].labels(name).set(0)


def retrans(name: str) -> None:
    try:
        out = client.containers.get(name).exec_run(["cat", "/proc/net/snmp"]).output.decode()
        lines = [l for l in out.splitlines() if l.startswith("Tcp:")]
        keys, vals = lines[0].split()[1:], lines[1].split()[1:]
        G["tcp_retrans_total"].labels(name).set(float(vals[keys.index("RetransSegs")]))
    except Exception:
        pass


if __name__ == "__main__":
    start_http_server(9200)
    while True:
        names = [c.name for c in client.containers.list(all=True) if c.name.startswith(PREFIX)]
        for n in names:
            if n not in streams or not streams[n].is_alive():
                running = client.containers.get(n).status == "running"
                if running:
                    streams[n] = threading.Thread(target=follow, args=(n,), daemon=True)
                    streams[n].start()
                else:
                    G["up"].labels(n).set(0)
                    G["cpu_percent"].labels(n).set(0)
                    G["mem_percent"].labels(n).set(0)
            if n.split("-", 1)[1] in ("gateway", "auth", "order", "payment"):
                retrans(n)
        time.sleep(5)
