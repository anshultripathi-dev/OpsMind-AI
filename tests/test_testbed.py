"""Offline tests for the Docker-testbed glue (no Docker required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testbed" / "collector"))

import collect  # noqa: E402

from opsmind.config import COMPONENTS, METRICS, RAW_COLUMNS  # noqa: E402


def test_every_raw_column_has_a_promql_query():
    qs = {f"{c}.{m}": collect.promql(c, m) for c in COMPONENTS for m in METRICS[c]}
    assert list(qs) == RAW_COLUMNS
    assert all(q and "{" in q or "(" in q for q in qs.values())


def test_absent_series_default_to_zero_but_gauges_do_not():
    assert collect.promql("order", "err_rate").endswith("or vector(0)")
    assert collect.promql("postgres", "locks").endswith("or vector(0)")
    assert "or vector(0)" not in collect.promql("order", "cpu")      # a missing CPU series must stay NaN


def test_database_up_ignores_exporter_connectivity():
    assert "pg_up" not in collect.promql("postgres", "up")           # exporter loses its connection during slowdowns
    assert 'job="auth"' in collect.promql("auth", "up")


def test_log_parsers():
    j = collect._parse("order", 1.0, '{"ts": 1, "service": "order", "level": "ERROR", "msg": "upstream payment timed out after 1000ms"}')
    assert j == (1.0, "order", "ERROR", "upstream payment timed out after 1000ms")
    pg = collect._parse("postgres", 2.0, "2026-10-04 09:40:12.123 UTC [64] ERROR:  canceling statement due to lock timeout")
    assert pg[2] == "ERROR" and "lock timeout" in pg[3]
    rd = collect._parse("redis", 3.0, "1:M 04 Oct 2026 09:34:14.070 # Warning: no config file specified")
    assert rd[2] == "WARN" and rd[3].startswith("Warning")
    assert collect._parse("gateway", 4.0, "   ") is None


def test_injector_plan_types_match_simulator_catalogue():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testbed" / "injector"))
    import injector
    from opsmind.config import FAULTS
    assert set(injector.FAULTS) == set(FAULTS)
