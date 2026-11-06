"""Train/serve consistency and end-to-end smoke tests (requires dataset + trained artifacts)."""
import numpy as np
import pandas as pd
import pytest

from opsmind.config import ARTIFACT_DIR, DATA_DIR, RAW_COLUMNS

pytestmark = pytest.mark.skipif(not (ARTIFACT_DIR / "bundle.joblib").exists(), reason="run train.py first")


@pytest.fixture(scope="module")
def rt():
    from opsmind.runtime import OpsMindRuntime
    return OpsMindRuntime()


def test_online_features_match_offline(rt):
    """The windows computed from a raw scrape stream equal the offline feature table."""
    from opsmind.sources import ReplaySource
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    run = feats[(feats.split == "test") & (feats.kind == "fault")].run_id.iloc[0]
    off = feats[feats.run_id == run].sort_values("w")
    src, st = ReplaySource(run), rt.new_state()
    vecs = []
    while True:
        s = src.step()
        if s is None:
            break
        t, row, logs, truth = s
        # replay raw log messages are re-parsed by Drain exactly as in serving
        x = np.array([row[c] for c in RAW_COLUMNS], np.float32)
        tick = rt.ingest(st, t, row, logs, truth)
        if tick is not None:
            vecs.append(rt.window_vector(st))
    on = np.stack(vecs)
    cols = [c for c in rt.names if not c.startswith("log.")]
    a = off[cols].to_numpy()[: len(on)]
    b = on[:, : len(cols)]
    # NaN scrapes are forward-filled online and offline alike
    assert np.allclose(a, b, atol=1e-3, rtol=1e-4)


def test_fault_is_detected_and_ranked(rt):
    from opsmind.sources import SimulatorSource
    src, st = SimulatorSource(seed=123), rt.new_state()
    for _ in range(90):
        t, row, logs, truth = src.step()
        rt.ingest(st, t, row, logs, truth)
    src.inject("db_slowdown", "postgres", duration=420, intensity=0.9)
    ticks = []
    for _ in range(120):
        t, row, logs, truth = src.step()
        tick = rt.ingest(st, t, row, logs, truth)
        if tick:
            ticks.append(tick)
    assert any(k["anomaly"]["sustained"] for k in ticks), "fault never detected"
    ranked = [k["rca"][0]["component"] for k in ticks if k["anomaly"]["sustained"] and k["rca"]]
    assert ranked and ranked[-1] == "postgres"
    opened = [k["incident"]["current"] for k in ticks if k["incident"]["current"]]
    assert opened and opened[-1]["type"] == "db_slowdown"
    assert ticks[-1]["incident"]["history"], "incident should be resolved and archived after the fault ends"
