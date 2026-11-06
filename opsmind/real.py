"""Windows of REAL Docker-testbed telemetry, built through the exact online path used for serving."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .features import label_windows, window_ends
from .runtime import OpsMindRuntime
from .sources import ReplaySource


def real_windows(rt: OpsMindRuntime | None = None):
    """Returns (X raw features, meta DataFrame, runs dict, ticks per run)."""
    rt = rt or OpsMindRuntime()
    merged = DATA_DIR / "real_features" / "runs.json"
    runs = {r["run_id"]: r for r in json.load(open(merged if merged.exists() else DATA_DIR / "real" / "runs.json"))}
    X, metas, ticks_by_run = [], [], {}
    for rid, truth in runs.items():
        src, st, ticks = ReplaySource(rid), rt.new_state(), []
        while (s := src.step()) is not None:
            t, row, logs, tr = s
            k = rt.ingest(st, t, row, logs, tr)
            if k is not None:
                ticks.append(k)
                X.append(rt.window_vector(st))
        lab = label_windows(truth, window_ends(len(src.metrics)))
        metas.append(pd.DataFrame(dict(run_id=rid, tw=lab["tw"], anomaly=lab["anomaly"].astype(int), grace=lab["grace"].astype(int),
                                       warn=lab["warn"].astype(int), warn_mask=lab["warn_mask"].astype(int),
                                       pre_fault=((lab["tw"] <= (truth["fault_start"] or 1e12)) & ~lab["grace"]).astype(int))))
        ticks_by_run[rid] = ticks
    return np.stack(X).astype(np.float32), pd.concat(metas, ignore_index=True), runs, ticks_by_run
