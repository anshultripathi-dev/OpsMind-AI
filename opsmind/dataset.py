"""Dataset construction: simulate the campaign, parse logs, build the labelled feature table."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, DATA_DIR, RAW_COLUMNS, RUN_SEC, SAMPLE_SEC, SLO_ERR_RATE, SLO_LATENCY_MS
from .features import (EXTRA_FEATURES, METRIC_FEATURES, extra_window_features, ffill, label_windows, log_window_features,
                       metric_window_features, window_ends)
from .logs import LogParser, LogVocab
from .sim.scenario import RunData, make_campaign, simulate_run


def assign_splits(specs, train: float = 0.60, val: float = 0.15) -> dict[str, str]:
    """Chronological split *within each fault type*, always by whole experiment run."""
    groups: dict[str, list] = {}
    for s in specs:
        groups.setdefault(s.fault["type"] if s.fault else "normal", []).append(s)
    out = {}
    for _, lst in groups.items():
        lst.sort(key=lambda s: s.index)
        n = len(lst)
        a, b = int(round(n * train)), int(round(n * (train + val)))
        for i, s in enumerate(lst):
            out[s.run_id] = "train" if i < a else "val" if i < b else "test"
    return out


def run_table(metrics: np.ndarray, ids, truth: dict, run_id: str, kind: str, split: str, vocab: LogVocab,
              load_norm: bool = False) -> pd.DataFrame:
    """Labelled feature rows (one per 30 s window) for one run; shared by simulated and real data."""
    names = METRIC_FEATURES + (EXTRA_FEATURES if load_norm else []) + vocab.names
    n_steps = metrics.shape[0]
    ends = window_ends(n_steps)
    Xm = metric_window_features(ffill(metrics))
    parts = [Xm] + ([extra_window_features(Xm)] if load_norm else []) + [log_window_features(vocab.step_counts(ids, n_steps))]
    X = np.concatenate(parts, axis=1)
    lab = label_windows(truth, ends)
    meta = pd.DataFrame({
        "run_id": run_id, "split": split, "kind": kind, "w": np.arange(len(ends)), "tw": lab["tw"],
        "anomaly": lab["anomaly"].astype(np.int8), "grace": lab["grace"].astype(np.int8),
        "warn": lab["warn"].astype(np.int8), "warn_mask": lab["warn_mask"].astype(np.int8),
        "active": lab["active"].astype(np.int8),
        "fault_type": np.where(lab["anomaly"], truth["fault_type"] or "normal", "normal"),
        "target": np.where(lab["anomaly"], truth["target"] or "", ""),
    })
    return pd.concat([meta, pd.DataFrame(X, columns=names)], axis=1)


def generate(n_per_fault: int = 35, n_normal: int = 44, seed: int = 7, verbose: bool = True) -> dict:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    t0 = time.time()
    specs = make_campaign(n_per_fault, n_normal, seed)
    splits = assign_splits(specs)
    log(f"[1/5] simulating {len(specs)} runs ({sum(s.kind == 'fault' for s in specs)} fault episodes, "
        f"{sum(s.kind == 'normal' for s in specs)} normal runs)")
    runs: list[RunData] = [simulate_run(s) for s in specs]

    # ---- Drain: learn templates on the training split, then match everything -------
    log("[2/5] parsing logs with Drain")
    train_msgs = [m for r in runs if splits[r.spec.run_id] == "train" for _, _, _, m in r.logs]
    parser = LogParser().fit(train_msgs)
    templates = parser.templates()
    cache: dict[str, int] = {}

    def match(m: str) -> int:
        v = cache.get(m)
        if v is None:
            v = cache[m] = parser.match(m)
        return v

    pair_counts: Counter = Counter()
    run_log_ids: list[list[tuple[float, str, str, int]]] = []
    for r in runs:
        rows = [(t, svc, lv, match(m)) for t, svc, lv, m in r.logs]
        run_log_ids.append(rows)
        if splits[r.spec.run_id] == "train":
            pair_counts.update((svc, tid) for _, svc, _, tid in rows)
    vocab = LogVocab.build(pair_counts, templates, min_count=40)
    parser.save(ARTIFACT_DIR / "drain.pkl")
    pd.to_pickle(vocab, ARTIFACT_DIR / "log_vocab.pkl")
    log(f"      {len(templates)} templates, {len(vocab.pairs)} (service, template) log features")

    # ---- raw tables ----------------------------------------------------------------
    log("[3/5] writing raw metrics and logs")
    mrows, lrows = [], []
    for r, ids in zip(runs, run_log_ids):
        df = pd.DataFrame(r.metrics, columns=RAW_COLUMNS)
        df.insert(0, "t", np.arange(len(df)) * SAMPLE_SEC)
        df.insert(0, "run_id", r.spec.run_id)
        mrows.append(df)
        ld = pd.DataFrame(r.logs, columns=["t", "service", "level", "msg"])
        ld["template_id"] = [x[3] for x in ids]
        ld.insert(0, "run_id", r.spec.run_id)
        lrows.append(ld)
    pd.concat(mrows, ignore_index=True).to_parquet(DATA_DIR / "raw_metrics.parquet", index=False)
    pd.concat(lrows, ignore_index=True).to_parquet(DATA_DIR / "raw_logs.parquet", index=False)

    # ---- feature table ---------------------------------------------------------------
    log("[4/5] feature engineering (30 s windows, 10 s stride)")
    names = METRIC_FEATURES + vocab.names
    frames, runs_meta = [], []
    for r, ids in zip(runs, run_log_ids):
        frames.append(run_table(r.metrics, ids, r.truth, r.spec.run_id, r.spec.kind, splits[r.spec.run_id], vocab))
        truth = dict(r.truth)
        truth["split"] = splits[r.spec.run_id]
        runs_meta.append(truth)
    feats = pd.concat(frames, ignore_index=True)
    feats.to_parquet(DATA_DIR / "features.parquet", index=False)

    with open(DATA_DIR / "runs.json", "w") as fh:
        json.dump(runs_meta, fh, indent=1)
    with open(DATA_DIR / "log_templates.json", "w") as fh:
        json.dump({str(k): v for k, v in templates.items()}, fh, indent=1)

    faults = [m for m in runs_meta if m["fault_type"]]
    impacted = [m for m in faults if m["impact_start"] is not None]
    stats = dict(
        runs=len(runs_meta), fault_episodes=len(faults), normal_runs=len(runs_meta) - len(faults),
        normal_hours=round((len(runs_meta) - len(faults)) * RUN_SEC / 3600, 1),
        total_hours=round(len(runs_meta) * RUN_SEC / 3600, 1),
        windows=int(len(feats)), n_features=len(names), n_metric_features=len(METRIC_FEATURES),
        n_log_features=len(vocab.names), n_templates=len(templates),
        log_lines=int(sum(len(r.logs) for r in runs)),
        impacted_episodes=len(impacted),
        episodes_by_fault=dict(Counter(m["fault_type"] for m in faults)),
        impacted_by_fault=dict(Counter(m["fault_type"] for m in impacted)),
        windows_by_split=feats["split"].value_counts().to_dict(),
        runs_by_split=dict(Counter(m["split"] for m in runs_meta)),
        seconds=round(time.time() - t0, 1),
    )
    with open(DATA_DIR / "dataset_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    log(f"[5/5] done in {stats['seconds']} s -> {len(feats):,} windows x {len(names)} features")
    log(f"      impacted (SLO-breaching) episodes: {len(impacted)}/{len(faults)}  by fault: {stats['impacted_by_fault']}")
    return stats


def load_features(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(data_dir / "features.parquet")
    meta_cols = ["run_id", "split", "kind", "w", "tw", "anomaly", "grace", "overload", "warn", "warn_mask",
                 "active", "fault_type", "target"]
    return df, [c for c in df.columns if c not in meta_cols]


def load_runs(data_dir: Path = DATA_DIR) -> list[dict]:
    with open(data_dir / "runs.json") as fh:
        return json.load(fh)


def _splits_for_runs(runs: list[dict], train: float = 0.60, val: float = 0.15) -> dict[str, str]:
    """Chronological per fault type. Healthy control runs are scarce but calibrate the alert budget,
    so they get a larger validation share (50 % / 33 % / 17 %)."""
    groups: dict[tuple, list] = {}
    for r in runs:                                   # split within each campaign and fault type, chronologically
        groups.setdefault((r.get("campaign", "c1"), r["fault_type"] or "normal"), []).append(r)
    out = {}
    for (_, ftype), lst in groups.items():
        lst.sort(key=lambda r: r["index"])
        n = len(lst)
        tr_, va_ = (0.5, 0.33) if ftype == "normal" else (train, val)
        a, b = int(round(n * tr_)), int(round(n * (tr_ + va_)))
        for i, r in enumerate(lst):
            out[r["run_id"]] = "train" if i < a else "val" if i < b else "test"
    return out


def build_real_dataset(raw_dir: Path = DATA_DIR / "real", out_dir: Path = DATA_DIR / "real_features",
                       art_dir: Path = ARTIFACT_DIR / "real", verbose: bool = True, load_norm: bool = False,
                       raw_dirs: list[Path] | None = None) -> dict:
    """Turn collector output (real Docker runs) into the labelled feature table used for training.

    Same features, labels and chronological per-fault split as the simulated dataset; Drain is
    re-fitted on the real training logs because real messages differ from simulated ones.
    """
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    out_dir.mkdir(exist_ok=True, parents=True)
    art_dir.mkdir(exist_ok=True, parents=True)
    runs, ms, ls, offset = [], [], [], 0
    for k, d in enumerate(raw_dirs or [raw_dir]):    # one or more collected campaigns, in chronological order
        rs = json.loads((d / "runs.json").read_text())
        for r in rs:
            r["campaign"], r["index"] = f"c{k + 1}", r["index"] + offset
        offset += len(rs)
        runs += rs
        ms.append(pd.read_parquet(d / "raw_metrics.parquet"))
        ls.append(pd.read_parquet(d / "raw_logs.parquet"))
    splits = _splits_for_runs(runs)
    m, l = pd.concat(ms, ignore_index=True), pd.concat(ls, ignore_index=True)
    log(f"{len(runs)} real runs: " + ", ".join(f"{k} {sum(v == k for v in splits.values())}" for k in ("train", "val", "test")))

    parser = LogParser().fit(l[l.run_id.map(lambda r: splits[r] == "train")]["msg"])
    templates = parser.templates()
    cache: dict[str, int] = {}

    def match(msg: str) -> int:
        v = cache.get(msg)
        if v is None:
            v = cache[msg] = parser.match(msg)
        return v

    l = l.assign(template_id=[match(x) for x in l["msg"]])
    tr_logs = l[l.run_id.map(lambda r: splits[r] == "train")]
    vocab = LogVocab.build(Counter(zip(tr_logs["service"], tr_logs["template_id"])), templates, min_count=10)
    parser.save(art_dir / "drain.pkl")
    pd.to_pickle(vocab, art_dir / "log_vocab.pkl")
    log(f"{len(templates)} Drain templates, {len(vocab.pairs)} (service, template) log features")

    frames = []
    for r in runs:
        rm = m[m.run_id == r["run_id"]].sort_values("t")
        rl = l[l.run_id == r["run_id"]]
        ids = list(zip(rl["t"], rl["service"], rl["level"], rl["template_id"]))
        frames.append(run_table(rm[RAW_COLUMNS].to_numpy(np.float32), ids, r, r["run_id"], r["kind"], splits[r["run_id"]], vocab, load_norm))
        r["split"] = splits[r["run_id"]]
    feats = pd.concat(frames, ignore_index=True)
    # Overload: user-facing SLO breached while NO fault is injected (e.g. a load burst the single-CPU gateway
    # cannot absorb). Neither healthy nor an injected fault, so it is excluded from training and scoring.
    breach = (feats["gateway.err_rate__mean"] > SLO_ERR_RATE) | (feats["gateway.lat_p95__mean"] > SLO_LATENCY_MS)
    overload = (breach & (feats["anomaly"] == 0) & (feats["grace"] == 0)).astype(np.int8)
    feats["grace"] = np.maximum(feats["grace"], overload).astype(np.int8)
    feats.insert(feats.columns.get_loc("grace") + 1, "overload", overload)
    log(f"{int(overload.sum())} overload windows ({overload.mean():.1%}) excluded: SLO breached with no fault injected")
    feats.to_parquet(out_dir / "features.parquet", index=False)
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=1))
    (out_dir / "log_templates.json").write_text(json.dumps({str(k): v for k, v in templates.items()}, indent=1))
    faults = [r for r in runs if r["fault_type"]]
    stats = dict(runs=len(runs), fault_episodes=len(faults), normal_runs=len(runs) - len(faults),
                 impacted_episodes=sum(r["impact_start"] is not None for r in faults), windows=int(len(feats)),
                 overload_windows=int(overload.sum()),
                 n_features=int(feats.shape[1] - 12), windows_by_split=feats["split"].value_counts().to_dict(),
                 runs_by_split={k: sum(v == k for v in splits.values()) for k in ("train", "val", "test")})
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"{len(feats):,} real windows; impacting episodes: {stats['impacted_episodes']}/{len(faults)}")
    return stats
