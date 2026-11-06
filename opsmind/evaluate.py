"""Evaluation helpers shared by every module: window metrics and alert/lead-time scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RECOVERY_SEC, STRIDE_SEC


def prf(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4), tp=tp, fp=fp, fn=fn)


def best_f1_threshold(scores: np.ndarray, y: np.ndarray, n_grid: int = 200) -> tuple[float, float]:
    """Threshold on ``scores`` (higher = more anomalous) maximising F1."""
    qs = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, n_grid)))
    best_t, best_f = float(qs[0]), -1.0
    for t in qs:
        f = prf(y, scores >= t)["f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def run_slices(run_ids: np.ndarray) -> list[tuple[str, slice]]:
    """Contiguous (start, stop) per run for an array sorted by run."""
    out, start = [], 0
    for i in range(1, len(run_ids) + 1):
        if i == len(run_ids) or run_ids[i] != run_ids[start]:
            out.append((run_ids[start], slice(start, i)))
            start = i
    return out


def persist_flags(flags: np.ndarray, run_ids: np.ndarray, persist: int) -> np.ndarray:
    """True only when the flag has been raised for ``persist`` consecutive windows (per run)."""
    if persist <= 1:
        return flags.copy()
    out = np.zeros_like(flags, dtype=bool)
    for _, sl in run_slices(run_ids):
        f = flags[sl].astype(int)
        c = np.convolve(f, np.ones(persist, int), mode="full")[: len(f)]
        out[sl] = c >= persist
    return out


def alert_starts(flags: np.ndarray) -> np.ndarray:
    """Indices where a new alert event begins (rising edge of the flag series)."""
    f = np.asarray(flags, bool)
    prev = np.concatenate([[False], f[:-1]])
    return np.flatnonzero(f & ~prev)


def alert_intervals(flags: np.ndarray, tw: np.ndarray) -> list[tuple[float, float]]:
    """Alert events as (start_time, end_time) — an alert stays open while the flag stays raised."""
    f = np.asarray(flags, bool)
    out, start = [], None
    for i, v in enumerate(f):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((float(tw[start]), float(tw[i - 1])))
            start = None
    if start is not None:
        out.append((float(tw[start]), float(tw[-1])))
    return out


def _overload_filter(df: pd.DataFrame, sl: slice):
    """Alert events touching an 'overload' window (SLO breached with no fault injected) are not false alarms."""
    if "overload" not in df.columns:
        return lambda a, b: False
    ov, t = df["overload"].to_numpy()[sl], df["tw"].to_numpy()[sl]
    return lambda a, b: bool(ov[(t >= a) & (t <= b)].any())


def alert_evaluation(df: pd.DataFrame, flags: np.ndarray, runs: dict[str, dict], persist: int = 2) -> dict:
    """Episode-level evaluation of an alert stream (early warning *or* threshold alert).

    ``df`` rows are windows (sorted by run); ``flags`` are per-window booleans.
    * true warning : an alert is open at some point in [fault start, user impact)
    * late         : the first alert opens only after the impact began
    * false alarm  : alert event that never overlaps [fault start, fault end + recovery]
    Lead time = impact start - max(alert start, fault start).
    """
    run_ids = df["run_id"].to_numpy()
    tw = df["tw"].to_numpy()
    fl = persist_flags(np.asarray(flags, bool), run_ids, persist)
    eligible = tp = late = missed = fp = 0
    leads: list[float] = []
    per_fault: dict[str, dict] = {}
    episodes = []
    quiet = ((df["anomaly"].to_numpy() == 0) & (df["grace"].to_numpy() == 0)).sum()
    for rid, sl in run_slices(run_ids):
        truth = runs[rid]
        events = alert_intervals(fl[sl], tw[sl])
        overloaded = _overload_filter(df, sl)
        if not truth["fault_type"]:
            fp += sum(1 for a, b in events if not overloaded(a, b))
            continue
        fs, fe, imp = truth["fault_start"], truth["fault_end"], truth["impact_start"]
        horizon_end = fe + RECOVERY_SEC
        fp += sum(1 for a, b in events if (b < fs or a > horizon_end) and not overloaded(a, b))
        if imp is None:
            continue
        eligible += 1
        pf = per_fault.setdefault(truth["fault_type"], dict(n=0, tp=0, leads=[], late=0))
        pf["n"] += 1
        pre = [(a, b) for a, b in events if a < imp and b >= fs]
        lead, status = None, "missed"
        if pre:
            tp += 1
            lead = float(imp - max(pre[0][0], fs))
            leads.append(lead)
            pf["tp"] += 1
            pf["leads"].append(lead)
            status = "warned"
        else:
            after = [(a, b) for a, b in events if a >= imp and a <= horizon_end]
            if after:
                late += 1
                pf["late"] += 1
                status = "late"
                lead = float(imp - after[0][0])        # negative = detected after impact
            else:
                missed += 1
        episodes.append(dict(run_id=rid, status=status, lead=lead))
    hours = max(quiet * STRIDE_SEC / 3600.0, 1e-9)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / eligible if eligible else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        eligible_episodes=eligible, warned=tp, late=late, missed=missed, false_alarms=int(fp),
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
        false_alarms_per_hour=round(fp / hours, 3),
        mean_lead_s=round(float(np.mean(leads)), 1) if leads else 0.0,
        median_lead_s=round(float(np.median(leads)), 1) if leads else 0.0,
        leads=[round(x, 1) for x in leads],
        per_fault={k: dict(n=v["n"], recall=round(v["tp"] / v["n"], 3) if v["n"] else 0.0,
                           mean_lead_s=round(float(np.mean(v["leads"])), 1) if v["leads"] else 0.0,
                           late=v["late"]) for k, v in per_fault.items()},
        episodes=episodes,
    )


def detection_delay(df: pd.DataFrame, flags: np.ndarray, runs: dict[str, dict], persist: int = 3) -> dict:
    """Time from injected fault start to the first sustained alert (anomaly detectors)."""
    run_ids, tw = df["run_id"].to_numpy(), df["tw"].to_numpy()
    fl = persist_flags(np.asarray(flags, bool), run_ids, persist)
    delays, detected, total, fp = [], 0, 0, 0
    quiet = ((df["anomaly"].to_numpy() == 0) & (df["grace"].to_numpy() == 0)).sum()
    for rid, sl in run_slices(run_ids):
        truth = runs[rid]
        events = alert_intervals(fl[sl], tw[sl])
        overloaded = _overload_filter(df, sl)
        if not truth["fault_type"]:
            fp += sum(1 for a, b in events if not overloaded(a, b))
            continue
        fs, fe = truth["fault_start"], truth["fault_end"]
        fp += sum(1 for a, b in events if (b < fs or a > fe + RECOVERY_SEC) and not overloaded(a, b))
        total += 1
        hit = [(a, b) for a, b in events if a <= fe and b >= fs]
        if hit:
            detected += 1
            delays.append(max(hit[0][0] - fs, 0.0))
    hours = max(quiet * STRIDE_SEC / 3600.0, 1e-9)
    return dict(episodes=total, detected=detected, episode_recall=round(detected / total, 4) if total else 0.0,
                mean_delay_s=round(float(np.mean(delays)), 1) if delays else 0.0,
                false_alarms=int(fp), false_alarms_per_hour=round(fp / hours, 3))
