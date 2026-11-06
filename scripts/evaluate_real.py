"""Zero-shot check of the simulator-trained pipeline on REAL Docker-testbed telemetry.

    python scripts/evaluate_real.py            # after: collect.py wrote data/real/{raw_metrics,raw_logs}.parquet

Every real run is streamed through the same online path the dashboard uses (OpsMindRuntime.ingest);
labels come only from the injector log. Writes artifacts/real_eval.json.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opsmind.baselines import ThresholdBaseline  # noqa: E402,F401
from opsmind.config import ARTIFACT_DIR, COMPONENTS, DATA_DIR, FAULT_TYPES, SAMPLE_SEC  # noqa: E402
from opsmind.evaluate import alert_evaluation, detection_delay, prf  # noqa: E402
from opsmind.features import label_windows, window_ends  # noqa: E402
from opsmind.runtime import OpsMindRuntime  # noqa: E402
from opsmind.sources import ReplaySource  # noqa: E402


def main() -> dict:
    runs = {r["run_id"]: r for r in json.load(open(DATA_DIR / "real" / "runs.json"))}
    rt = OpsMindRuntime()
    rows, per_run = [], {}
    for rid, truth in runs.items():
        src, st, ticks = ReplaySource(rid), rt.new_state(), []
        while (s := src.step()) is not None:
            t, row, logs, tr = s
            k = rt.ingest(st, t, row, logs, tr)
            if k is not None:
                ticks.append(k)
        ends = window_ends(len(src.metrics))
        assert len(ends) == len(ticks), (len(ends), len(ticks))
        lab = label_windows(truth, ends)
        d = pd.DataFrame(dict(run_id=rid, tw=lab["tw"], anomaly=lab["anomaly"].astype(int), grace=lab["grace"].astype(int),
                              active=lab["active"].astype(int),
                              ml=[k["anomaly"]["flagged"] for k in ticks], base=[bool(k["baseline"]["firing"]) for k in ticks],
                              risk=[k["risk"]["prob"] for k in ticks], fault=[k["classifier"]["likely_fault"] for k in ticks],
                              incident_label=[k["classifier"]["label"] for k in ticks]))
        rows.append(d)
        per_run[rid] = ticks
    df = pd.concat(rows, ignore_index=True)
    ok = df["grace"] == 0
    out: dict = dict(runs=len(runs), fault_episodes=sum(1 for r in runs.values() if r["fault_type"]),
                     impacted=sum(1 for r in runs.values() if r["impact_start"] is not None),
                     windows=int(len(df)), note="Zero-shot: models trained on simulator data only; labels from the injector log.")
    out["detection"] = {
        "static_thresholds": {**prf(df.anomaly[ok], df.base[ok]), **{k: v for k, v in detection_delay(df, df.base.to_numpy(), runs, 3).items() if k != "episodes"}},
        "opsmind_detector": {**prf(df.anomaly[ok], df.ml[ok]), **{k: v for k, v in detection_delay(df, df.ml.to_numpy(), runs, 3).items() if k != "episodes"}},
    }
    thr = rt.ew_thr
    ew = alert_evaluation(df, (df.risk >= thr).to_numpy(), runs, 2)
    bw = alert_evaluation(df, df.base.to_numpy(), runs, 2)
    keep = ("eligible_episodes", "warned", "late", "missed", "false_alarms", "false_alarms_per_hour", "mean_lead_s", "recall", "precision")
    out["early_warning"] = dict(opsmind={k: ew[k] for k in keep}, static_thresholds={k: bw[k] for k in keep})
    # classification and root cause on fault windows
    fa = df[df.anomaly == 1]
    out["classification"] = dict(
        windows=int(len(fa)),
        accuracy=round(float((fa.fault == pd.Series([runs[r]["fault_type"] for r in fa.run_id], index=fa.index)).mean()), 4) if len(fa) else None)
    per_fault, rca_eps = {}, []
    for rid, truth in runs.items():
        if not truth["fault_type"]:
            continue
        w = df[(df.run_id == rid) & (df.anomaly == 1)]
        pred = w.fault.mode().iloc[0] if len(w) else None
        sc = np.zeros(len(COMPONENTS)); n = 0
        for i in w.index:
            k = per_run[rid][int(df.loc[i].name - df.index[df.run_id == rid][0])]
            if k["rca"]:
                for r in k["rca"]:
                    sc[COMPONENTS.index(r["component"])] += r["score"]
                n += 1
        order = [COMPONENTS[i] for i in np.argsort(-sc)[:3]] if n else []
        per_fault[rid] = dict(fault=truth["fault_type"], target=truth["target"], classified=pred, rca_top3=order,
                              top1=bool(order and order[0] == truth["target"]), top3=truth["target"] in order,
                              impact_start=truth["impact_start"], intensity=truth["intensity"])
        rca_eps.append(per_fault[rid])
    out["rootcause"] = dict(top1=round(float(np.mean([e["top1"] for e in rca_eps])), 4), top3=round(float(np.mean([e["top3"] for e in rca_eps])), 4), n=len(rca_eps))
    out["episodes"] = per_fault
    (ARTIFACT_DIR / "real_eval.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=float))
