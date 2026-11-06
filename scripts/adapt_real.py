"""Calibrate 'normal' on REAL telemetry and evaluate with leave-one-run-out (no run scores itself).

    python scripts/adapt_real.py

Compares, on the real Docker runs: static thresholds | simulator-trained detector (zero-shot) |
PCA-reconstruction detector fit on real healthy windows of the OTHER runs. The decision rule is fixed
in advance (no tuning on these few runs): alert when the reconstruction error exceeds 1.25x the
largest error seen on the training-normal windows. Root cause uses the same z-evidence + dependency
graph with the real-normal scaler. Writes artifacts/real_adapt.json.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from opsmind.baselines import ThresholdBaseline  # noqa: E402
from opsmind.config import ARTIFACT_DIR, COMPONENTS  # noqa: E402
from opsmind.evaluate import detection_delay, prf  # noqa: E402
from opsmind.features import NormalScaler  # noqa: E402
from opsmind.real import real_windows  # noqa: E402
from opsmind.runtime import OpsMindRuntime  # noqa: E402

rt = OpsMindRuntime()
X, df, runs, ticks = real_windows(rt)
run_ids = df.run_id.to_numpy()
normal = ((df.anomaly == 0) & (df.grace == 0) & ((df.pre_fault == 1) | df.run_id.map(lambda r: runs[r]["fault_type"] is None))).to_numpy()
ok = (df.grace == 0).to_numpy()
print(f"{len(df)} real windows, {int(normal.sum())} healthy windows usable for calibration")

flags_ml = np.zeros(len(df), bool)
spe_all = np.zeros(len(df))
Zreal = np.zeros_like(X)
for rid in runs:
    te = run_ids == rid
    tr = ~te & normal
    sc = NormalScaler().fit(X[tr], X[~te])
    Z = sc.transform(X)
    Zreal[te] = Z[te]
    pca = PCA(n_components=8, random_state=0).fit(Z[tr])
    spe = lambda A: np.mean((A - pca.inverse_transform(pca.transform(A))) ** 2, axis=1)
    thr = 1.25 * spe(Z[tr]).max()
    spe_all[te] = spe(Z[te])
    flags_ml[te] = spe_all[te] >= thr

base = ThresholdBaseline(rt.names).flags(X)
zero = np.array([k["anomaly"]["flagged"] for r in runs for k in ticks[r]])
y = df.anomaly.to_numpy()

def det(flags):
    return {**prf(y[ok], flags[ok]), **{k: v for k, v in detection_delay(df, flags, runs, 3).items() if k != "episodes"}}

out = dict(windows=int(len(df)), healthy_windows_for_calibration=int(normal.sum()), protocol="leave-one-run-out, rule fixed in advance",
           detection=dict(static_thresholds=det(base), simulator_zero_shot=det(zero), real_calibrated=det(flags_ml)))

# root cause with real-normal z-scores + dependency graph (no learned classifier involved)
absz = np.abs(Zreal)
z_c = np.stack([np.sort(absz[:, rt.comp_cols[c]], axis=1)[:, -5:].mean(1) for c in range(len(COMPONENTS))], axis=1)
params = dict(rt.rca.params, alpha=0.0)
scores = rt.rca.combine(np.ones_like(z_c), z_c, params)
hits = []
for rid, t in runs.items():
    if not t["fault_type"]:
        continue
    m = (run_ids == rid) & (y == 1)
    order = np.argsort(-scores[m].mean(0))[:3]
    top = [COMPONENTS[i] for i in order]
    hits.append(dict(run=rid, fault=t["fault_type"], target=t["target"], top3=top, top1=top[0] == t["target"], in3=t["target"] in top))
out["rootcause_real_calibrated"] = dict(top1=round(float(np.mean([h["top1"] for h in hits])), 3), top3=round(float(np.mean([h["in3"] for h in hits])), 3),
                                        n=len(hits), episodes=hits)
(ARTIFACT_DIR / "real_adapt.json").write_text(json.dumps(out, indent=1, default=float))
for k, v in out["detection"].items():
    print(f"{k:22s} F1 {v['f1']:.3f}  P {v['precision']:.3f}  R {v['recall']:.3f}  episodes {v['detected']}/6  FA/h {v['false_alarms_per_hour']}  delay {v['mean_delay_s']}s")
print("root cause (real-calibrated z + graph): top1", out["rootcause_real_calibrated"]["top1"], "top3", out["rootcause_real_calibrated"]["top3"])
for h in hits:
    print("  ", h["run"], h["fault"], "->", h["target"], "| ranked", h["top3"])
