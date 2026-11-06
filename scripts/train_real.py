"""Train and evaluate OpsMind on REAL Docker-testbed runs.

    python scripts/train_real.py      # after: make real-collect

1. builds the labelled feature table from data/real (same features/labels as the simulator data,
   chronological per-fault split by whole run, Drain re-fitted on real logs),
2. trains every module on the real training runs and evaluates on the real test runs
   -> artifacts/real/results.json (Model lab "Real data" view) and artifacts/real/bundle.joblib,
3. scores the simulator-trained models on the SAME real test runs (zero-shot) for a fair comparison
   -> artifacts/real/sim_vs_real.json.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from opsmind.config import ARTIFACT_DIR, DATA_DIR  # noqa: E402
from opsmind.dataset import build_real_dataset  # noqa: E402
from opsmind.evaluate import alert_evaluation, detection_delay, prf  # noqa: E402
from opsmind.pipeline import train_all  # noqa: E402
from opsmind.real import real_windows  # noqa: E402
from opsmind.runtime import OpsMindRuntime  # noqa: E402

FEAT, ART = DATA_DIR / "real_features", ARTIFACT_DIR / "real"


def main() -> None:
    print("== 1/3 building real feature table")
    dirs = [d for d in (DATA_DIR / "real", DATA_DIR / "real2") if (d / "runs.json").exists()]
    print("campaigns:", [d.name for d in dirs])
    build_real_dataset(raw_dirs=dirs, out_dir=FEAT, art_dir=ART)
    print("== 2/3 training all modules on real runs")
    res = train_all(data_dir=FEAT, artifact_dir=ART)

    print("== 3/3 simulator-trained models on the same real test runs")
    runs = {r["run_id"]: r for r in json.load(open(FEAT / "runs.json"))}
    X, df, _, ticks = real_windows(OpsMindRuntime())
    # score under the same rules as training: SLO-breaching windows with no fault injected are excluded
    import pandas as pd
    meta = pd.read_parquet(FEAT / "features.parquet", columns=["run_id", "grace", "overload"])
    assert (meta["run_id"].to_numpy() == df["run_id"].to_numpy()).all()
    df["grace"], df["overload"] = meta["grace"].to_numpy(), meta["overload"].to_numpy()
    test_ids = [r for r, v in runs.items() if v["split"] == "test"]
    keep = df.run_id.isin(test_ids).to_numpy()
    d = df[keep].reset_index(drop=True)
    ml = np.array([k["anomaly"]["flagged"] for r in runs for k in ticks[r] if r in test_ids])
    risk = np.array([k["risk"]["prob"] for r in runs for k in ticks[r] if r in test_ids])
    from opsmind.baselines import ThresholdBaseline
    base = ThresholdBaseline(OpsMindRuntime().names).flags(X[keep])
    ok = (d.grace == 0).to_numpy()
    y = d.anomaly.to_numpy()

    def det(f):
        return {**prf(y[ok], f[ok]), **{k: v for k, v in detection_delay(d, f, runs, 3).items() if k != "episodes"}}

    rt = OpsMindRuntime()
    out = dict(test_runs=len(test_ids), test_fault_episodes=sum(1 for r in test_ids if runs[r]["fault_type"]),
               detection=dict(static_thresholds=det(base), simulator_zero_shot=det(ml)),
               early_warning=dict(
                   static_thresholds={k: v for k, v in alert_evaluation(d, base, runs, 2).items() if k in ("warned", "late", "missed", "eligible_episodes", "mean_lead_s", "false_alarms_per_hour")},
                   simulator_zero_shot={k: v for k, v in alert_evaluation(d, risk >= rt.ew_thr, runs, 2).items() if k in ("warned", "late", "missed", "eligible_episodes", "mean_lead_s", "false_alarms_per_hour")}))
    (ART / "sim_vs_real.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["detection"], indent=1, default=float))


if __name__ == "__main__":
    main()
