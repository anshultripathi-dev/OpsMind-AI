"""End-to-end training and evaluation of all OpsMind modules.

``python scripts/train.py`` runs this after the dataset has been generated and writes
``artifacts/bundle.joblib``, ``artifacts/lstm.pt`` and ``artifacts/results.json``
(the latter feeds the dashboard's Model Lab).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from . import __version__
from .baselines import ThresholdBaseline
from .config import ARTIFACT_DIR, CLASSES, COMPONENTS, DATA_DIR, FAULT_TYPES, SEQ_LEN, SEQ_STEP_WINDOWS
from .dataset import load_features, load_runs
from .evaluate import (alert_evaluation, best_f1_threshold, detection_delay, persist_flags, prf, run_slices)
from .features import NormalScaler, sequence_index
from .logs import LogVocab
from .models import anomaly as anom
from .models import classify as clf_mod
from .models import cluster as clu
from .models import dimred
from .models import earlywarning as ew
from .models.rootcause import RootCauseRanker, topk_accuracy
from .naming import fault_label

SEED = 0


def _curve(y, s, kind="pr", n=60):
    if kind == "pr":
        p, r, _ = precision_recall_curve(y, s)
        x, yv = r, p
    else:
        fpr, tpr, _ = roc_curve(y, s)
        x, yv = fpr, tpr
    sel = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [[round(float(x[i]), 4), round(float(yv[i]), 4)] for i in sel]


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    return o


def pick_alert_threshold(prob, df, runs, target_fa: float = 1.0, persist: int = 2) -> float:
    qs = np.unique(np.quantile(prob, np.linspace(0.80, 0.9998, 90)))
    best, fallback = None, None
    for t in qs:
        ev = alert_evaluation(df, prob >= t, runs, persist)
        key = (ev["recall"], ev["precision"])
        if ev["false_alarms_per_hour"] <= target_fa and (best is None or key > best[0]):
            best = (key, float(t))
        if fallback is None or ev["false_alarms_per_hour"] < fallback[0]:
            fallback = (ev["false_alarms_per_hour"], float(t))
    return best[1] if best else fallback[1]


def pick_detector_threshold(scores, y, ok, df, runs, max_fa: float = 0.5) -> float:
    """Best-F1 threshold subject to an alert budget (sustained false alarms per hour, on validation)."""
    qs = np.unique(np.quantile(scores[ok], np.linspace(0.5, 0.9998, 120)))
    best, fallback = None, None
    for t in qs:
        flags = scores >= t
        f1 = prf(y[ok], flags[ok])["f1"]
        fa = detection_delay(df, flags, runs, 3)["false_alarms_per_hour"]
        if fa <= max_fa and (best is None or f1 > best[0]):
            best = (f1, float(t))
        if fallback is None or fa < fallback[0]:
            fallback = (fa, float(t))
    return best[1] if best else fallback[1]


ALERT_BUDGET_FA_PER_HOUR = 0.5


def train_all(log=lambda *a: print(*a, flush=True), data_dir=DATA_DIR, artifact_dir=ARTIFACT_DIR) -> dict:
    t_start = time.time()
    artifact_dir.mkdir(exist_ok=True, parents=True)
    df, names = load_features(data_dir)
    runs = {r["run_id"]: r for r in load_runs(data_dir)}
    vocab: LogVocab = pd.read_pickle(artifact_dir / "log_vocab.pkl")
    templates = vocab.templates
    X = df[names].to_numpy(np.float32)
    split = df["split"].to_numpy()
    anomaly = df["anomaly"].to_numpy().astype(int)
    grace = df["grace"].to_numpy().astype(int)
    ftype = df["fault_type"].to_numpy()
    target = df["target"].to_numpy()
    run_ids = df["run_id"].to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    df_va, df_te = df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    log(f"windows: train {len(tr):,}  val {len(va):,}  test {len(te):,}   features: {len(names)}")

    normal_tr = tr[(anomaly[tr] == 0) & (grace[tr] == 0)]
    scaler = NormalScaler().fit(X[normal_tr], X[tr])
    Z = scaler.transform(X)
    res: dict = {"meta": dict(version=__version__, trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              n_features=len(names), seed=SEED)}

    # ------------------------------------------------------------------ 2. dimensionality reduction
    log("[M2] PCA / random projection")
    pca_all = dimred.fit_pca(Z[tr], 0.95, 64, SEED)
    fault_tr = tr[anomaly[tr] == 1]
    fault_te = te[anomaly[te] == 1]
    y_ft = np.array([FAULT_TYPES.index(f) for f in ftype[fault_tr]])
    y_fe = np.array([FAULT_TYPES.index(f) for f in ftype[fault_te]])
    pca2 = PCA(2, random_state=SEED).fit(Z[tr])
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(te, min(3500, len(te)), replace=False))
    emb = pca2.transform(Z[sample])
    res["dimred"] = dict(
        report=dimred.pca_report(Z[tr], SEED),
        chosen_components=int(pca_all.n_components_),
        comparison=dimred.compare_projections(Z[fault_tr], y_ft, Z[fault_te], y_fe, int(pca_all.n_components_), SEED),
        explained_2d=[round(float(v), 4) for v in pca2.explained_variance_ratio_],
        map=[dict(x=round(float(e[0]), 3), y=round(float(e[1]), 3), label=str(ftype[i]),
                  run=str(run_ids[i])) for e, i in zip(emb, sample)],
    )

    # ------------------------------------------------------------------ 3. anomaly detection
    log("[M3] anomaly detection (unsupervised, trained on normal windows only)")
    baseline = ThresholdBaseline(names)
    ok_va, ok_te = grace[va] == 0, grace[te] == 0
    yv, yt = anomaly[va], anomaly[te]
    detectors, scores_all, thr = {}, {}, {}
    rows = []
    for key, cls in anom.DETECTORS.items():
        t0 = time.time()
        det = cls(seed=SEED).fit(Z[normal_tr])
        s_all = det.score(Z)
        detectors[key], scores_all[key] = det, s_all
        thr[key] = pick_detector_threshold(s_all[va], yv, ok_va, df_va, runs, ALERT_BUDGET_FA_PER_HOUR)
        flags = s_all[te] >= thr[key]
        row = dict(model=det.name, key=key, threshold=round(float(thr[key]), 5),
                   **prf(yt[ok_te], flags[ok_te]),
                   auc_roc=round(float(roc_auc_score(yt[ok_te], s_all[te][ok_te])), 4),
                   avg_precision=round(float(average_precision_score(yt[ok_te], s_all[te][ok_te])), 4),
                   **{k: v for k, v in detection_delay(df_te, flags, runs, 3).items() if k != "episodes"},
                   pr_curve=_curve(yt[ok_te], s_all[te][ok_te], "pr"),
                   roc_curve=_curve(yt[ok_te], s_all[te][ok_te], "roc"),
                   per_fault_recall={f: round(float(flags[(ftype[te] == f)].mean()), 3) for f in FAULT_TYPES},
                   fit_seconds=round(time.time() - t0, 1))
        rows.append(row)
        log(f"      {det.name:<26} F1 {row['f1']:.3f}  P {row['precision']:.3f}  R {row['recall']:.3f}  AUC {row['auc_roc']:.3f}  FA/h {row['false_alarms_per_hour']}")
    base_flags = baseline.flags(X[te])
    brow = dict(model="Static thresholds (baseline)", key="threshold", threshold=None, **prf(yt[ok_te], base_flags[ok_te]),
                auc_roc=None, avg_precision=None,
                **{k: v for k, v in detection_delay(df_te, base_flags, runs, 3).items() if k != "episodes"},
                pr_curve=[], roc_curve=[],
                per_fault_recall={f: round(float(base_flags[(ftype[te] == f)].mean()), 3) for f in FAULT_TYPES},
                fit_seconds=0.0)
    rows.insert(0, brow)
    log(f"      {'Static thresholds':<26} F1 {brow['f1']:.3f}  P {brow['precision']:.3f}  R {brow['recall']:.3f}  FA/h {brow['false_alarms_per_hour']}")
    res["anomaly"] = dict(models=rows, primary="iforest", alert_budget_fa_per_hour=ALERT_BUDGET_FA_PER_HOUR)

    # the detector that will be deployed
    primary = max((r for r in rows if r["key"] != "threshold"), key=lambda r: r["f1"])["key"]
    res["anomaly"]["primary"] = primary
    flag_all = scores_all[primary] >= thr[primary]

    # ------------------------------------------------------------------ 4/5. classification + ensembles
    log("[M4/M5] incident classification: single models vs ensembles")
    groups = run_ids[fault_tr]
    va_f = va[anomaly[va] == 1]
    yva_f = np.array([FAULT_TYPES.index(f) for f in ftype[va_f]])
    singles = clf_mod.baseline_models(SEED)
    fit_s, rows_s = clf_mod.fit_and_score(singles, Z[fault_tr], y_ft, Z[va_f], yva_f, Z[fault_te], y_fe, FAULT_TYPES)
    ens = clf_mod.ensemble_models(groups, Z[fault_tr], y_ft, len(FAULT_TYPES), SEED)
    fit_e, rows_e = clf_mod.fit_and_score(ens, Z[fault_tr], y_ft, Z[va_f], yva_f, Z[fault_te], y_fe, FAULT_TYPES)
    for r in rows_s:
        r["family"] = "baseline" if r["model"] == "Majority class" else "single"
        log(f"      {r['model']:<22} macro-F1 {r['macro_f1']:.3f}  ({r['train_seconds']}s)")
    for r in rows_e:
        r["family"] = "ensemble"
        log(f"      {r['model']:<22} macro-F1 {r['macro_f1']:.3f}  ({r['train_seconds']}s)")
    best_single = max((r for r in rows_s if r["family"] == "single"), key=lambda r: r["macro_f1"])
    best_ens = max(rows_e, key=lambda r: -1.0 if r["val_macro_f1"] is None else r["val_macro_f1"])   # no val data: first listed
    res["classification"] = dict(classes=FAULT_TYPES, models=rows_s + rows_e,
                                 best_single=best_single["model"], best_ensemble=best_ens["model"])

    # deployed classifier: XGBoost over [normal + 6 faults]; hard negatives teach it to dismiss benign anomalies
    log("      deployed classifier: XGBoost with a benign/normal class")
    flagged_normal = tr[(anomaly[tr] == 0) & (grace[tr] == 0) & (scores_all[primary][tr] >= thr[primary] * 0.9)]
    rest_normal = rng.choice(normal_tr, min(2500, len(normal_tr)), replace=False)
    rows7 = np.unique(np.concatenate([fault_tr, flagged_normal, rest_normal]))
    y7 = np.array([CLASSES.index(f) for f in ftype[rows7]])
    xgb7 = clf_mod._xgb(SEED, len(CLASSES)).fit(Z[rows7], y7)
    proba7 = xgb7.predict_proba(Z)
    pred7 = proba7.argmax(1)
    # detector -> classifier pipeline on test
    flags_det = flag_all[te]
    flags_pipe = flags_det & (pred7[te] != 0)
    res["classification"]["pipeline"] = dict(
        detector_only=dict(**prf(yt[ok_te], flags_det[ok_te]), **{k: v for k, v in detection_delay(df_te, flags_det, runs, 3).items() if k != "episodes"}),
        detector_plus_classifier=dict(**prf(yt[ok_te], flags_pipe[ok_te]), **{k: v for k, v in detection_delay(df_te, flags_pipe, runs, 3).items() if k != "episodes"}),
    )
    log(f"      detector only F1 {res['classification']['pipeline']['detector_only']['f1']:.3f} FA/h {res['classification']['pipeline']['detector_only']['false_alarms_per_hour']}"
        f"  -> + classifier F1 {res['classification']['pipeline']['detector_plus_classifier']['f1']:.3f} FA/h {res['classification']['pipeline']['detector_plus_classifier']['false_alarms_per_hour']}")

    # ------------------------------------------------------------------ 6. clustering
    log("[M6] clustering flagged windows (no labels used)")
    flagged = np.flatnonzero(flag_all & (grace == 0))
    flagged_fit = flagged[np.isin(split[flagged], ["train", "val"])]
    flagged_te = flagged[split[flagged] == "test"]
    deploy = clu.run_clustering(Z[flagged_fit], ftype[flagged_fit], SEED)
    desc = clu.describe_clusters(Z[flagged_fit], deploy["labels"]["kmeans"], ftype[flagged_fit], names, templates)
    evalc = clu.run_clustering(Z[flagged_te], ftype[flagged_te], SEED)
    cl_pred_te = deploy["kmeans"].predict(deploy["pca"].transform(Z[sample]))
    res["clustering"] = dict(
        algorithms=[dict(model=k, **v) for k, v in evalc["results"].items()],
        deployed_k=deploy["k"], k_scores=deploy["k_scores"], clusters=desc,
        n_flagged_fit=int(len(flagged_fit)), n_flagged_test=int(len(flagged_te)),
        map=[dict(x=round(float(e[0]), 3), y=round(float(e[1]), 3), cluster=int(c), label=str(ftype[i]))
             for e, c, i in zip(emb, cl_pred_te, sample)],
    )
    for k, v in evalc["results"].items():
        log(f"      {k:<22} k={v['n_clusters']:<2} silhouette {v['silhouette']:.3f}  ARI {v['ari']:.3f}  NMI {v['nmi']:.3f}")

    # ------------------------------------------------------------------ 7. early warning
    log("[M7] early-warning LSTM / GRU vs gradient boosting vs threshold alerts")
    pca_seq = PCA(32, random_state=SEED).fit(Z[tr])
    P = pca_seq.transform(Z).astype(np.float32)
    p_mu, p_sd = P[tr].mean(0), P[tr].std(0) + 1e-6
    P = (P - p_mu) / p_sd
    run_start = np.zeros(len(df), int)
    for _, sl in run_slices(run_ids):
        run_start[sl] = sl.start
    idx = sequence_index(run_start, len(df))
    y_w = df["warn"].to_numpy().astype(int)
    m_w = df["warn_mask"].to_numpy().astype(int)

    ew_rows, ew_probs = [], {}
    seq = ew.train_sequence_models(P, idx, y_w, m_w, tr, va, SEED, log)
    lstm, hist, hist_g = seq["lstm"], seq["history_lstm"], seq["history_gru"]
    ew_probs["LSTM"] = seq["probs_lstm"]
    ew_probs["GRU"] = seq["probs_gru"]
    gb = ew.GBForecaster(SEED).fit(P, idx, y_w, tr[m_w[tr] == 1])
    ew_probs["Gradient boosting"] = gb.predict_proba(P, idx)
    ew_probs["Isolation Forest alert"] = (scores_all[primary] - scores_all[primary][normal_tr].mean()) / (scores_all[primary][normal_tr].std() + 1e-9)

    # Platt-scale the learned forecasters on validation so that "risk" is a meaningful probability
    calib = {}
    for name in ("LSTM", "GRU", "Gradient boosting"):
        pr = np.clip(ew_probs[name], 1e-5, 1 - 1e-5)
        lg = np.log(pr / (1 - pr))
        sel = va[m_w[va] == 1]
        cal = LogisticRegression(C=100.0, max_iter=500).fit(lg[sel, None], y_w[sel])
        calib[name] = (float(cal.coef_[0, 0]), float(cal.intercept_[0]))
        ew_probs[name] = cal.predict_proba(lg[:, None])[:, 1]

    mt = m_w[te] == 1
    ew_thr = {}
    for name, prob in ew_probs.items():
        thr_w = pick_alert_threshold(prob[va], df_va, runs, target_fa=1.0)
        ew_thr[name] = thr_w
        ev = alert_evaluation(df_te, prob[te] >= thr_w, runs, persist=2)
        ap = float(average_precision_score(y_w[te][mt], prob[te][mt]))
        ev.update(model=name, threshold=round(float(thr_w), 4), window_ap=round(ap, 4),
                  pr_curve=_curve(y_w[te][mt], prob[te][mt], "pr"))
        ew_rows.append(ev)
        log(f"      {name:<24} recall {ev['recall']:.3f}  precision {ev['precision']:.3f}  lead {ev['mean_lead_s']:.0f}s  FA/h {ev['false_alarms_per_hour']}  AP {ap:.3f}")
    ev_b = alert_evaluation(df_te, base_flags, runs, persist=2)
    ev_b.update(model="Static thresholds (baseline)", threshold=None, window_ap=None, pr_curve=[])
    ew_rows.insert(0, ev_b)
    log(f"      {'Static thresholds':<24} recall {ev_b['recall']:.3f}  precision {ev_b['precision']:.3f}  lead {ev_b['mean_lead_s']:.0f}s  late {ev_b['late']}  FA/h {ev_b['false_alarms_per_hour']}")
    primary_ew = "LSTM"
    res["earlywarning"] = dict(models=ew_rows, primary=primary_ew, history=hist, history_gru=hist_g,
                               horizon_s=120, seq_len=SEQ_LEN, seq_step_s=SEQ_STEP_WINDOWS * 10)

    # ------------------------------------------------------------------ 8. root-cause ranking
    log("[M8] root-cause ranking (SHAP + dependency graph)")
    rca = RootCauseRanker(names, xgb7, templates)

    def fault_cls(rows):
        return proba7[rows][:, 1:].argmax(1) + 1

    def comp_idx(rows):
        return np.array([COMPONENTS.index(t) for t in target[rows]])

    va_a = va[anomaly[va] == 1]
    te_a = te[anomaly[te] == 1]
    ev_va = rca.evidence(Z[va_a], fault_cls(va_a))
    params = rca.tune(ev_va, comp_idx(va_a))
    ev_te = rca.evidence(Z[te_a], fault_cls(te_a))
    truth_te = comp_idx(te_a)
    score_full = rca.combine(ev_te["shap"], ev_te["z"])
    score_shap = ev_te["shap"] / (ev_te["shap"].sum(1, keepdims=True) + 1e-9)
    e_te = 1 - np.exp(-ev_te["z"] / 6.0)
    cpu_cols = [names.index(f"{c}.cpu__mean") for c in COMPONENTS]
    score_cpu = X[te_a][:, cpu_cols]
    methods = {
        "Highest raw CPU (baseline)": score_cpu,
        "Largest z-score deviation": ev_te["z"],
        "SHAP only": score_shap,
        "SHAP + dependency graph": score_full,
    }
    imp = np.array([runs[r]["impact_start"] if runs[r]["impact_start"] is not None else -1.0 for r in run_ids[te_a]])
    tw_a = df["tw"].to_numpy()[te_a]
    rca_rows = []
    for mname, sc in methods.items():
        w_acc = topk_accuracy(sc, truth_te)
        ep_hits1, ep_hits3, n_ep, by_fault = 0, 0, 0, {}
        for rid, _ in run_slices(run_ids[te_a]):
            sel = np.flatnonzero(run_ids[te_a] == rid)
            use = sel[tw_a[sel] >= imp[sel]] if imp[sel][0] >= 0 else sel
            if len(use) == 0:
                use = sel
            rk = np.argsort(-sc[use].mean(0))
            t_idx = truth_te[sel[0]]
            h1, h3 = int(rk[0] == t_idx), int(t_idx in rk[:3])
            ep_hits1 += h1
            ep_hits3 += h3
            n_ep += 1
            b = by_fault.setdefault(runs[rid]["fault_type"], [0, 0, 0])
            b[0] += h1
            b[1] += h3
            b[2] += 1
        rca_rows.append(dict(
            method=mname, window=w_acc,
            episode=dict(top1=round(ep_hits1 / n_ep, 4), top3=round(ep_hits3 / n_ep, 4), n=n_ep),
            by_fault={f: dict(top1=round(v[0] / v[2], 3), top3=round(v[1] / v[2], 3), n=v[2]) for f, v in by_fault.items()}))
        log(f"      {mname:<28} episode top-1 {ep_hits1 / n_ep:.3f}  top-3 {ep_hits3 / n_ep:.3f}   window top-1 {w_acc['top1']:.3f}")
    res["rootcause"] = dict(methods=rca_rows, params=params, components=COMPONENTS)

    # ------------------------------------------------------------------ per-episode table (test split)
    ev_by_model = {r["model"]: {e["run_id"]: e for e in r["episodes"]} for r in ew_rows}
    ep_rows = []
    for rid, sl in run_slices(run_ids[te]):
        tr_ = runs[rid]
        if not tr_["fault_type"]:
            continue
        rows_i = te[sl]
        a_rows = rows_i[anomaly[rows_i] == 1]
        cls_mode = int(np.bincount(fault_cls(a_rows) - 1, minlength=6).argmax()) if len(a_rows) else 0
        if len(a_rows):
            sc = rca.combine(*[v for k, v in rca.evidence(Z[a_rows], fault_cls(a_rows)).items() if k in ("shap", "z")]).mean(0)
            order = [COMPONENTS[i] for i in np.argsort(-sc)[:3]]
        else:
            order = []
        ep_rows.append(dict(
            run_id=rid, fault_type=tr_["fault_type"], target=tr_["target"], intensity=tr_["intensity"],
            fault_start=tr_["fault_start"], fault_end=tr_["fault_end"], impact_start=tr_["impact_start"],
            predicted_type=FAULT_TYPES[cls_mode], rca_top3=order,
            lstm=ev_by_model["LSTM"].get(rid), threshold=ev_by_model["Static thresholds (baseline)"].get(rid),
            gb=ev_by_model["Gradient boosting"].get(rid)))
    res["episodes"] = ep_rows

    # ------------------------------------------------------------------ persist
    log("saving artifacts")
    joblib.dump(dict(
        names=names, load_norm=any(n.endswith('cpu_per_req__mean') for n in names), scaler=scaler, pca_all=pca_all, detector=detectors[primary], detector_key=primary,
        detector_threshold=float(thr[primary]), classifier=xgb7, classes=CLASSES,
        cluster_pca=deploy["pca"], kmeans=deploy["kmeans"], cluster_info=desc,
        pca_seq=pca_seq, p_mu=p_mu, p_sd=p_sd, ew_threshold=float(ew_thr[primary_ew]), ew_calib=calib[primary_ew],
        ew_hidden=lstm.hidden, rca_params=params, templates=templates,
        score_ref=dict(mean=float(scores_all[primary][normal_tr].mean()), std=float(scores_all[primary][normal_tr].std()),
                       p99=float(np.quantile(scores_all[primary][normal_tr], 0.99))),
        cluster_dist_p99=float(np.quantile(
            np.linalg.norm(deploy["P"] - deploy["kmeans"].cluster_centers_[deploy["kmeans"].labels_], axis=1), 0.99)),
    ), artifact_dir / "bundle.joblib", compress=3)
    lstm.save(artifact_dir / "lstm.npz")
    res["meta"]["train_seconds"] = round(time.time() - t_start, 1)
    with open(artifact_dir / "results.json", "w") as fh:
        json.dump(_py(res), fh)
    log(f"done in {res['meta']['train_seconds']} s")
    return res
