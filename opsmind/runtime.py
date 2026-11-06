"""Online inference: turns a stream of 5 s scrapes into per-window OpsMind verdicts.

One :class:`OpsMindRuntime` holds the trained models; every live session owns a
:class:`StreamState` (buffers + incident lifecycle). Feeding a scrape with
:meth:`OpsMindRuntime.ingest` returns a *tick* (a dict ready for the dashboard)
whenever a new 30 s window completes (every 10 s).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .baselines import ThresholdBaseline
from .config import (ARTIFACT_DIR, CLASSES, COMPONENTS, COMPONENT_LABEL, METRICS, RAW_COLUMNS, SAMPLE_SEC,
                     SEQ_LEN, SEQ_STEP_WINDOWS, SLO_ERR_RATE, SLO_LATENCY_MS, STRIDE_SAMPLES, WINDOW_SAMPLES)
from .features import single_window_features
from .logs import LogParser, LogVocab
from .models.earlywarning import NumpyLSTM
from .models.rootcause import RootCauseRanker
from .naming import fault_label

HIST = WINDOW_SAMPLES * (SEQ_LEN - 1) + WINDOW_SAMPLES + 2     # samples needed for the sequence model


@dataclass
class StreamState:
    cleaned: deque = field(default_factory=lambda: deque(maxlen=HIST))
    logvecs: deque = field(default_factory=lambda: deque(maxlen=HIST))
    P_hist: deque = field(default_factory=lambda: deque(maxlen=(SEQ_LEN - 1) * SEQ_STEP_WINDOWS + 1))
    last_valid: np.ndarray | None = None
    n: int = 0
    # alert hysteresis
    flag_run: int = 0
    ok_run: int = 0
    ew_run: int = 0
    base_run: int = 0
    slo_run: int = 0
    ew_active: bool = False
    base_active: bool = False
    last_warning_t: float | None = None
    last_baseline_t: float | None = None
    # incident lifecycle
    incident: dict | None = None
    incidents: list = field(default_factory=list)
    inc_counter: int = 0
    inc_votes: Counter = field(default_factory=Counter)
    inc_rca_sum: np.ndarray | None = None
    inc_rca_n: int = 0
    recent_ticks: deque = field(default_factory=lambda: deque(maxlen=6))


class OpsMindRuntime:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        b = joblib.load(artifact_dir / "bundle.joblib")
        self.b = b
        self.names: list[str] = b["names"]
        self.load_norm: bool = b.get("load_norm", False)
        self.scaler = b["scaler"]
        self.detector, self.det_thr = b["detector"], b["detector_threshold"]
        self.score_ref = b["score_ref"]
        self.clf = b["classifier"]
        self.classes: list[str] = b["classes"]
        self.cluster_pca, self.kmeans, self.cluster_info = b["cluster_pca"], b["kmeans"], b["cluster_info"]
        self.cluster_dist_p99 = b["cluster_dist_p99"]
        self.pca_seq, self.p_mu, self.p_sd = b["pca_seq"], b["p_mu"], b["p_sd"]
        self.ew_thr, (self.ew_a, self.ew_b) = b["ew_threshold"], b["ew_calib"]
        self.lstm = NumpyLSTM.load(artifact_dir / "lstm.npz")
        self.templates: dict[int, str] = b["templates"]
        self.parser = LogParser.load(artifact_dir / "drain.pkl")
        self.vocab: LogVocab = pd.read_pickle(artifact_dir / "log_vocab.pkl")
        self.baseline = ThresholdBaseline(self.names)
        self.rca = RootCauseRanker(self.names, self.clf, self.templates)
        self.rca.params = b["rca_params"]
        self.col_index = {c: i for i, c in enumerate(RAW_COLUMNS)}
        self.comp_cols = [np.flatnonzero(self.rca.comp_of_feat == c) for c in range(len(COMPONENTS))]
        self._tpl_cache: dict[str, int] = {}
        self.cluster_names = self._cluster_names()

    # -- helpers --------------------------------------------------------------------------
    def new_state(self) -> StreamState:
        return StreamState()

    def _cluster_names(self) -> dict[int, dict]:
        out = {}
        for c in self.cluster_info:
            sig = " · ".join(f"{s['label']} {'↑' if s['z'] > 0 else '↓'}" for s in c["signature"][:2])
            out[c["cluster"]] = dict(id=c["cluster"], name=f"Pattern {c['cluster'] + 1}", signature=sig,
                                     dominant=c["dominant"], purity=c["purity"], size=c["size"])
        return out

    def template_id(self, msg: str) -> int:
        v = self._tpl_cache.get(msg)
        if v is None:
            v = self._tpl_cache[msg] = self.parser.match(msg)
            if len(self._tpl_cache) > 200_000:
                self._tpl_cache.clear()
        return v

    def sigmoid_cal(self, logit: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.ew_a * logit + self.ew_b))))

    # -- main entry ---------------------------------------------------------------------------
    def ingest(self, st: StreamState, t: float, row: dict[str, float], logs: list[tuple[float, str, str, str]],
               truth: dict | None = None) -> dict | None:
        x = np.array([row[c] for c in RAW_COLUMNS], dtype=np.float32)
        if st.last_valid is None:
            st.last_valid = np.nan_to_num(x, nan=0.0)
        nan = np.isnan(x)
        x[nan] = st.last_valid[nan]
        st.last_valid = x.copy()
        parsed = [(lt, svc, lv, self.template_id(m)) for lt, svc, lv, m in logs]
        st.cleaned.append(x)
        st.logvecs.append(self.vocab.step_vector(parsed))
        st.n += 1
        if st.n < WINDOW_SAMPLES or st.n % STRIDE_SAMPLES != 0:
            return None
        return self._tick(st, t + SAMPLE_SEC, x, parsed, truth)

    def window_vector(self, st: StreamState) -> np.ndarray:
        buf = np.stack(list(st.cleaned)[-WINDOW_SAMPLES:])
        lg = np.stack(list(st.logvecs)[-WINDOW_SAMPLES:])
        return single_window_features(buf, lg, self.load_norm)

    # -- per window -----------------------------------------------------------------------------
    def _tick(self, st: StreamState, tw: float, last: np.ndarray, parsed_logs, truth) -> dict:
        x = self.window_vector(st)
        z = self.scaler.transform(x[None])[0]

        # module 3: anomaly detection
        score = float(self.detector.score(z[None])[0])
        flagged = score >= self.det_thr
        ref = self.score_ref
        ratio = (score - ref["mean"]) / max(self.det_thr - ref["mean"], 1e-9)
        st.flag_run = st.flag_run + 1 if flagged else 0

        # modules 4/5: incident classification
        proba = self.clf.predict_proba(z[None])[0]
        pred = int(proba.argmax())
        fault_cls = int(proba[1:].argmax()) + 1

        # module 6: failure-pattern assignment
        pz = self.cluster_pca.transform(z[None])
        cl = int(self.kmeans.predict(pz)[0])
        dist = float(np.linalg.norm(pz[0] - self.kmeans.cluster_centers_[cl]))
        pattern = dict(self.cluster_names.get(cl, dict(id=cl, name=f"Pattern {cl + 1}")),
                       distance=round(dist, 2), novel=bool(dist > self.cluster_dist_p99 * 1.25 and flagged))

        # module 7: early warning
        p = (self.pca_seq.transform(z[None])[0] - self.p_mu) / self.p_sd
        st.P_hist.append(p.astype(np.float32))
        hist = list(st.P_hist)
        seq = np.stack([hist[max(len(hist) - 1 - (SEQ_LEN - 1 - k) * SEQ_STEP_WINDOWS, 0)] for k in range(SEQ_LEN)])
        logit = float(self.lstm.forward(seq[None])[0])
        risk = self.sigmoid_cal(logit)
        st.ew_run = st.ew_run + 1 if risk >= self.ew_thr else 0
        ew_alert = st.ew_run >= 2
        if ew_alert and not st.ew_active:
            st.last_warning_t = tw
        st.ew_active = ew_alert

        # static-threshold baseline
        firing = self.baseline.firing(x)
        st.base_run = st.base_run + 1 if firing else 0
        base_alert = st.base_run >= 2
        if base_alert and not st.base_active:
            st.last_baseline_t = tw
        st.base_active = base_alert

        # per-component deviation from normal (for the topology view) and SLO state
        absz = np.abs(z)
        dev = {}
        for ci, c in enumerate(COMPONENTS):
            top = np.sort(absz[self.comp_cols[ci]])[-5:].mean()
            dev[c] = round(float(1.0 - np.exp(-top / 6.0)), 3)
        gi = {m: self.names.index(f"gateway.{m}__mean") for m in ("err_rate", "lat_p95")}
        slo_breach = bool(x[gi["err_rate"]] > SLO_ERR_RATE or x[gi["lat_p95"]] > SLO_LATENCY_MS)
        st.slo_run = st.slo_run + 1 if slo_breach else 0
        impacted = st.slo_run >= 2

        # module 8: root-cause ranking (only when something looks wrong)
        rca = None
        if flagged or ew_alert or risk > 0.3:
            exp = self.rca.explain(z, fault_cls)
            rca = exp["ranking"]
            if flagged:
                vec = np.array([r["score"] for r in sorted(rca, key=lambda r: COMPONENTS.index(r["component"]))])
                st.inc_rca_sum = vec if st.inc_rca_sum is None else st.inc_rca_sum + vec
                st.inc_rca_n += 1

        # incident lifecycle ------------------------------------------------------------------------
        confirmed = flagged and pred != 0
        if confirmed:
            st.inc_votes[CLASSES[pred]] += 1
        st.ok_run = 0 if flagged else st.ok_run + 1
        self._update_incident(st, tw, flagged, confirmed, risk, impacted, rca)

        sample = {c: {m: round(float(last[self.col_index[f"{c}.{m}"]]), 3) for m in METRICS[c]} for c in COMPONENTS}
        tick = dict(
            type="tick", t=tw, sample=sample, deviation=dev,
            anomaly=dict(score=round(score, 4), threshold=round(self.det_thr, 4), ratio=round(ratio, 3),
                         flagged=bool(flagged), sustained=bool(st.flag_run >= 3)),
            classifier=dict(label=CLASSES[pred], name=fault_label(CLASSES[pred]), confidence=round(float(proba[pred]), 3),
                            probs={c: round(float(v), 3) for c, v in zip(CLASSES, proba)},
                            likely_fault=CLASSES[fault_cls], likely_fault_name=fault_label(CLASSES[fault_cls])),
            risk=dict(prob=round(risk, 4), threshold=round(self.ew_thr, 4), alert=bool(ew_alert)),
            baseline=dict(firing=firing, alert=bool(base_alert)),
            rca=rca, pattern=pattern,
            slo=dict(impacted=bool(impacted), err=round(float(x[gi["err_rate"]]), 2), lat=round(float(x[gi["lat_p95"]]), 1)),
            logs=[dict(t=round(lt, 1), service=svc, level=lv, template_id=tid,
                       msg=self._render(svc, lv, tid)) for lt, svc, lv, tid in parsed_logs if lv != "INFO"][-8:],
            incident=self._incident_view(st),
            truth=truth,
        )
        return tick

    def _render(self, svc, lv, tid) -> str:
        return self.templates.get(tid, "unparsed event").replace("<NUM>", "#").replace("<*>", "*")

    # -- incidents ---------------------------------------------------------------------------------
    def _update_incident(self, st, tw, flagged, confirmed, risk, impacted, rca):
        inc = st.incident
        if inc is None:
            if st.flag_run >= 3 and sum(st.inc_votes.values()) >= 2:
                st.inc_counter += 1
                warned = st.last_warning_t if st.last_warning_t is not None and tw - st.last_warning_t < 900 else None
                base = st.last_baseline_t if st.last_baseline_t is not None and tw - st.last_baseline_t < 900 else None
                st.incident = inc = dict(id=f"INC-{st.inc_counter:03d}", started_at=tw - 20, detected_at=tw,
                                         warning_at=warned, baseline_at=base, impact_at=tw if impacted else None,
                                         resolved_at=None, peak_risk=risk, type=None, rca=None, status="open")
        if inc is None:
            return
        inc["peak_risk"] = max(inc["peak_risk"], risk)
        if impacted and inc["impact_at"] is None:
            inc["impact_at"] = tw
        if inc["baseline_at"] is None and st.last_baseline_t is not None and st.last_baseline_t >= inc["started_at"] - 60:
            inc["baseline_at"] = st.last_baseline_t
        if st.inc_votes:
            inc["type"] = st.inc_votes.most_common(1)[0][0]
        if st.inc_rca_sum is not None and st.inc_rca_n:
            mean = st.inc_rca_sum / st.inc_rca_n
            order = np.argsort(-mean)
            inc["rca"] = [dict(component=COMPONENTS[i], score=round(float(mean[i] / (mean.sum() + 1e-9)), 3)) for i in order[:3]]
        if st.ok_run >= 6:
            inc["status"] = "resolved"
            inc["resolved_at"] = tw
            st.incidents.insert(0, inc)
            st.incident = None
            st.inc_votes = Counter()
            st.inc_rca_sum, st.inc_rca_n = None, 0

    def _incident_view(self, st) -> dict:
        cur = st.incident
        lst = ([cur] if cur else []) + st.incidents[:14]
        out = []
        for i in lst:
            lead = None
            if i["warning_at"] is not None and i["impact_at"] is not None:
                lead = round(i["impact_at"] - i["warning_at"], 1)
            base_lead = None
            if i["baseline_at"] is not None and i["impact_at"] is not None:
                base_lead = round(i["impact_at"] - i["baseline_at"], 1)
            out.append(dict(i, type_name=fault_label(i["type"]) if i["type"] else None,
                            lead_s=lead, baseline_lead_s=base_lead))
        return dict(current=out[0] if cur else None, history=out[1:] if cur else out)
