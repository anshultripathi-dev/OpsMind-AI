"""Module 8 — root-cause ranking: SHAP attribution mapped to services + dependency-graph walk.

Step 1  SHAP values of the incident classifier show which features drove the
        prediction; every feature belongs to one component, so the positive
        SHAP mass is aggregated per component.
Step 2  Among components that show anomalies, the one closest to the *provider*
        side of the dependency graph (the one others depend on) is boosted,
        and components whose own dependencies are anomalous are discounted,
        because failures spread from providers to their callers.
The result is a ranked list of *likely* causes, not a proven causal one.
"""
from __future__ import annotations

import itertools

import numpy as np
import shap

from ..config import COMPONENTS, EDGES, column_component
from ..naming import feature_label


def _closure(edges: list[tuple[str, str]]) -> tuple[dict[str, set], dict[str, set]]:
    callees: dict[str, set] = {c: set() for c in COMPONENTS}
    for a, b in edges:
        callees[a].add(b)
    desc = {c: set() for c in COMPONENTS}
    for c in COMPONENTS:
        stack = list(callees[c])
        while stack:
            n = stack.pop()
            if n not in desc[c]:
                desc[c].add(n)
                stack.extend(callees[n])
    anc = {c: {a for a in COMPONENTS if c in desc[a]} for c in COMPONENTS}
    return anc, desc


class RootCauseRanker:
    DEFAULT = dict(alpha=0.5, tau=0.6, lam=1.0, mu=0.5)

    def __init__(self, feature_names: list[str], xgb_model, templates: dict[int, str] | None = None):
        self.names = feature_names
        self.templates = templates
        self.comp_of_feat = np.array([COMPONENTS.index(column_component(n)) for n in feature_names])
        self.explainer = shap.TreeExplainer(xgb_model)
        self.anc, self.desc = _closure(EDGES)
        self.params = dict(self.DEFAULT)
        self.anc_mat = np.array([[1.0 if a in self.anc[c] else 0.0 for a in COMPONENTS] for c in COMPONENTS])
        self.desc_mat = np.array([[1.0 if d in self.desc[c] else 0.0 for d in COMPONENTS] for c in COMPONENTS])
        self.anc_n = np.maximum(self.anc_mat.sum(1), 1)
        self.desc_n = np.maximum(self.desc_mat.sum(1), 1)

    # -- evidence ----------------------------------------------------------------------
    def shap_for_class(self, Z: np.ndarray, cls: np.ndarray) -> np.ndarray:
        sv = self.explainer.shap_values(Z)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        return np.take_along_axis(sv, cls[:, None, None], axis=2)[..., 0]       # (n, F)

    def evidence(self, Z: np.ndarray, cls: np.ndarray) -> dict[str, np.ndarray]:
        contrib = self.shap_for_class(Z, cls)
        pos = np.clip(contrib, 0, None)
        n_comp = len(COMPONENTS)
        shap_c = np.zeros((len(Z), n_comp))
        z_c = np.zeros((len(Z), n_comp))
        absZ = np.abs(Z)
        for c in range(n_comp):
            cols = np.flatnonzero(self.comp_of_feat == c)
            shap_c[:, c] = pos[:, cols].sum(1)
            top = np.sort(absZ[:, cols], axis=1)[:, -5:]
            z_c[:, c] = top.mean(1)
        return dict(shap=shap_c, z=z_c, contrib=contrib)

    # -- combination -------------------------------------------------------------------------
    def combine(self, shap_c: np.ndarray, z_c: np.ndarray, params: dict | None = None) -> np.ndarray:
        p = params or self.params
        e = 1.0 - np.exp(-z_c / 6.0)                                    # deviation evidence in [0, 1)
        bn = shap_c / (shap_c.sum(1, keepdims=True) + 1e-9)
        en = e / (e.sum(1, keepdims=True) + 1e-9)
        r = p["alpha"] * bn + (1 - p["alpha"]) * en
        anomalous = (e >= p["tau"]).astype(float)
        anc_frac = (anomalous @ self.anc_mat.T) / self.anc_n            # share of my callers that are anomalous
        desc_frac = (anomalous @ self.desc_mat.T) / self.desc_n         # share of my dependencies that are anomalous
        return r * (1 + p["lam"] * anc_frac) * (1 - p["mu"] * desc_frac)

    def tune(self, ev: dict, truth_idx: np.ndarray) -> dict:
        grid = itertools.product([0.0, 0.3, 0.5, 0.7, 1.0], [0.5, 0.7, 0.85], [0.0, 0.5, 1.0, 2.0], [0.0, 0.3, 0.6, 0.9])
        best, best_p = -1.0, dict(self.DEFAULT)
        for a, t, l, m in grid:
            p = dict(alpha=a, tau=t, lam=l, mu=m)
            s = self.combine(ev["shap"], ev["z"], p)
            rank = np.argsort(-s, axis=1)
            top1 = (rank[:, 0] == truth_idx).mean()
            top3 = (rank[:, :3] == truth_idx[:, None]).any(1).mean()
            obj = top1 + 0.5 * top3
            if obj > best:
                best, best_p = obj, p
        self.params = best_p
        return best_p

    # -- online explanation ---------------------------------------------------------------------
    def explain(self, z: np.ndarray, cls: int, top_drivers: int = 5) -> dict:
        ev = self.evidence(z[None, :], np.array([cls]))
        scores = self.combine(ev["shap"], ev["z"])[0]
        norm = scores / (scores.sum() + 1e-9)
        contrib = ev["contrib"][0]
        ranking = []
        for c in np.argsort(-scores):
            cols = np.flatnonzero(self.comp_of_feat == c)
            order = cols[np.argsort(-contrib[cols])][:top_drivers]
            drivers = [dict(feature=self.names[j], label=feature_label(self.names[j], self.templates),
                            shap=round(float(contrib[j]), 4), z=round(float(z[j]), 2))
                       for j in order if contrib[j] > 0]
            ranking.append(dict(component=COMPONENTS[c], score=round(float(norm[c]), 4),
                                evidence=round(float(1.0 - np.exp(-ev["z"][0, c] / 6.0)), 3),
                                drivers=drivers))
        return dict(ranking=ranking)


def topk_accuracy(scores: np.ndarray, truth_idx: np.ndarray, ks=(1, 3)) -> dict:
    rank = np.argsort(-scores, axis=1)
    return {f"top{k}": round(float((rank[:, :k] == truth_idx[:, None]).any(1).mean()), 4) for k in ks}
