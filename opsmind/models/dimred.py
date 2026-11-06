"""Module 2 — dimensionality reduction: PCA vs random projection, plus a 2-D system-state map."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.random_projection import GaussianRandomProjection


def fit_pca(Z: np.ndarray, variance: float = 0.95, max_components: int = 64, seed: int = 0) -> PCA:
    full = PCA(n_components=min(max_components, Z.shape[1]), svd_solver="randomized", random_state=seed).fit(Z)
    k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), variance) + 1)
    k = min(k, full.n_components_)
    return PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(Z)


def pca_report(Z: np.ndarray, seed: int = 0, max_components: int = 80) -> dict:
    full = PCA(n_components=min(max_components, Z.shape[1]), svd_solver="randomized", random_state=seed).fit(Z)
    cum = np.cumsum(full.explained_variance_ratio_)
    return dict(
        cumulative_variance=[round(float(x), 4) for x in cum[:40]],
        components_for={f"{int(v * 100)}%": int(np.searchsorted(cum, v) + 1) for v in (0.8, 0.9, 0.95, 0.99)},
        n_input_features=int(Z.shape[1]),
    )


def compare_projections(Z_tr, y_tr, Z_te, y_te, k: int, seed: int = 0) -> list[dict]:
    """Downstream fault-type macro-F1 and reconstruction error: full vs PCA vs random projection."""
    rows = []

    def downstream(A, B) -> float:
        clf = LogisticRegression(max_iter=300, C=1.0).fit(A, y_tr)
        return float(f1_score(y_te, clf.predict(B), average="macro"))

    rows.append(dict(method="Full features", dims=int(Z_tr.shape[1]), macro_f1=round(downstream(Z_tr, Z_te), 4),
                     recon_error=0.0))
    pca = PCA(n_components=k, random_state=seed).fit(Z_tr)
    rec = np.mean((Z_te - pca.inverse_transform(pca.transform(Z_te))) ** 2)
    rows.append(dict(method="PCA", dims=k, macro_f1=round(downstream(pca.transform(Z_tr), pca.transform(Z_te)), 4),
                     recon_error=round(float(rec), 4)))
    rp = GaussianRandomProjection(n_components=k, random_state=seed).fit(Z_tr)
    rec = np.mean((Z_te - rp.inverse_transform(rp.transform(Z_te))) ** 2)
    rows.append(dict(method="Random projection", dims=k, macro_f1=round(downstream(rp.transform(Z_tr), rp.transform(Z_te)), 4),
                     recon_error=round(float(rec), 4)))
    return rows
