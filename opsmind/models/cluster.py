"""Module 6 — clustering: discover recurring failure patterns without labels."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from ..naming import short_feature_label


def _sil(P, labels, seed=0, n=3000):
    ok = labels >= 0
    if len(set(labels[ok])) < 2:
        return 0.0
    P, labels = P[ok], labels[ok]
    if len(P) > n:
        idx = np.random.default_rng(seed).choice(len(P), n, replace=False)
        P, labels = P[idx], labels[idx]
    return float(silhouette_score(P, labels))


def choose_k(P: np.ndarray, k_range=range(4, 13), seed: int = 0) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in k_range:
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(P)
        scores[k] = round(_sil(P, lab, seed), 4)
    return max(scores, key=scores.get), scores


def _scores(P, pred, truth, seed) -> dict:
    n_clusters = len(set(pred[pred >= 0]))
    return dict(
        n_clusters=n_clusters,
        noise_fraction=round(float((pred < 0).mean()), 4),
        silhouette=round(_sil(P, pred, seed), 4),
        ari=round(float(adjusted_rand_score(truth, pred)), 4),
        nmi=round(float(normalized_mutual_info_score(truth, pred)), 4),
    )


def knee_eps(P: np.ndarray, min_samples: int = 12, quantile: float = 0.9) -> float:
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(P).kneighbors(P)
    return float(np.quantile(d[:, -1], quantile))


def run_clustering(Z: np.ndarray, truth: np.ndarray, seed: int = 0, n_pca: int = 20, max_hier: int = 3500) -> dict:
    """Cluster ``Z`` (scaled features of flagged windows) and score against ``truth`` (string labels)."""
    rng = np.random.default_rng(seed)
    pca = PCA(n_pca, random_state=seed).fit(Z)
    P = pca.transform(Z)
    k, k_scores = choose_k(P, seed=seed)

    km = KMeans(k, n_init=10, random_state=seed).fit(P)
    results = {"K-means": _scores(P, km.labels_, truth, seed) | dict(k=k)}

    sub = np.arange(len(P)) if len(P) <= max_hier else np.sort(rng.choice(len(P), max_hier, replace=False))
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(P[sub])
    results["Hierarchical (Ward)"] = _scores(P[sub], ag, truth[sub], seed) | dict(k=k, n_points=int(len(sub)))

    eps = knee_eps(P)
    db = DBSCAN(eps=eps, min_samples=12).fit_predict(P)
    results["DBSCAN"] = _scores(P, db, truth, seed) | dict(eps=round(eps, 3))

    rand = rng.integers(0, k, len(P))
    results["Random assignment"] = _scores(P, rand, truth, seed) | dict(k=k)
    return dict(pca=pca, P=P, k=k, k_scores=k_scores, kmeans=km, results=results,
                labels={"kmeans": km.labels_, "dbscan": db}, subsample_hier=sub, hier_labels=ag)


def describe_clusters(Z: np.ndarray, labels: np.ndarray, truth: np.ndarray, names: list[str],
                      templates: dict[int, str] | None = None, top: int = 3) -> list[dict]:
    """Signature (strongest deviating features) + purity vs the true fault label of every cluster."""
    out = []
    for c in sorted(set(labels[labels >= 0])):
        m = labels == c
        mean_z = Z[m].mean(0)
        # keep one stat per metric for readability: prefer the strongest of mean/max/delta
        order = np.argsort(-np.abs(mean_z))
        seen, sig = set(), []
        for j in order:
            n = names[j]
            key = n.split("__")[0] if "__" in n else n
            if key in seen:
                continue
            seen.add(key)
            sig.append(dict(feature=n, label=short_feature_label(n, templates),
                            z=round(float(mean_z[j]), 2)))
            if len(sig) == top:
                break
        vals, counts = np.unique(truth[m], return_counts=True)
        dom = int(np.argmax(counts))
        out.append(dict(cluster=int(c), size=int(m.sum()), dominant=str(vals[dom]),
                        purity=round(float(counts[dom] / m.sum()), 3),
                        composition={str(v): int(n) for v, n in zip(vals, counts)}, signature=sig))
    return out
