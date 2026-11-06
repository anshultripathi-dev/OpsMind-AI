"""Module 3 — unsupervised anomaly detection (trained on normal windows only)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class IForestDetector:
    """Isolation Forest on a PCA projection of the normal feature space (module 2 -> module 3)."""
    name = "Isolation Forest"

    def __init__(self, k: int = 32, seed: int = 0):
        self.k, self.seed = k, seed

    def fit(self, Z: np.ndarray) -> "IForestDetector":
        self.pca = PCA(n_components=self.k, random_state=self.seed).fit(Z)
        self.m = IsolationForest(n_estimators=300, max_samples=512, random_state=self.seed, n_jobs=-1)
        self.m.fit(self.pca.transform(Z))
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        return -self.m.score_samples(self.pca.transform(Z))


class OCSVMDetector:
    name = "One-Class SVM"

    def __init__(self, k: int = 24, nu: float = 0.02, max_fit: int = 6000, seed: int = 0):
        self.k, self.nu, self.max_fit, self.seed = k, nu, max_fit, seed

    def fit(self, Z: np.ndarray) -> "OCSVMDetector":
        self.pca = PCA(n_components=self.k, random_state=self.seed).fit(Z)
        P = self.pca.transform(Z)
        self.sd = P.std(0) + 1e-6
        rng = np.random.default_rng(self.seed)
        sub = P[rng.choice(len(P), min(self.max_fit, len(P)), replace=False)] / self.sd
        self.m = OneClassSVM(nu=self.nu, gamma="scale", kernel="rbf").fit(sub)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        return -self.m.decision_function(self.pca.transform(Z) / self.sd)


class PCAReconDetector:
    """Squared prediction error of a PCA model of normal behaviour."""
    name = "PCA reconstruction error"

    def __init__(self, variance: float = 0.95, seed: int = 0):
        self.variance, self.seed = variance, seed

    def fit(self, Z: np.ndarray) -> "PCAReconDetector":
        self.pca = PCA(n_components=self.variance, svd_solver="full", random_state=self.seed).fit(Z)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        return np.mean((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2, axis=1)


DETECTORS = {"iforest": IForestDetector, "ocsvm": OCSVMDetector, "pca_recon": PCAReconDetector}
