"""Modules 4 & 5 — incident classification: single models vs ensembles."""
from __future__ import annotations

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def _lr(seed):
    return LogisticRegression(max_iter=500, C=0.5, random_state=seed)


def _svm(seed):
    return make_pipeline(PCA(40, random_state=seed), SVC(C=3.0, probability=True, random_state=seed))


def _knn(seed):
    return make_pipeline(PCA(24, random_state=seed), KNeighborsClassifier(15, weights="distance"))


def _nb(seed):
    return make_pipeline(PCA(24, random_state=seed), GaussianNB())


def _dt(seed):
    return DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=seed)


def _xgb(seed, n_classes):
    return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8,
                         colsample_bytree=0.5, tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
                         random_state=seed, objective="multi:softprob", num_class=n_classes)


def baseline_models(seed: int = 0) -> dict:
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": _lr(seed),
        "KNN": _knn(seed),
        "Naive Bayes": _nb(seed),
        "Decision Tree": _dt(seed),
        "SVM (RBF)": _svm(seed),
    }


def ensemble_models(groups: np.ndarray, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> dict:
    cv = list(GroupKFold(n_splits=3).split(X, y, groups))      # folds never split an experiment run
    return {
        "Random Forest": RandomForestClassifier(300, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": _xgb(seed, n_classes),
        "Voting (soft)": VotingClassifier(
            [("lr", _lr(seed)), ("svm", _svm(seed)), ("knn", _knn(seed)), ("dt", _dt(seed))], voting="soft"),
        "Stacking": StackingClassifier(
            [("lr", _lr(seed)), ("svm", _svm(seed)), ("knn", _knn(seed)), ("dt", _dt(seed)), ("nb", _nb(seed))],
            final_estimator=LogisticRegression(max_iter=500), cv=cv, n_jobs=1, passthrough=False),
    }


def fit_and_score(models: dict, Xtr, ytr, Xva, yva, Xte, yte, classes: list[str]) -> tuple[dict, list[dict]]:
    fitted, rows = {}, []
    labels = list(range(len(classes)))
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        train_s = time.time() - t0
        t0 = time.time()
        pte = m.predict(Xte)
        infer_ms = (time.time() - t0) / max(len(Xte), 1) * 1000
        pva = m.predict(Xva) if len(Xva) else None
        rows.append(dict(
            model=name,
            macro_f1=round(float(f1_score(yte, pte, average="macro", labels=labels, zero_division=0)), 4),
            val_macro_f1=(round(float(f1_score(yva, pva, average="macro", labels=labels, zero_division=0)), 4)
                          if pva is not None else None),
            accuracy=round(float(accuracy_score(yte, pte)), 4),
            train_seconds=round(train_s, 2),
            infer_ms_per_window=round(infer_ms, 4),
            per_class_f1={c: round(float(v), 4) for c, v in zip(
                classes, f1_score(yte, pte, average=None, labels=labels, zero_division=0))},
            confusion=confusion_matrix(yte, pte, labels=labels).tolist(),
        ))
        fitted[name] = m
    return fitted, rows
