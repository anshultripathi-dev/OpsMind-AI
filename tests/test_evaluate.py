import numpy as np
import pandas as pd

from opsmind.evaluate import alert_evaluation, alert_intervals, persist_flags, prf
from opsmind.features import NormalScaler, label_windows, sequence_index, window_ends


def _df(n=60, run="r1"):
    tw = np.arange(n) * 10.0 + 30
    return pd.DataFrame(dict(run_id=run, tw=tw, anomaly=0, grace=0))


TRUTH = {"r1": dict(fault_type="db_slowdown", fault_start=200.0, fault_end=400.0, impact_start=300.0)}


def test_prf():
    r = prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


def test_persist_requires_consecutive_windows():
    f = np.array([1, 0, 1, 1, 1, 0], bool)
    out = persist_flags(f, np.array(["a"] * 6), 3)
    assert out.tolist() == [False, False, False, False, True, False]


def test_alert_open_before_fault_still_counts_as_warning():
    df = _df()
    flags = (df.tw >= 150) & (df.tw < 330)          # alert opens before the fault and stays open
    ev = alert_evaluation(df, flags.to_numpy(), TRUTH, persist=1)
    assert ev["warned"] == 1 and ev["false_alarms"] == 0
    assert ev["leads"][0] == 100.0                   # impact 300 - fault start 200


def test_late_and_false_alarm_accounting():
    df = _df()
    late = ((df.tw >= 320) & (df.tw < 360)).to_numpy()
    ev = alert_evaluation(df, late, TRUTH, persist=1)
    assert ev["late"] == 1 and ev["warned"] == 0
    early_fp = (df.tw < 80).to_numpy()
    ev = alert_evaluation(df, early_fp, TRUTH, persist=1)
    assert ev["false_alarms"] == 1 and ev["missed"] == 1


def test_labels_come_only_from_injector_log():
    ends = window_ends(360)
    lab = label_windows(dict(fault_type="x", fault_start=600.0, fault_end=900.0, impact_start=700.0), ends)
    assert lab["anomaly"].sum() > 25 and lab["anomaly"][lab["tw"] < 600].sum() == 0
    assert lab["warn"].sum() > 0 and lab["warn"][lab["tw"] < 580].sum() == 0   # no warning label 2 min+ before impact
    none = label_windows(dict(fault_type=None), ends)
    assert not none["anomaly"].any() and none["warn_mask"].all()


def test_scaler_floors_zero_variance():
    X = np.zeros((100, 2), np.float32)
    Xa = np.concatenate([X, np.array([[0, 5.0]] * 10, np.float32)])
    z = NormalScaler().fit(X, Xa).transform(Xa)
    assert np.isfinite(z).all() and z.max() <= 25


def test_sequence_index_never_crosses_runs():
    run_start = np.array([0] * 5 + [5] * 5)
    idx = sequence_index(run_start, 10, seq_len=3, step=2)
    assert (idx[5:] >= 5).all() and idx[7, -1] == 7
