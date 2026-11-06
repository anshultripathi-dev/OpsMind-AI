"""Human-readable names for features (used in SHAP drivers and cluster descriptions)."""
from __future__ import annotations

import re

from .config import COMPONENT_LABEL, FAULTS, METRIC_INFO

STAT_LABEL = {"mean": "", "std": "variability of ", "min": "minimum ", "max": "peak ", "delta": "trend of "}


def feature_label(name: str, templates: dict[int, str] | None = None) -> str:
    if name.startswith("log."):
        _, svc, rest = name.split(".", 2)
        svc_l = COMPONENT_LABEL.get(svc, svc)
        if rest.startswith("n_"):
            return f"{svc_l} {rest[2:]} log lines"
        tid = int(rest[1:])
        tpl = (templates or {}).get(tid, f"template {tid}")
        tpl = re.sub(r"<NUM>", "#", tpl)
        tpl = re.sub(r"<\*>", "*", tpl)
        return f"{svc_l} log: “{tpl}”"
    col, stat = name.split("__")
    comp, metric = col.split(".", 1)
    label = METRIC_INFO.get(metric, (metric, ""))[0]
    return f"{STAT_LABEL.get(stat, '')}{COMPONENT_LABEL.get(comp, comp)} {label}".replace("  ", " ")


def short_feature_label(name: str, templates: dict[int, str] | None = None) -> str:
    """Compact variant: ``order.mem__mean`` -> ``order · memory``."""
    if name.startswith("log."):
        return feature_label(name, templates)
    col, stat = name.split("__")
    comp, metric = col.split(".", 1)
    stat_s = {"mean": "", "std": " σ", "min": " min", "max": " peak", "delta": " Δ"}[stat]
    return f"{comp} · {METRIC_INFO.get(metric, (metric,))[0].lower()}{stat_s}"


def fault_label(ftype: str | None) -> str:
    if ftype in (None, "normal"):
        return "Normal"
    return FAULTS[ftype]["label"]
