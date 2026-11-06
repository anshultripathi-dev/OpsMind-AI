"""Validate the Drain log-parsing stage on the public Loghub HDFS benchmark.

Drain is used as published (He et al., 2017): depth 4, similarity 0.5, with the standard HDFS
masks for block ids, IP addresses and numbers. We report *grouping accuracy* — the share of log
lines whose predicted template group is exactly one ground-truth event group — which is the
metric used in the Loghub / loghub-2.0 parsing benchmarks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from .config import ARTIFACT_DIR, DATA_DIR

URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv"
PATH = DATA_DIR / "loghub" / "HDFS_2k.log_structured.csv"


def grouping_accuracy(truth: pd.Series, pred: pd.Series) -> float:
    df = pd.DataFrame({"t": truth.values, "p": pred.values})
    ok = 0
    pred_groups = df.groupby("p")["t"].agg(lambda s: set(s))
    truth_sizes = df["t"].value_counts()
    pred_sizes = df["p"].value_counts()
    for p, ts in pred_groups.items():
        if len(ts) == 1:
            t = next(iter(ts))
            if truth_sizes[t] == pred_sizes[p]:
                ok += int(pred_sizes[p])
    return ok / len(df)


def run(path: Path = PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — download {URL}")
    df = pd.read_csv(path)
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th, cfg.drain_depth, cfg.profiling_enabled = 0.5, 4, False
    cfg.masking_instructions = [
        MaskingInstruction(r"blk_(|-)[0-9]+", "BLK"),
        MaskingInstruction(r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)", "IP"),
        MaskingInstruction(r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$", "NUM"),
    ]
    miner = TemplateMiner(config=cfg)
    pred = [miner.add_log_message(m)["cluster_id"] for m in df["Content"]]
    out = dict(
        dataset="Loghub HDFS (2k sample, expert-labelled templates)",
        lines=int(len(df)),
        ground_truth_templates=int(df["EventId"].nunique()),
        drain_templates=int(len(miner.drain.clusters)),
        grouping_accuracy=round(grouping_accuracy(df["EventId"], pd.Series(pred)), 4),
        published_reference="Drain is commonly reported at ≈0.998 on HDFS in the Loghub-2k benchmark (check the paper before citing)",
    )
    (ARTIFACT_DIR / "loghub_validation.json").write_text(json.dumps(out, indent=1))
    return out
