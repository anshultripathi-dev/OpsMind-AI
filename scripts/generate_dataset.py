"""Generate the labelled dataset: ``python scripts/generate_dataset.py``."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opsmind.dataset import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-fault", type=int, default=35, help="fault episodes per fault type")
    ap.add_argument("--normal", type=int, default=44, help="normal 30-minute runs")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate(a.per_fault, a.normal, a.seed)
