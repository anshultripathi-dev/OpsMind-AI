"""Benchmark Drain on Loghub HDFS: ``python scripts/validate_loghub.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opsmind import loghub  # noqa: E402

if __name__ == "__main__":
    import json
    print(json.dumps(loghub.run(), indent=1, ensure_ascii=False))
