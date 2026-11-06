"""Train and evaluate all modules: ``python scripts/train.py`` (run generate_dataset.py first)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

from opsmind.pipeline import train_all  # noqa: E402

if __name__ == "__main__":
    train_all()
