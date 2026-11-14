"""Portable location of RepoPremium data (override with REPO_PREMIUM_DATA)."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("REPO_PREMIUM_DATA", Path(__file__).resolve().parents[1] / "data"))
