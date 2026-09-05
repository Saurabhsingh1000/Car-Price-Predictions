#!/usr/bin/env python3
"""
Entry-point script to evaluate the persisted model on the held-out test set.

Usage:
    python scripts/evaluate_model.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import evaluate_saved_model  # noqa: E402


def main() -> None:
    metrics = evaluate_saved_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
